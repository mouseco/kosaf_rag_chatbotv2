"""
관리자용 PDF 일괄 업로드 스크립트 (v2: 성능 및 UX 개선)

개선점:
- 임베딩 저장 시 배치(Batch) 삽입을 적용하여 성능 극대화
- 'tqdm' 라이브러리로 시각적인 프로그레스 바 제공
- 'rich' 라이브러리로 더 깔끔하고 가독성 좋은 콘솔 출력
"""

import os
import warnings
import uuid
from pathlib import Path
import google.generativeai as genai
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List ,Dict
from tqdm import tqdm  
from rich import print  

# 환경변수 로드
load_dotenv()

# 설정
PDF_DIRECTORY = "pdf_files_to_upload"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # service_role 키 사용
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# 환경변수 검증
if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_API_KEY]):
    print("[bold red]❌ 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.[/bold red]")
    print("필요한 변수: SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_API_KEY")
    exit(1)

# 타입 검증을 위한 assert 추가
assert SUPABASE_URL is not None
assert SUPABASE_SERVICE_KEY is not None  
assert GOOGLE_API_KEY is not None

def extract_text_from_pdf(file_path: str) -> str:
    """PDF 파일에서 텍스트를 추출합니다."""
    try:
        extracted_text = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += f"\n\n--- 페이지 {page_num} ---\n\n"
                        extracted_text += page_text
        return extracted_text.strip()
    except Exception as e:
        raise Exception(f"PDF 텍스트 추출 중 오류 발생: {str(e)}")



# [교체] 새로운 청킹 함수
def split_legal_text_into_chunks(text: str, max_chunk_size: int = 2000) -> List[Dict]:
    """
    소제목 단위로 청킹하는 개선 버전.
    - 소제목 아래 모든 Q&A를 하나의 청크로 묶음.
    - 청크가 길면 재귀 분할 (오버랩 200자).
    - 반환: [{'content': str, 'metadata': dict}, ...]
    """
    
    # 1. 소제목 패턴 정의 (강화됨)
    section_patterns = [
        r'^(\d+\.\s*[가-힣\s·,]+)$',  # "1. 공통사항"
        r'^([가-힣\s·,]+?(?:부서\s*간|사업|예산|경비|공통|관리|업무|계약|집행|인사|제도|법률)[^\n]*?)$',  # 키워드 포함 소제목
    ]
    
    # 전체 텍스트에서 소제목 위치 찾기 (re.finditer로 위치 캡처)
    section_matches = []
    for pat in section_patterns:
        matches = list(re.finditer(pat, text, re.MULTILINE))
        section_matches.extend(matches)
    section_matches.sort(key=lambda m: m.start())  # 시작 위치 순 정렬
    
    # 2. 소제목으로 텍스트 분할 (섹션 간 경계)
    sections = []
    prev_end = 0
    for match in section_matches:
        section_title = match.group(1).strip()
        section_start = match.start()
        section_end = match.end()
        
        # 다음 섹션 시작 전까지의 내용 (Q&A 포함)
        next_start = section_matches[section_matches.index(match) + 1].start() if section_matches.index(match) + 1 < len(section_matches) else len(text)
        section_content = text[prev_end:next_start].strip()
        
        sections.append({
            'title': section_title,
            'content': section_content,
            'start_pos': prev_end,
            'end_pos': next_start
        })
        prev_end = next_start
    
    # 마지막 섹션 (fallback)
    if prev_end < len(text):
        sections.append({
            'title': '기타',
            'content': text[prev_end:].strip(),
            'start_pos': prev_end,
            'end_pos': len(text)
        })
    
    # 3. 각 섹션 내 Q&A 그룹화 및 청크 생성
    structured_chunks = []
    for sec in sections:
        section_text = sec['content']
        if not section_text:
            continue
        
        # 섹션 내 Q&A 패턴 매치 (전체 섹션 스캔)
        qa_pattern = r"(Q: [^\?]+\?)\s*(A: .+?)(?=(Q: |$))"
        qa_matches = list(re.finditer(qa_pattern, section_text, re.DOTALL | re.MULTILINE))
        
        if not qa_matches:
            # Q&A 없으면 섹션 전체를 청크로
            chunk_content = section_text
            qa_list = []
        else:
            # Q&A들을 하나의 문자열로 합침
            qa_contents = []
            qa_list = []
            for match in qa_matches:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                qa_contents.append(f"{question}\n{answer}")
                qa_list.append({'question': question, 'answer': answer})
            chunk_content = "\n\n".join(qa_contents)
        
        # 메타데이터 설정
        metadata = {
            'category': sec['title'],
            'qa_list': qa_list,  # 전체 Q&A 목록 (검색 시 상세 참조)
            'section_start': sec['start_pos'],
            'section_end': sec['end_pos']
        }
        
        structured_chunks.append({
            'content': chunk_content,
            'metadata': metadata
        })
    
    # 4. 재귀적 분할 (긴 청크만, max_chunk_size=2000으로 조정)
    final_chunks = []
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,  # 오버랩 증가로 맥락 유지
        separators=["\n\n--- Q&A --- \n\n", "\n\n", "\n", " ", ""]  # Q&A 경계 우선
    )
    
    for struct_chunk in structured_chunks:
        content = struct_chunk['content']
        metadata = struct_chunk['metadata']
        if len(content) <= max_chunk_size:
            final_chunks.append({'content': content, 'metadata': metadata})
        else:
            sub_chunks = recursive_splitter.split_text(content)
            for j, sub_content in enumerate(sub_chunks):
                sub_meta = metadata.copy()
                sub_meta['sub_chunk_index'] = j
                if j > 0:
                    sub_content = f"{metadata['category']} (계속 {j})\n{sub_content}"
                final_chunks.append({'content': sub_content, 'metadata': sub_meta})
    
    return [chunk for chunk in final_chunks if chunk.get('content', '').strip()]

def process_single_pdf(file_path: str, filename: str, supabase: Client) -> bool:
    """단일 PDF 파일을 처리합니다. (배치 처리 적용)"""
    try:
        print(f"\n--- 📄 [bold cyan]'{filename}'[/bold cyan] 처리 시작 ---")
        
        # 1. 기존 데이터 확인 및 삭제 (멱등성 보장)
        print("   ✓ 기존 데이터 확인 중...")
        existing_doc = supabase.table('documents').select('id, file_path').eq('title', filename).execute()
        
        if existing_doc.data:
            doc_id = existing_doc.data[0]['id']
            old_file_path = existing_doc.data[0].get('file_path')
            
            print(f"   ⚠️  기존 '{filename}' 발견 - 덮어쓰기 진행")
            
            # 기존 임베딩 삭제
            supabase.table('document_embeddings').delete().eq('document_id', doc_id).execute()
            
            # 기존 Storage 파일 삭제
            if old_file_path:
                try:
                    supabase.storage.from_('documents').remove([old_file_path])
                except:
                    pass  # Storage 삭제 실패는 무시
            
            # 기존 문서 삭제
            supabase.table('documents').delete().eq('id', doc_id).execute()
            print("   ✓ 기존 데이터 삭제 완료")

        # 2. 텍스트 추출
        print("   📝 PDF에서 텍스트 추출 중...")
        content = extract_text_from_pdf(file_path)
        
        if len(content.strip()) < 10:
            print("   ❌ 텍스트가 너무 짧습니다 - 건너뜀")
            return False

        print(f"   ✓ 텍스트 추출 완료 ([bold green]{len(content):,}[/bold green]자)")

        # [수정] 3. Storage에 저장할 안전한 파일명 생성
        print("   � Storage에 저장할 안전한 파일명 생성 중...")
        file_extension = Path(filename).suffix  # 원본 파일의 확장자 (예: '.pdf')
        safe_filename = f"{uuid.uuid4()}{file_extension}"
        storage_path = f"public/{safe_filename}"
        print(f"   ✓ 생성된 경로: [dim]{storage_path}[/dim]")

        # 4. Storage에 원본 파일 업로드 (생성된 safe_filename 사용)
        print("   📁 Storage에 파일 업로드 중...")
        with open(file_path, 'rb') as f:
            supabase.storage.from_('documents').upload(
                path=storage_path,
                file=f,
                file_options={"upsert": "true", "content-type": "application/pdf"}
            )
        
        print("   ✓ Storage 업로드 완료")

        # [수정] 5. 문서 메타데이터 저장 (title과 file_path 분리)
        print("   💾 문서 정보 DB 저장 중...")
        doc_result = supabase.table('documents').insert({
            'title': filename,          # 표시용 이름: 원본 파일명 사용
            'content': content,
            'file_path': storage_path,  # 저장용 경로: 새로 생성한 안전한 경로 사용
            'is_public': True,  # 모든 문서를 공용으로 설정
            # user_id 필드 제거 (NULL로 설정됨)
        }).execute()
        
        if not doc_result.data:
            raise Exception("문서 정보 저장 실패")
        
        document_id = doc_result.data[0]['id']
        print("   ✓ 문서 정보 저장 완료")

        # 5. 텍스트 청크 분할
        print("   ✂️  텍스트 청크 분할 중...")
        chunks = split_legal_text_into_chunks(content)
        chunks = [chunk for chunk in chunks if chunk.get('content', '').strip()]
        
        if not chunks:
            print("   ❌ 유효한 청크가 없습니다")
            return False
        
        print(f"   ✓ [bold green]{len(chunks)}[/bold green]개 청크 생성")

        # [수정] 6. 임베딩 생성 및 배치 리스트 준비
        print("   🧠 임베딩 생성 중 (배치 처리 준비)...")
        embeddings_to_insert = []
        
        # tqdm을 사용하여 프로그레스 바 생성
        for i, chunk_dict in enumerate(tqdm(chunks, desc="      생성 진행률", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")):
            try:
                chunk_content = chunk_dict['content']
                embedding_result = genai.embed_content(  # type: ignore
                    model="models/text-embedding-004",
                    content=chunk_content,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                embedding = embedding_result['embedding']
                
                # DB에 바로 insert하지 않고, 리스트에 추가
                embeddings_to_insert.append({
                    'document_id': document_id,
                    'chunk_index': i,
                    'content': chunk_content,
                    'embedding': embedding
                })
            except Exception as chunk_error:
                print(f"   [yellow]⚠️ 청크 {i+1} 임베딩 생성 실패: {str(chunk_error)[:100]}...[/yellow]")
                continue
        
        # [수정] 7. 준비된 임베딩 리스트를 DB에 한 번에 저장
        if embeddings_to_insert:
            print(f"   🚀 [bold green]{len(embeddings_to_insert)}[/bold green]개 임베딩을 DB에 일괄 저장 중...")
            supabase.table('document_embeddings').insert(embeddings_to_insert).execute()
            print(f"   🎉 [bold green]'{filename}'[/bold green] 처리 완료!")
            return True
        else:
            print(f"   ❌ [bold red]'{filename}' 처리 실패 - 유효한 임베딩이 없습니다.[/bold red]")
            return False
            
    except Exception as e:
        print(f"   ❌ [bold red]'{filename}' 처리 중 심각한 오류 발생: {e}[/bold red]")
        # 실패 시 생성된 문서 정보 롤백 (선택적)
        if 'document_id' in locals():
            try:
                supabase.table('documents').delete().eq('id', document_id).execute()
            except:
                pass
        return False

def main():
    """메인 실행 함수"""
    print("[bold green]🚀 PDF 일괄 업로드 스크립트 시작 (v2)[/bold green]")
    print("=" * 60)
    
    # 클라이언트 초기화
    print("🔧 Supabase 및 Google API 초기화 중...")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) # type: ignore
        genai.configure(api_key=GOOGLE_API_KEY)  # type: ignore
        print("✅ 초기화 완료")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    # PDF 폴더 확인
    pdf_dir = Path(PDF_DIRECTORY)
    if not pdf_dir.exists():
        print(f"📁 '{PDF_DIRECTORY}' 폴더가 없습니다. 폴더를 생성합니다...")
        pdf_dir.mkdir(exist_ok=True)
        print(f"📋 '{PDF_DIRECTORY}' 폴더에 PDF 파일을 넣고 다시 실행하세요.")
        return

    # PDF 파일 목록 가져오기
    pdf_files = [f for f in pdf_dir.iterdir() if f.suffix.lower() == '.pdf']
    
    if not pdf_files:
        print(f"📂 '{PDF_DIRECTORY}' 폴더에 PDF 파일이 없습니다.")
        print("📋 처리할 PDF 파일들을 폴더에 넣어주세요.")
        return

    print(f"🎯 총 [bold yellow]{len(pdf_files)}[/bold yellow]개의 PDF 파일 발견")
    print("📋 파일 목록:")
    for i, file_path in enumerate(pdf_files, 1):
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB로 변환
        print(f"  {i:2d}. {file_path.name} ([cyan]{file_size:.1f}MB[/cyan])")
    
    print("\n" + "=" * 60)
    
    # 처리 시작
    successful_files = 0
    failed_files = []
    
    for file_path in pdf_files:
        success = process_single_pdf(str(file_path), file_path.name, supabase)
        if success:
            successful_files += 1
        else:
            failed_files.append(file_path.name)

    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("[bold green]🏁 모든 작업 완료![/bold green]")
    print(f"   - ✅ [bold green]성공: {successful_files}개[/bold green]")
    print(f"   - ❌ [bold red]실패: {len(failed_files)}개[/bold red]")
    
    if failed_files:
        print("\n🔍 실패한 파일들:")
        for filename in failed_files:
            print(f"  - [red]{filename}[/red]")
        print("\n💡 실패한 파일들을 다시 확인해보세요:")
        print("  - PDF가 손상되었거나 암호로 보호된 파일인지 확인")
        print("  - 텍스트가 포함된 PDF인지 확인 (스캔본은 OCR 필요)")
    
    print(f"\n🎉 업로드된 [bold green]{successful_files}[/bold green]개 파일은 Streamlit 앱에서 확인할 수 있습니다!")

if __name__ == "__main__":
    main()