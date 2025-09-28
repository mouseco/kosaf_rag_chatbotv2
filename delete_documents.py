"""
문서 삭제 관리 스크립트 (v2: 성능 최적화 적용)

개선점:
- 여러 문서를 삭제할 때 개별적으로 API를 호출하지 않고,
  ID 목록을 사용하여 임베딩, 스토리지, 문서를 한 번에 삭제 (Bulk Deletion)
- API 호출 횟수를 획기적으로 줄여 성능 및 안정성 향상
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any
from rich import print # 가독성을 위해 rich 라이브러리 사용

# --- 환경설정 ---
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    print("[bold red]❌ 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.[/bold red]")
    exit(1)

assert SUPABASE_URL is not None
assert SUPABASE_SERVICE_KEY is not None

# --- 핵심 함수 ---

def list_all_documents(supabase: Client) -> List[Dict[str, Any]]:
    """모든 문서 목록을 조회합니다."""
    try:
        result = supabase.table('documents').select('id, title, file_path, created_at').order('created_at', desc=True).execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"❌ 문서 목록 조회 실패: {e}")
        return []

def bulk_delete_documents(supabase: Client, docs_to_delete: List[Dict[str, Any]]) -> bool:
    """
    [신규] 문서 목록을 받아 관련된 모든 데이터를 한 번에 삭제합니다.
    """
    if not docs_to_delete:
        return True

    doc_ids = [doc['id'] for doc in docs_to_delete]
    file_paths = [doc['file_path'] for doc in docs_to_delete if doc.get('file_path')]
    
    try:
        print(f"🗑️ 총 [bold yellow]{len(doc_ids)}[/bold yellow]개 문서의 데이터를 일괄 삭제합니다...")

        # 1. 임베딩 일괄 삭제
        print("   ✓ 임베딩 데이터 삭제 중...")
        supabase.table('document_embeddings').delete().in_('document_id', doc_ids).execute()

        # 2. Storage 파일 일괄 삭제
        if file_paths:
            print("   ✓ Storage 파일 삭제 중...")
            try:
                supabase.storage.from_('documents').remove(file_paths)
            except Exception as storage_error:
                print(f"   [yellow]⚠️ Storage 파일 일부 삭제 실패 (수동 확인 필요): {storage_error}[/yellow]")

        # 3. 문서 메타데이터 일괄 삭제 (가장 마지막에 수행)
        print("   ✓ 문서 정보 삭제 중...")
        supabase.table('documents').delete().in_('id', doc_ids).execute()

        print(f"✅ [bold green]일괄 삭제 완료![/bold green]")
        return True

    except Exception as e:
        print(f"❌ [bold red]일괄 삭제 중 심각한 오류 발생: {e}[/bold red]")
        return False

def interactive_delete_menu():
    """대화형 삭제 메뉴"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) # type: ignore
    except Exception as e:
        print(f"[bold red]❌ Supabase 클라이언트 초기화 실패: {e}[/bold red]")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("📋 [bold cyan]문서 삭제 관리 메뉴 (v2)[/bold cyan]")
        print("=" * 50)
        print("1. 문서 목록 조회")
        print("2. 특정 파일 삭제 (파일명으로)")
        print("3. 전체 문서 삭제 ([bold red]위험![/bold red])")
        print("4. 종료")
        print("=" * 50)
        
        choice = input("선택하세요 (1-4): ").strip()

        if choice == "1":
            docs = list_all_documents(supabase)
            if docs:
                print(f"\n📚 총 [bold green]{len(docs)}[/bold green]개 문서:")
                for i, doc in enumerate(docs, 1):
                    created = doc.get('created_at', '').split('T')[0]
                    print(f"  {i:2d}. [dim]{created}[/dim] - {doc['title']}")
            else:
                print("\n📂 등록된 문서가 없습니다.")

        elif choice == "2":
            print("\n삭제할 파일명들을 입력하세요 (여러 개일 경우 쉼표로 구분):")
            filenames_input = input(">> ").strip()
            if not filenames_input:
                continue

            filenames = [name.strip() for name in filenames_input.split(',')]
            
            try:
                # [개선] 여러 파일을 한 번에 조회
                docs_result = supabase.table('documents').select('id, title, file_path').in_('title', filenames).execute()
                
                if not docs_result.data:
                    print(f"🤷 '{', '.join(filenames)}' 파일을 찾을 수 없습니다.")
                    continue
                
                bulk_delete_documents(supabase, docs_result.data)

            except Exception as e:
                print(f"❌ 파일 조회 중 오류: {e}")

        elif choice == "3":
            print("\n[bold red]⚠️  경고: 모든 문서와 관련 데이터가 영구적으로 삭제됩니다![/bold red]")
            confirm = input("정말 모든 문서를 삭제하시겠습니까? 삭제를 원하시면 '전체삭제'라고 입력하세요: ").strip()
            
            if confirm == "전체삭제":
                all_docs = list_all_documents(supabase)
                if not all_docs:
                    print("📂 이미 모든 문서가 비어있습니다.")
                    continue
                bulk_delete_documents(supabase, all_docs)
            else:
                print("✅ 삭제가 취소되었습니다.")

        elif choice == "4":
            print("👋 프로그램을 종료합니다.")
            break

        else:
            print("❌ 잘못된 선택입니다. 다시 입력해주세요.")


if __name__ == "__main__":
    interactive_delete_menu()