import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)


# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv(ENV_PATH)

EMBEDDING_MODEL = "models/gemini-embedding-001"


def validate_env() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(".env에 GOOGLE_API_KEY가 없습니다.")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"data 폴더를 찾을 수 없습니다: {DATA_DIR}"
        )


def load_documents() -> List:
    docs = []

    print("📁 data 폴더의 파일들을 확장자별로 읽어옵니다...\n")

    for filename in os.listdir(DATA_DIR):
        file_path = DATA_DIR / filename
        ext = file_path.suffix.lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                loaded = loader.load()
                docs.extend(loaded)
                print(f"✅ [PDF] {filename} 로드 완료 ({len(loaded)}개 문서)")

            elif ext == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded = loader.load()
                docs.extend(loaded)
                print(f"✅ [TXT] {filename} 로드 완료 ({len(loaded)}개 문서)")

            elif ext == ".xlsx":
                loader = UnstructuredExcelLoader(str(file_path))
                loaded = loader.load()
                docs.extend(loaded)
                print(f"✅ [EXCEL] {filename} 로드 완료 ({len(loaded)}개 문서)")

            elif ext == ".hwp":
                print(f"⚠️ [HWP] {filename} -> PDF로 변환 후 넣어주세요.")

            else:
                print(f"⏩ [SKIP] {filename} -> 지원하지 않는 확장자입니다.")

        except Exception as e:
            print(f"❌ {filename} 읽기 실패 (에러: {e})")

    print(f"\n📄 총 {len(docs)}개의 원본 문서를 불러왔습니다.")
    return docs


def preview_docs(docs: List, limit: int = 10) -> None:
    print("\n=== 원본 docs metadata 확인 ===")
    print("총 docs 개수:", len(docs))

    for i, doc in enumerate(docs[:limit]):
        print(f"\n--- docs[{i}] ---")
        print("metadata:", doc.metadata)
        print("content preview:", repr(doc.page_content[:120]))


def split_documents(docs: List) -> List:
    print("\n✂️ 문서를 조각내는 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ 총 {len(splits)}개의 청크 생성 완료")
    return splits


def build_vector_db(splits: List, reset: bool = False) -> None:
    if reset and CHROMA_DIR.exists():
        print(f"\n🗑️ 기존 Chroma DB 삭제 중: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    print("\n🧠 Vector DB를 생성하는 중...")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"✅ Vector DB 구축 완료: {CHROMA_DIR}")


def main():
    validate_env()

    print("기존 chroma_db를 삭제하고 새로 만들까요?")
    reset_input = input("(y/n) > ").strip().lower()
    reset = reset_input == "y"

    docs = load_documents()

    if not docs:
        raise ValueError("로드된 문서가 없습니다. data 폴더 내용을 확인하세요.")

    preview_docs(docs)
    splits = split_documents(docs)
    build_vector_db(splits, reset=reset)

    print("\n🎉 build_vector_db.py 실행 완료")


if __name__ == "__main__":
    main()