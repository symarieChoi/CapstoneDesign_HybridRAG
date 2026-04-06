import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from source_mapping import source_to_ids
from graph.graph_expand import GraphExpander, extract_query_keywords


# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

# 네가 다시 만든 Chroma 폴더명으로 맞춰줘
CHROMA_DIR = BASE_DIR / "chroma_db"

# --------------------------------------------------
# 모델 설정
# --------------------------------------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"


def validate_env() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(".env에 GOOGLE_API_KEY가 없습니다.")

    if not os.getenv("NEO4J_URI"):
        raise ValueError(".env에 NEO4J_URI가 없습니다.")

    if not os.getenv("NEO4J_USERNAME"):
        raise ValueError(".env에 NEO4J_USERNAME이 없습니다.")

    if not os.getenv("NEO4J_PASSWORD"):
        raise ValueError(".env에 NEO4J_PASSWORD가 없습니다.")


def format_vector_docs(docs) -> str:
    """Vector 검색 결과를 프롬프트용 문자열로 정리"""
    blocks = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        source_name = os.path.basename(source)

        blocks.append(
            f"[Vector 문서 {i}]\n"
            f"source: {source_name}\n"
            f"content:\n{doc.page_content}"
        )

    return "\n\n".join(blocks)


def format_graph_docs(graph_docs: List[Dict]) -> str:
    """Graph 검색 결과를 프롬프트용 문자열로 정리"""
    if not graph_docs:
        return "관련 그래프 문서 없음"

    blocks = []
    for i, item in enumerate(graph_docs, start=1):
        blocks.append(
            f"[Graph 문서 {i}]\n"
            f"id: {item.get('id', '')}\n"
            f"title: {item.get('title', '')}\n"
            f"text:\n{item.get('text', '')}"
        )

    return "\n\n".join(blocks)


def get_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )


def get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0
    )


def get_unique_sources(docs) -> List[str]:
    """검색 결과에서 source 중복 제거"""
    unique_sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "")
        if source and source not in seen:
            seen.add(source)
            unique_sources.append(source)

    return unique_sources


def build_seed_ids_from_sources(sources: List[str]) -> List[str]:
    """source -> JSON notice ids 변환"""
    seed_ids = []

    for source in sources:
        mapped_ids = source_to_ids(source)
        seed_ids.extend(mapped_ids)

    # 순서 유지 중복 제거
    return list(dict.fromkeys(seed_ids))


def answer_question(question: str, vector_k: int = 5, graph_limit: int = 10) -> Dict:
    """
    질문 1개에 대해
    - vector 검색
    - source -> seed_ids 변환
    - graph 확장
    - LLM 답변 생성
    을 수행
    """
    vectorstore = get_vectorstore()
    llm = get_llm()

    # 1. Vector 검색
    vector_docs = vectorstore.similarity_search(question, k=vector_k)

    # 2. 검색 결과 source 추출
    unique_sources = get_unique_sources(vector_docs)

    # 3. source -> seed_ids 변환
    seed_ids = build_seed_ids_from_sources(unique_sources)

    # 4. 질문에서 키워드 추출
    query_keywords = extract_query_keywords(question)

    # 5. Graph 검색 / 확장
    expander = GraphExpander()
    try:
        graph_docs = expander.hybrid_graph_search(
            seed_ids=seed_ids,
            query_keywords=query_keywords,
            limit=graph_limit
        )
    finally:
        expander.close()

    # 6. 프롬프트용 문맥 생성
    vector_context = format_vector_docs(vector_docs)
    graph_context = format_graph_docs(graph_docs)

    prompt = f"""
너는 경북대학교 컴퓨터학부 공지사항 기반 질의응답 챗봇이다.

반드시 아래 문맥에 근거해서만 답변하라.
문맥에 없는 사실은 추측하지 말고, 문맥만으로 확실하지 않으면 "제공된 자료만으로는 확실하지 않습니다."라고 답하라.

답변 원칙:
1. 질문에 직접 답하라.
2. 가능하면 핵심 조건, 기준, 절차를 항목별로 정리하라.
3. 서로 다른 문서가 연결된 경우 이를 자연스럽게 종합하라.
4. 서식 문서는 절차 보조 정보로만 활용하고, 기준/요건 문서는 우선적으로 반영하라.
5. 너무 장황하지 않게, 그러나 빠진 핵심이 없도록 답하라.

[질문]
{question}

[Vector 검색 결과]
{vector_context}

[Graph 확장 결과]
{graph_context}
"""

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "vector_docs": vector_docs,
        "unique_sources": unique_sources,
        "seed_ids": seed_ids,
        "query_keywords": query_keywords,
        "graph_docs": graph_docs,
    }


def debug_print(result: Dict) -> None:
    print("\n==============================")
    print("질문:", result["question"])
    print("==============================\n")

    print("[추출된 query keywords]")
    print(result["query_keywords"])

    print("\n[Vector 검색 source]")
    for s in result["unique_sources"]:
        print("-", os.path.basename(s))

    print("\n[Graph seed ids]")
    for sid in result["seed_ids"]:
        print("-", sid)

    print("\n[Graph 검색 결과]")
    if not result["graph_docs"]:
        print("없음")
    else:
        for item in result["graph_docs"]:
            print(f"- {item['id']} | {item['title']}")

    print("\n[최종 답변]")
    print(result["answer"])


def main():
    validate_env()

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma DB 폴더를 찾을 수 없습니다: {CHROMA_DIR}\n"
            f"CHROMA_DIR 경로를 확인하세요."
        )

    print("Hybrid RAG 실행 준비 완료")
    print("종료하려면 exit 입력\n")

    while True:
        question = input("질문 입력 > ").strip()

        if not question:
            print("질문이 비어 있습니다.\n")
            continue

        if question.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break

        try:
            result = answer_question(question)
            debug_print(result)
            print("\n")
        except Exception as e:
            print(f"\n오류 발생: {e}\n")


if __name__ == "__main__":
    main()