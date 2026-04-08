import os
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_openai import ChatOpenAI
from langchain_upstage import ChatUpstage

from source_mapping import source_to_ids
from graph.graph_expand import GraphExpander, extract_query_keywords


# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

CHROMA_DIR = BASE_DIR / "chroma_db"

# --------------------------------------------------
# 모델 설정
# --------------------------------------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "gpt": "gpt-4o-mini",
    "upstage": "solar-pro3",
}


def normalize_model_name(provider: str, model_name: str) -> str:
    provider = provider.strip().lower()
    model_name = model_name.strip().lower()

    if provider == "upstage":
        aliases = {
            "solar pro 3": "solar-pro3",
            "solar-pro-3": "solar-pro3",
            "solar-3-pro": "solar-pro3",
            "solar-pro3": "solar-pro3",
            "solar pro 2": "solar-pro2",
            "solar-pro-2": "solar-pro2",
            "solar-2-pro": "solar-pro2",
            "solar-pro2": "solar-pro2",
            "solar mini": "solar-mini",
            "solar-mini": "solar-mini",
        }
        return aliases.get(model_name, model_name)

    return model_name


def validate_common_env() -> None:
    if not os.getenv("NEO4J_URI"):
        raise ValueError(".env에 NEO4J_URI가 없습니다.")

    if not os.getenv("NEO4J_USERNAME"):
        raise ValueError(".env에 NEO4J_USERNAME이 없습니다.")

    if not os.getenv("NEO4J_PASSWORD"):
        raise ValueError(".env에 NEO4J_PASSWORD가 없습니다.")

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            ".env에 GOOGLE_API_KEY가 없습니다. "
            "(현재 임베딩 모델이 Gemini라서 provider와 상관없이 필요합니다.)"
        )


def validate_provider_env(provider: str) -> None:
    provider = provider.strip().lower()

    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(".env에 GOOGLE_API_KEY가 없습니다.")

    elif provider == "gpt":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(".env에 OPENAI_API_KEY가 없습니다.")

    elif provider == "upstage":
        if not os.getenv("UPSTAGE_API_KEY"):
            raise ValueError(".env에 UPSTAGE_API_KEY가 없습니다.")

    else:
        raise ValueError(
            "지원하지 않는 provider입니다. 'gemini', 'gpt', 'upstage' 중 하나를 사용하세요."
        )


def format_vector_docs(docs) -> str:
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


def get_llm(provider: str, model_name: str):
    provider = provider.strip().lower()
    model_name = normalize_model_name(provider, model_name)

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0
        )

    if provider == "gpt":
        return ChatOpenAI(
            model=model_name,
            temperature=0
        )

    if provider == "upstage":
        return ChatUpstage(
            model=model_name,
            temperature=0
        )

    raise ValueError(
        "지원하지 않는 provider입니다. 'gemini', 'gpt', 'upstage' 중 하나를 사용하세요."
    )


def choose_provider_and_model() -> Tuple[str, str]:
    while True:
        provider = input("사용할 provider 입력 (gemini / gpt / upstage) > ").strip().lower()

        if provider not in DEFAULT_MODELS:
            print("지원하지 않는 provider입니다. 다시 입력하세요.\n")
            continue

        default_model = DEFAULT_MODELS[provider]
        model_name = input(
            f"모델명 입력 (엔터 시 기본값 사용: {default_model}) > "
        ).strip()

        if not model_name:
            model_name = default_model

        model_name = normalize_model_name(provider, model_name)
        validate_provider_env(provider)

        print(f"\n선택된 provider: {provider}")
        print(f"선택된 model: {model_name}\n")

        return provider, model_name


def get_unique_sources(docs) -> List[str]:
    unique_sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "")
        if source and source not in seen:
            seen.add(source)
            unique_sources.append(source)

    return unique_sources


def build_seed_ids_from_sources(sources: List[str]) -> List[str]:
    seed_ids = []

    for source in sources:
        mapped_ids = source_to_ids(source)
        seed_ids.extend(mapped_ids)

    return list(dict.fromkeys(seed_ids))


def answer_question(
    question: str,
    provider: str,
    model_name: str,
    vectorstore: Chroma,
    vector_k: int = 5,
    graph_limit: int = 10,
) -> Dict:
    llm = get_llm(provider, model_name)

    vector_docs = vectorstore.similarity_search(question, k=vector_k)
    unique_sources = get_unique_sources(vector_docs)
    seed_ids = build_seed_ids_from_sources(unique_sources)
    query_keywords = extract_query_keywords(question)

    expander = GraphExpander()
    try:
        graph_docs = expander.hybrid_graph_search(
            seed_ids=seed_ids,
            query_keywords=query_keywords,
            limit=graph_limit
        )
    finally:
        expander.close()

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
        "provider": provider,
        "model_name": model_name,
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
    print("provider:", result["provider"])
    print("model:", result["model_name"])
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
    validate_common_env()

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma DB 폴더를 찾을 수 없습니다: {CHROMA_DIR}\n"
            f"먼저 build_vector_db.py를 실행해서 벡터 DB를 생성하세요."
        )

    vectorstore = get_vectorstore()
    provider, model_name = choose_provider_and_model()

    print("Hybrid RAG 실행 준비 완료")
    print("종료하려면 exit 입력")
    print("모델을 다시 고르려면 :model 입력\n")

    while True:
        question = input("질문 입력 > ").strip()

        if not question:
            print("질문이 비어 있습니다.\n")
            continue

        if question.lower() in {"exit", "quit"}:
            print("종료합니다.")
            break

        if question.lower() == ":model":
            provider, model_name = choose_provider_and_model()
            continue

        try:
            result = answer_question(
                question=question,
                provider=provider,
                model_name=model_name,
                vectorstore=vectorstore,
            )
            debug_print(result)
            print("\n")
        except Exception as e:
            print(f"\n오류 발생: {e}\n")


if __name__ == "__main__":
    main()