import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

# --------------------------------------------------
# 경로 설정
# graph/build_graph.py 기준으로 프로젝트 루트 계산
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "notices_graph.json"
ENV_PATH = BASE_DIR / ".env"

# --------------------------------------------------
# 환경변수 로드
# .env 예시:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=capstone1
# --------------------------------------------------
load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def validate_env() -> None:
    missing = []
    if not NEO4J_URI:
        missing.append("NEO4J_URI")
    if not NEO4J_USERNAME:
        missing.append("NEO4J_USERNAME")
    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")

    if missing:
        raise ValueError(
            f".env에 다음 값이 없습니다: {', '.join(missing)}"
        )


def load_notices(data_path: Path) -> list[dict]:
    if not data_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 최상위 구조는 리스트(list)여야 합니다.")

    required_keys = {"id", "title", "text", "topics", "keywords", "related_ids"}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{i}번째 항목이 dict가 아닙니다.")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(
                f"{i}번째 항목에 필수 키가 없습니다: {', '.join(sorted(missing))}"
            )

    return data


def create_constraints(tx) -> None:
    tx.run("""
    CREATE CONSTRAINT notice_id_unique IF NOT EXISTS
    FOR (n:Notice)
    REQUIRE n.id IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT topic_name_unique IF NOT EXISTS
    FOR (t:Topic)
    REQUIRE t.name IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT keyword_name_unique IF NOT EXISTS
    FOR (k:Keyword)
    REQUIRE k.name IS UNIQUE
    """)


def upsert_notice(tx, item: dict) -> None:
    tx.run(
        """
        MERGE (n:Notice {id: $id})
        SET n.title = $title,
            n.text = $text
        """,
        id=item["id"],
        title=item["title"],
        text=item["text"],
    )


def add_topics(tx, notice_id: str, topics: list[str]) -> None:
    for topic in topics:
        if not topic:
            continue

        tx.run(
            """
            MATCH (n:Notice {id: $notice_id})
            MERGE (t:Topic {name: $topic})
            MERGE (n)-[:ABOUT]->(t)
            """,
            notice_id=notice_id,
            topic=topic,
        )


def add_keywords(tx, notice_id: str, keywords: list[str]) -> None:
    for keyword in keywords:
        if not keyword:
            continue

        tx.run(
            """
            MATCH (n:Notice {id: $notice_id})
            MERGE (k:Keyword {name: $keyword})
            MERGE (n)-[:MENTIONS]->(k)
            """,
            notice_id=notice_id,
            keyword=keyword,
        )


def add_related_links(tx, source_id: str, target_ids: list[str]) -> None:
    for target_id in target_ids:
        if not target_id or source_id == target_id:
            continue

        tx.run(
            """
            MATCH (a:Notice {id: $source_id})
            MATCH (b:Notice {id: $target_id})
            MERGE (a)-[:RELATED_TO]->(b)
            """,
            source_id=source_id,
            target_id=target_id,
        )


def count_graph(tx) -> dict:
    result = tx.run(
        """
        MATCH (n:Notice)
        WITH count(n) AS notice_count
        MATCH (t:Topic)
        WITH notice_count, count(t) AS topic_count
        MATCH (k:Keyword)
        WITH notice_count, topic_count, count(k) AS keyword_count
        MATCH ()-[r:RELATED_TO]->()
        RETURN notice_count, topic_count, keyword_count, count(r) AS related_count
        """
    )
    return result.single().data()


def main() -> None:
    print("그래프 적재를 시작합니다...")

    validate_env()
    notices = load_notices(DATA_PATH)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    try:
        with driver.session() as session:
            print("제약조건 생성 중...")
            session.execute_write(create_constraints)

            print("Notice 노드 생성 중...")
            for item in notices:
                session.execute_write(upsert_notice, item)

            print("Topic / Keyword 관계 생성 중...")
            for item in notices:
                session.execute_write(add_topics, item["id"], item.get("topics", []))
                session.execute_write(add_keywords, item["id"], item.get("keywords", []))

            print("RELATED_TO 관계 생성 중...")
            for item in notices:
                session.execute_write(
                    add_related_links,
                    item["id"],
                    item.get("related_ids", []),
                )

            stats = session.execute_read(count_graph)

        print("\n적재 완료")
        print(f"- Notice: {stats['notice_count']}")
        print(f"- Topic: {stats['topic_count']}")
        print(f"- Keyword: {stats['keyword_count']}")
        print(f"- RELATED_TO: {stats['related_count']}")

    except Neo4jError as e:
        print(f"Neo4j 작업 중 오류 발생: {e}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    main()