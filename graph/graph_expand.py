import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

# --------------------------------------------------
# 경로 설정
# graph/graph_expand.py 기준으로 프로젝트 루트 계산
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class GraphExpander:
    def __init__(self):
        if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
            raise ValueError("Neo4j 접속 정보가 .env에 없습니다.")

        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    # --------------------------------------------------
    # 1) seed_ids 기반 확장
    # 예: Vector 검색 결과 source -> notice ids 변환 후 사용
    # --------------------------------------------------
    def expand_related_notices(self, seed_ids: List[str], limit: int = 10) -> List[Dict]:
        if not seed_ids:
            return []

        query = """
        MATCH (n:Notice)
        WHERE n.id IN $seed_ids
        OPTIONAL MATCH (n)-[:RELATED_TO]-(m:Notice)
        WITH collect(DISTINCT n) + collect(DISTINCT m) AS nodes
        UNWIND nodes AS x
        WITH DISTINCT x
        WHERE x IS NOT NULL
        RETURN x.id AS id, x.title AS title, x.text AS text
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, seed_ids=seed_ids, limit=limit)
                return [record.data() for record in result]
        except Neo4jError as e:
            print(f"expand_related_notices 오류: {e}")
            return []

    # --------------------------------------------------
    # 2) 질문 키워드 기반 그래프 검색
    # 질문 안의 키워드가 Topic/Keyword 노드와 맞으면 공지 검색
    # --------------------------------------------------
    def search_by_keywords(self, query_keywords: List[str], limit: int = 10) -> List[Dict]:
        if not query_keywords:
            return []

        cypher = """
        MATCH (n:Notice)
        OPTIONAL MATCH (n)-[:ABOUT]->(t:Topic)
        OPTIONAL MATCH (n)-[:MENTIONS]->(k:Keyword)
        WITH n,
             collect(DISTINCT t.name) AS topics,
             collect(DISTINCT k.name) AS keywords
        WITH n, topics, keywords,
             [x IN $query_keywords WHERE x IN topics OR x IN keywords] AS matched
        WHERE size(matched) > 0
        RETURN n.id AS id,
               n.title AS title,
               n.text AS text,
               matched,
               size(matched) AS score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher, query_keywords=query_keywords, limit=limit)
                return [record.data() for record in result]
        except Neo4jError as e:
            print(f"search_by_keywords 오류: {e}")
            return []

    # --------------------------------------------------
    # 3) seed_ids + 질문 키워드 둘 다 활용
    # Hybrid에서 제일 쓰기 좋은 형태
    # --------------------------------------------------
    def hybrid_graph_search(
        self,
        seed_ids: List[str],
        query_keywords: List[str],
        limit: int = 15
    ) -> List[Dict]:
        seed_results = self.expand_related_notices(seed_ids, limit=limit)
        keyword_results = self.search_by_keywords(query_keywords, limit=limit)

        merged = {}
        for item in seed_results:
            merged[item["id"]] = item

        for item in keyword_results:
            if item["id"] not in merged:
                merged[item["id"]] = {
                    "id": item["id"],
                    "title": item["title"],
                    "text": item["text"]
                }

        return list(merged.values())[:limit]


# --------------------------------------------------
# 간단한 질문 키워드 추출 함수
# 나중에 더 고도화 가능
# --------------------------------------------------
def extract_query_keywords(question: str) -> List[str]:
    candidates = [
        "졸업", "졸업요건", "졸업학점", "졸업트랙",
        "교양", "교양학점", "교과구분", "일반선택",
        "전공", "전공학점", "전공필수",
        "창업", "창업교과목", "기술창업역량", "스타트업", "공동창업",
        "현장실습", "인턴십",
        "해외복수학위", "교환학생", "해외대학인정학점", "MOOC",
        "TOPCIT", "졸업자격인정원", "영어성적",
        "다중전공", "복수전공", "부전공", "융합전공", "연계전공",
        "창업경진대회", "도전 K-스타트업", "K-스타트업"
    ]

    found = []
    q = question.strip()

    for keyword in candidates:
        if keyword in q:
            found.append(keyword)

    return found


# --------------------------------------------------
# 단독 실행 테스트용
# --------------------------------------------------
if __name__ == "__main__":
    expander = GraphExpander()

    # 1) seed id 기반 테스트
    seed_ids = ["NOTICE_GLSW_STARTUP_003"]
    print("\n=== seed_ids 기반 확장 ===")
    results = expander.expand_related_notices(seed_ids, limit=10)
    for r in results:
        print(r["id"], ":", r["title"])

    # 2) 질문 키워드 기반 테스트
    question = "창업교과목 인정 기준이 뭐야?"
    keywords = extract_query_keywords(question)
    print("\n=== 질문 키워드 ===")
    print(keywords)

    print("\n=== 질문 기반 그래프 검색 ===")
    results = expander.search_by_keywords(keywords, limit=10)
    for r in results:
        print(r["id"], ":", r["title"], "| matched:", r.get("matched"))

    expander.close()