from langchain_neo4j import Neo4jGraph

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "capstone1"

print("🔌 Neo4j 데이터베이스에 연결을 시도합니다...")

try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )

    print("✅ Graph DB 연결 성공!")

    result = graph.query("MATCH (n) RETURN count(n) AS node_count")
    print(f"📊 현재 DB의 노드 개수: {result[0]['node_count']}개")

except Exception as e:
    print(f"❌ 연결 실패: {e}")