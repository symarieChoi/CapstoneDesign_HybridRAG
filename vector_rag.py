import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 환경변수 및 모델 세팅
load_dotenv(override=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # 답변을 일관되게 하도록 온도 0 설정
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. 로컬에 저장된 Vector DB 불러오기 (문서를 다시 파싱할 필요가 없습니다!)
print("🧠 로컬에 저장된 Chroma DB를 불러옵니다...")
vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
# 질문과 가장 잘 맞는 문서 조각 4개(k=4)를 찾아오도록 설정
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
print("✅ DB 로드 완료!\n")

# 3. RAG 체인 (프롬프트) 만들기
prompt = ChatPromptTemplate.from_template("""
당신은 경북대학교 컴퓨터학부(글로벌소프트웨어융합전공)의 친절하고 정확한 학사 조교입니다.
아래의 제공된 문서 내용(Context)만 바탕으로 질문에 답변해 주세요.
문서에 없는 내용이라면 지어내지 말고 '제공된 문서에서는 해당 내용을 찾을 수 없습니다'라고 명확히 답해주세요.

Context: {context}

Question: {input}

Answer:
""")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 4. 테스트 질문 던지기
question = "2025학년도 졸업 사정 기준을 보면, '다전공'을 하는 학생과 '단일전공'을 하는 학생의 전공 이수 요구 학점이 다를 텐데, 구체적으로 각각 몇 학점씩 들어야 하나요?"

print(f"❓ 질문: {question}")
print("-" * 50)
print("🤖 AI 조교가 문서를 검색하여 답변을 생성하고 있습니다...\n")

response = rag_chain.invoke({"input": question})

print("✨ [최종 답변]")
print(response["answer"])