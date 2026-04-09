import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
#query_generator.py
from query_generator import generate_dynamic_queries

load_dotenv()

def run_rag_pipeline():

    queries = generate_dynamic_queries()
    
    if not queries:
        print("not query : no trigger")
        return

    # 2. Vector DB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory="../data/chroma_db", embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt_template = """
    너는 최고 수준의 글로벌 공급망 및 지정학 리스크 분석가야.
    아래 제공된 [수치 데이터 팩트]와 이와 관련된 [검색된 뉴스 문맥]을 종합하여 심층 리포트를 작성해줘.

    [수치 데이터 팩트]:
    {query_fact}

    [검색된 관련 뉴스 문맥]:
    {retrieved_context}

    다음 네 가지 항목을 반드시 포함해서 마크다운 형식으로 명확하게 작성해:
    1. 무역 수치 하락의 지정학적 원인 분석
    2. 공급망 리스크 등급 (Low / Medium / High 중 택 1) 및 부여 이유
    3. 향후 3개월 내 해당 품목의 단기 전망
    4. 출처 표기 (반드시 참조한 뉴스 문맥의 발행일과 핵심 키워드를 리포트 하단에 명시할 것)
    """
    prompt = PromptTemplate.from_template(prompt_template)

    for idx, q in enumerate(queries, 1):
        print(f"\n[{idx}/{len(queries)}] 리포트 생성 중: {q['country']} ({q['trade_period']})...")
        
        #Retrieval 시 트리거 발생 월(int) 이하의 데이터만 검색하는 메타데이터 필터 적용
        search_filter = {"date_int": {"$lte": q['target_month_int']}}
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "filter": search_filter}
        )
        
        docs = retriever.invoke(q['query'])
        context_text = "\n\n".join([f"- 기사 발행일({doc.metadata.get('date_str', 'N/A')}): {doc.page_content}" for doc in docs])
        
        # prompt and gemini
        final_prompt = prompt.format(query_fact=q['query'], retrieved_context=context_text)
        
        try:
            response = llm.invoke(final_prompt)
            
            #target : 22-10 미중 수출규제 여파
            #random: 국가/시기 무관 랜덤 샘플링
            prefix = "Target_ExportControl" if q in target_queries else "Random"
            filename = f"RiskReport_{prefix}_{country_name}_{period}.md"
            filepath = os.path.join(report_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# 📄 공급망 리스크 리포트 - {country_name} / {period}\n\n")
                f.write(response.content)
                
            print(f"  -> 저장 완료: {filepath}")
            
        except Exception as e:
            print(f"API error while generating report: {e}")

if __name__ == "__main__":
    run_rag_pipeline()