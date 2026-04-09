import os
import random
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
    
    target_countries = ['미국', '중국', '한국', '대만']
    
    # tatrget filtering ('2022-10' ~ '2023-10'/ target_countries)
    # 수출 규제 발표(22년 10월) 이후 1년간의 파급 효과
    filtered_targets = [
        q for q in queries 
        if '2022-10' <= q['trade_period'] <= '2023-10' and q['country'] in target_countries
    ]
    
    # target
    target_sample_size = min(5, len(filtered_targets))
    target_queries = random.sample(filtered_targets, target_sample_size)
    
    # random sample(국가,시기 무관)
    remaining_queries = [q for q in queries if q not in target_queries]
    
    random_sample_size = min(5, len(remaining_queries))
    random_queries = random.sample(remaining_queries, random_sample_size)
    
    sampled_queries = target_queries + random_queries
    
    print(f"\n [보고서용 추출 완료] 타겟 케이스 {len(target_queries)}건 + 일반 랜덤 샘플 {len(random_queries)}건 (총 {len(sampled_queries)}건)")

    report_dir = "../reports"
    os.makedirs(report_dir, exist_ok=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory="../data/chroma_db", embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    prompt_template = """
    너는 대기업의 C-Level(최고경영진)에게 직보하는 최고 수준의 '글로벌 공급망 및 지정학 리스크 수석 분석가(Senior Supply Chain Risk Analyst)'야.
    너의 임무는 아래 제공된 관세청의 [수치 데이터 팩트]와 이와 연관된 [검색된 뉴스 문맥]을 종합하여, 경영진이 즉각적인 SCM(공급망 관리) 대응 전략을 세울 수 있도록 논리적이고 빈틈없는 심층 리포트를 작성하는 거야.

    [수치 데이터 팩트 (Trigger Event)]:
    {query_fact}

    [검색된 관련 뉴스 문맥 (Retrieved Context)]:
    {retrieved_context}

    ---
    [분석 및 작성 가이드라인]
    아래 4가지 항목을 철저하게 지켜서 논리적으로 전개해.

    1. 무역 수치 하락의 지정학적 원인 분석:
       - 단순한 수치 나열을 넘어, 15% 하락이라는 '결과'를 촉발한 핵심 지정학적 '원인(예: 수출 규제, 전쟁, 지진 등)'을 [검색된 뉴스 문맥]에서 찾아내 인과관계를 증명할 것.
       - 뉴스에 없는 외부 지식(특히 해당 시점 이후의 미래 정보)을 끌어다 쓰는 환각(Hallucination)을 절대 금지함.

    2. 공급망 리스크 등급 (Low / Medium / High) 및 산정 논리:
       - 다음의 엄격한 기준에 따라 단 하나의 리스크 등급을 부여하고, 그 근거를 1~2문장으로 논리적으로 방어할 것.
       - **[등급 산정 기준표]**
         * High (고위험): 수출 통제, 제재, 전쟁 등 국가 정책이나 구조적 붕괴로 인해 글로벌 공급망 전체에 장기적이고 연쇄적인 타격이 예상되는 경우.
         * Medium (중위험): 특정 국가/지역에 국한된 분쟁이나 사고로, 대체 공급선 확보에 단기적인 지연이나 비용 상승이 동반되는 경우.
         * Low (저위험): 일시적인 자연재해, 단기 파업 등으로 1~2개월 내에 공급망 정상화가 확실시되는 경우.

    3. 향후 3개월 단기 전망:
       - 파악된 원인과 리스크 등급을 바탕으로, 향후 3개월간 해당 국가 및 글로벌 시장의 수급 상황(예: 재고 부족, 우회 수출 증가, 가격 변동 등)을 전문가적 시각으로 예측할 것.

    4. 출처 표기 (Source Citation):
       - 보고서의 신뢰성을 입증하기 위해, 반드시 참조한 뉴스의 [기사 발행일]과 [핵심 키워드/요약]을 리포트 맨 하단에 명시할 것.

    ---
    [출력 양식]
    반드시 아래 마크다운 양식을 엄격하게 적용하여 작성해.

    # 공급망 리스크 심층 분석 리포트

    ## 1. 수치 하락의 지정학적 원인
    (분석 내용)

    ## 2. 공급망 리스크 등급 및 산정 근거
    - **리스크 등급:** [Low / Medium / High]
    - **산정 근거:** (논리적 근거)

    ## 3. 향후 3개월 단기 전망
    (전망 내용)

    ## 4. 참고 문헌 (References)
    - (발행일): (기사 핵심 내용)
    """

    prompt = PromptTemplate.from_template(prompt_template)

    for idx, q in enumerate(sampled_queries, 1):
        country_name = q['country']
        period = q['trade_period']
        
        case_type = "타겟(수출규제 여파)" if q in target_queries else "랜덤(일반 샘플)"
        print(f"[{idx}/{len(sampled_queries)}] 리포트 생성 중 ({case_type}): {country_name} ({period})...")
        
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
            print(f"  -> API 에러 발생: {e}")

if __name__ == "__main__":
    run_rag_pipeline()