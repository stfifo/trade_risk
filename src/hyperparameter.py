import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

print("===정량적 분석: Hyperparameter (Top-k) Study===")

def run_hyperparameter_study():
    # case: 22년 10월, 미국 상무부 대중 수출규제
    test_country = "중국"
    test_period = "2022-10"
    target_month_int = 202210
    
    test_query = (
        "2022-10 시점, 중국의 반도체 무역 수치가 15% 이상 급감했습니다. "
        "직전 1~2개월 동안 미국과 중국 지역에서 발생한 지정학적 이슈(수출 통제 등) 관련 정황을 "
        "바탕으로 공급망 타격 뉴스를 찾아줘."
    )
    
    ground_truth = (
        "2022년 10월, 미국 상무부가 중국의 첨단 반도체 및 반도체 생산 장비에 대한 포괄적인 수출 통제 조치를 발표함. "
        "이로 인해 중국 내 반도체 수입이 급감하고 단기적 공급망 차질이 발생함."
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory="../data/chroma_db", embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    gen_prompt = PromptTemplate.from_template("""
    너는 최고 수준의 '글로벌 공급망 및 지정학 리스크 수석 분석가(Senior Supply Chain Risk Analyst)'야.
    너의 임무는 관세청의 [무역 수치 하락 팩트]와 이와 연관된 [검색된 관련 뉴스]를 종합하여, 기업의 SCM(공급망 관리) 경영진이 즉각적으로 의사결정을 내릴 수 있는 심층 리스크 리포트를 작성하는 거야.

    [무역 수치 하락 팩트 (Trigger Event)]:
    {query}

    [검색된 관련 뉴스 (Retrieved Context)]:
    {context}

    ---
    [리포트 작성 기준 및 가이드라인]
    아래 4가지 항목을 기준으로 논리적이고 체계적인 리포트를 작성해.

    1. 지정학적 원인 분석 (Cause Analysis):
       - 무역 수치가 15% 이상 급감했다는 '결과'와 검색된 뉴스 속의 '지정학적 사건(원인)' 간의 인과관계를 명확하게 연결하여 서술할 것.
       - 외부 지식이나 상상을 절대 배제하고 오직 제공된 [검색된 관련 뉴스]에만 기반하여 팩트 위주로 작성할 것.

    2. 공급망 리스크 등급 산정 (Risk Assessment):
       - 상황의 심각성을 고려하여 [Low / Medium / High] 중 하나의 등급을 명확히 부여할 것.
       - **[등급 산정 논리적 기준]**
         * High (고위험): 글로벌 공급망 전체에 연쇄적인 파급 효과가 예상되며, 장기화될 가능성이 큰 구조적/정책적 제재인 경우 (예: 강도 높은 수출 통제, 전쟁 등)
         * Medium (중위험): 특정 국가나 지역에 국한된 문제이나, 대체 공급선을 확보하는 데 단기적인 비용이나 지연이 발생하는 경우
         * Low (저위험): 일시적인 자연재해나 단기 파업 등으로, 1~2개월 내에 정상적인 수급이 가능한 경우
       - 위 기준을 바탕으로 해당 등급을 부여한 논리적 근거를 1~2문장으로 명시할 것.

    3. 향후 3개월 단기 전망 (Short-Term Outlook):
       - 단순한 상황 요약을 넘어, 공급망 담당자 관점에서 향후 3개월간 해당 품목(예: 반도체)의 수급 상황이나 대체 경로 모색 등에 대한 전문가적 예측을 서술할 것.

    4. 출처 표기 (Source Citation):
       - 작성된 리포트의 신뢰성을 증명하기 위해, 참고한 뉴스의 [기사 발행일]과 [핵심 키워드]를 리포트 하단에 반드시 기재할 것.

    ---
    [출력 양식]
    반드시 아래 마크다운 양식을 엄격하게 지켜서 출력해.

    # 공급망 리스크 심층 분석 리포트

    ## 1. 수치 하락의 지정학적 원인
    (분석 내용 작성)

    ## 2. 공급망 리스크 등급 및 근거
    - **리스크 등급:** [Low / Medium / High]
    - **부여 근거:** (논리적 근거 작성)

    ## 3. 향후 3개월 단기 전망
    (전망 내용 작성)

    ## 4. 참고 출처 (References)
    - (발행일): (참조한 뉴스의 요약 내용)
    """)


    eval_prompt = PromptTemplate.from_template("""
    너는 글로벌 공급망 리스크 분석 리포트의 품질을 엄격하게 검증하는 '수석 인공지능 평가관(Senior AI Evaluator)'이야. 
    너의 목표는 RAG(검색 증강 생성) 시스템이 작성한 리포트가 실제 발생한 지정학적 팩트와 일치하는지, 제공된 뉴스 문맥을 벗어난 거짓 정보(환각)는 없는지, 그리고 분석과 전망이 논리적으로 타당한지 객관적인 수치로 평가하는 거야.

    아래 제공된 [정답], [검색된 뉴스 문맥], 그리고 시스템이 [생성한 리포트]를 교차 검증하여 3가지 평가 항목을 각각 10점 만점으로 채점해 줘.

    [정답 (Ground Truth)]:
    {ground_truth}

    [검색된 뉴스 문맥 (Retrieved Context)]:
    {context}

    [생성된 리포트 (Generated Report)]:
    {report}

    ---
    [세부 평가 기준 (Rubric)]
    1. 원인 도출 정확성 (10점)
       - 10점: [정답]에 명시된 핵심 지정학적 원인(예: 수출 통제, 지진 등)을 완벽하게 짚어냄.
       - 7점: 원인을 파악했으나 핵심 내용이 약간 모호하거나 부수적인 요인에 집중함.
       - 3점: 현상만 나열하고 제대로 된 원인을 도출하지 못함.
       - 0점: 전혀 다른 엉뚱한 원인을 지목함.

    2. 신뢰성 및 환각 통제 (10점)
       - 10점: 오직 제공된 [검색된 뉴스 문맥]과 팩트에만 기반하여 작성됨. 외부 지식이나 과장이 전혀 없음.
       - 7점: 대체로 사실에 기반하나, 사소한 수치나 기간의 오류/과장이 존재함.
       - 3점: [검색된 뉴스 문맥]에 없는 외부 지식을 끌어와 논리를 전개하는 심각한 환각(Hallucination) 발생.
       - 0점: 전혀 존재하지 않는 가상의 사건이나 미래의 정보를 날조함.

    3. 논리적 완결성 (10점)
       - 10점: 리스크 등급 부여 근거가 명확하고, 원인부터 향후 3개월 단기 전망까지의 인과관계가 매우 설득력 있음.
       - 7점: 구조는 갖추었으나 리스크 등급 근거가 다소 빈약하거나 전망이 피상적임.
       - 3점: 원인 분석과 리스크 등급/전망이 서로 모순되거나 구조적으로 부실함.
       - 0점: 논리적 흐름이 붕괴되어 전문가의 리포트라 볼 수 없음.

    ---
    [출력 양식]
    반드시 점수를 도출한 이유를 먼저 짧게 설명한 후, 맨 마지막 줄에 기존 양식대로 점수를 출력해. (파싱을 위해 괄호 안의 숫자 포맷을 반드시 지킬 것)

    - 1. 원인 도출 정확성: [평가 이유 작성] -> [X]점
    - 2. 신뢰성 및 환각 통제: [평가 이유 작성] -> [Y]점
    - 3. 논리적 완결성: [평가 이유 작성] -> [Z]점
    - 총점: [Total]점 / 30점

    """)

    k_values = [1, 3, 4, 5, 7, 9, 11, 13, 15, 20]
    results = []

    print(f"테스트 케이스: {test_country} ({test_period})")
    
    for k in k_values:
        print(f"\n[실험] Top-k = {k} 진행 중...")
        
        search_filter = {"date_int": {"$lte": target_month_int}}
        retriever = vectorstore.as_retriever(search_kwargs={"k": k, "filter": search_filter})
        
        docs = retriever.invoke(test_query)
        context_text = "\n".join([doc.page_content for doc in docs])
        
        report_response = llm.invoke(gen_prompt.format(query=test_query, context=context_text))
        generated_report = report_response.content
        
        eval_response = evaluator_llm.invoke(eval_prompt.format(
            ground_truth=ground_truth, 
            context=context_text, 
            report=generated_report))
        eval_text = eval_response.content
        
        try:
            score1 = int(re.search(r"원인 도출 정확성:\s*\[?(\d+)\]?점", eval_text).group(1))
            score2 = int(re.search(r"신뢰성 및 환각 통제:\s*\[?(\d+)\]?점", eval_text).group(1))
            score3 = int(re.search(r"논리적 완결성:\s*\[?(\d+)\]?점", eval_text).group(1))
            total = int(re.search(r"총점:\s*\[?(\d+)\]?점", eval_text).group(1))
        except AttributeError:
            print("ERROR: 점수 파싱 에러. 기본값 0 처리")
            score1, score2, score3, total = 0, 0, 0, 0

        results.append({
            "Top-k": k,
            "원인 도출 (10)": score1,
            "환각 통제 (10)": score2,
            "논리성 (10)": score3,
            "총점 (30)": total
        })
        print(f"-> 총점: {total}/30")

    df_results = pd.DataFrame(results)
    print("\n[실험 결과 요약]")
    print(df_results.to_string(index=False))
    
    df_results.to_csv("../reports/hyperparameter_result.csv", index=False, encoding='utf-8-sig')
    print("\n실험 결과가 CSV로 저장")

if __name__ == "__main__":
    run_hyperparameter_study()