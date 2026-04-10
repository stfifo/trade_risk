import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

print("===LLM-as-a-judge 채점관 가동 ===")

def evaluate_report():
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    eval_prompt_template = """
    너는 글로벌 공급망 리스크 리포트의 품질을 엄격하게 검증하는 '수석 인공지능 평가관(Senior AI Evaluator)'이야. 
    너의 목표는 AI가 작성한 리포트가 실제 발생한 지정학적 팩트와 일치하는지, 제공된 뉴스 문맥을 벗어난 거짓 정보(환각)는 없는지, 그리고 분석과 전망이 논리적으로 타당한지 객관적인 수치로 평가하는 거야.

    아래 제공된 [실제 팩트(정답)], [모델이 참고한 뉴스 컨텍스트], 그리고 [모델이 생성한 리포트]를 교차 검증하여 3가지 평가 항목을 각각 10점 만점으로 채점해 줘.

    [실제 팩트 (Ground Truth)]:
    {ground_truth_text}

    [모델이 참고한 뉴스 컨텍스트 (Retrieved Context)]:
    {retrieved_context_text}

    [모델이 생성한 리포트 (Generated Report)]:
    {generated_report_text}

    ---
    [세부 평가 기준 (Rubric)]
    
    1. 원인 도출 정확성 (Accuracy - 10점)
       - 10점: [실제 팩트]에 명시된 핵심 지정학적 원인을 완벽하고 정확하게 짚어냄.
       - 7점: 원인을 파악했으나 핵심 내용(예: 주체, 규제 내용 등)이 약간 모호함.
       - 3점: 경제적 현상만 나열하고 근본적인 지정학적 원인을 도출하지 못함.
       - 0점: 전혀 다른 엉뚱한 원인을 지목함.

    2. 신뢰성 및 환각 통제 (Faithfulness - 10점)
       - 10점: 오직 제공된 [뉴스 컨텍스트]에만 기반하여 작성됨. 외부 지식이나 과장이 전혀 없음.
       - 7점: 대체로 사실에 기반하나, 사소한 수치나 기간의 오류가 존재함.
       - 3점: [뉴스 컨텍스트]에 없는 외부 지식이나 미래의 사건을 끌어와 논리를 전개하는 환각(Hallucination) 발생.
       - 0점: 전혀 존재하지 않는 가상의 사건을 날조함.

    3. 논리적 완결성 (Coherence - 10점)
       - 10점: 리스크 등급(Low/Medium/High) 산정 기준이 명확하고, 원인부터 향후 단기 전망까지의 인과관계가 SCM 전문가 수준으로 완벽함.
       - 7점: 구조는 갖추었으나 리스크 등급 근거가 다소 빈약하거나 전망이 피상적임.
       - 3점: 원인 분석과 리스크 등급이 서로 모순되거나 논리 전개가 억지스러움.
       - 0점: 논리적 흐름이 붕괴되어 리포트로서의 가치가 없음.

    ---
    [출력 형식]
    충실한 사고의 사슬(Chain-of-Thought)을 유도하기 위해, 반드시 '평가 이유'를 먼저 상세히 작성한 후 점수를 매겨.

    - **1. 원인 도출 정확성**
      - 평가 이유: (모델이 정답의 어떤 부분을 잘 맞췄는지 혹은 놓쳤는지 구체적으로 분석)
      - 점수: [X]점 / 10점

    - **2. 신뢰성 및 환각 통제**
      - 평가 이유: (뉴스 컨텍스트에 기반해 충실히 썼는지, 아니면 어떤 환각이 발생했는지 증거를 들어 분석)
      - 점수: [Y]점 / 10점

    - **3. 논리적 완결성**
      - 평가 이유: (리스크 등급과 전망의 논리적 타당성 분석)
      - 점수: [Z]점 / 10점

    - **총점 및 총평**: 
      - 총점: [Total]점 / 30점
      - 총평: (모델의 종합적인 성능 분석 및 한계점 요약)
    """
    prompt = PromptTemplate.from_template(eval_prompt_template)

    TARGET_REPORT_FILE = "testcase_Target_china_2023-08.md" 
    
    GROUND_TRUTH = """
    2022년 10월 7일, 미국 상무부 산업안보국(BIS)이 중국의 첨단 반도체 생산 및 컴퓨팅 칩 확보를 제한하는 광범위한 수출 통제 조치를 발표함. 
    이로 인해 미국산 첨단 반도체 장비의 대중국 수출이 전면 통제되었으며, 글로벌 반도체 장비 기업들의 탈중국 러시와 단기적인 공급망 디커플링 충격이 발생하여 중국의 반도체 수입 수치가 급감함.
    """
    
    RETRIEVED_CONTEXT = """
    - 기사 발행시기(2022-10): 미국 정부가 중국 반도체 산업을 겨냥해 광범위한 수출 통제 조치를 단행했다. 상무부는 특정 나노미터 이하의 D램, 낸드플래시, 로직칩 등을 생산할 수 있는 첨단 장비 및 기술의 중국 수출을 엄격히 통제한다고 발표했다.
    - 기사 발행시기(2022-10 ~ 2022-11): 미국의 강력한 수출 제재 여파로 어플라이드 머티어리얼즈(AMAT), 램리서치 등 글로벌 반도체 장비 업체들이 중국 내 주요 고객사 공장에서 자사 인력들을 철수시키고 신규 장비 납품을 중단하는 사태가 벌어지고 있다.
    - 기사 발행시기(2022-11): 제재의 여파가 가시화되면서 중국 반도체 팹들의 신규 라인 증설에 제동이 걸렸으며, 중국 해관총서 데이터에 따르면 단기적으로 중국의 반도체 제조 장비 수입액이 급감하는 추세를 보이고 있다.
    """

    report_path = os.path.join("../RAG_report", TARGET_REPORT_FILE)
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            generated_report = f.read()
    except FileNotFoundError:
        print(f"에러: {report_path} 파일을 찾을 수 없습니다.")
        return

    print(f"\n[{TARGET_REPORT_FILE}] judge...\n")

    final_prompt = prompt.format(
        ground_truth_text=GROUND_TRUTH,
        retrieved_context_text=RETRIEVED_CONTEXT,
        generated_report_text=generated_report
    )
    
    try:
        response = evaluator_llm.invoke(final_prompt)
        result_text = response.content
        print("="*60)
        print("[LLM-as-a-Judge]")
        print("="*60)
        print(result_text)
        print("="*60)
        
        eval_filepath = os.path.join("../RAG_report", "llm_judge_testcase-china_08.md")
        with open(eval_filepath, "w", encoding="utf-8") as f:
            f.write(result_text)
            
        print(f"평가 결과가 저장되었습니다: {eval_filepath}")

    except Exception as e:
        print(f"API 에러 발생: {e}")

if __name__ == "__main__":
    evaluate_report()