import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

print("=== 5. LLM-as-a-judge 채점관 가동 ===")

def evaluate_report():
    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    eval_prompt_template = """
    너는 글로벌 공급망 리스크 리포트의 품질을 엄격하게 검증하는 '수석 평가관(Senior Evaluator)'이야. 
    아래 제공된 [실제 팩트(정답)], [모델이 참고한 뉴스 컨텍스트], 그리고 [모델이 생성한 리포트]를 꼼꼼히 비교한 뒤, 다음 3가지 기준에 따라 각각 10점 만점으로 평가해 줘.

    [실제 팩트 (Ground Truth)]:
    {ground_truth_text}

    [모델이 참고한 뉴스 컨텍스트 (Retrieved Context)]:
    {retrieved_context_text}

    [모델이 생성한 리포트 (Generated Report)]:
    {generated_report_text}

    ---
    [평가 기준]
    1. 원인 도출 정확성 (Accuracy - 10점): 
       - 생성된 리포트가 [실제 팩트]에 명시된 핵심 지정학적 원인을 정확하게 도출했는가?
    2. 신뢰성 및 환각 통제 (Faithfulness - 10점): 
       - 리포트의 내용이 오직 제공된 [모델이 참고한 뉴스 컨텍스트]에만 기반하고 있는가? 외부 지식을 끌어와 환각(Hallucination)을 만들지 않았는가?
    3. 논리적 완결성 (Coherence - 10점): 
       - 리스크 등급(Low/Medium/High) 부여 근거가 타당하며, 향후 3개월 단기 전망이 설득력 있게 작성되었는가?

    ---
    [출력 형식]
    - **1. 원인 도출 정확성 ( [X]점 / 10점 )**
      - 평가 이유: ...
    - **2. 신뢰성 및 환각 통제 ( [Y]점 / 10점 )**
      - 평가 이유: ...
    - **3. 논리적 완결성 ( [Z]점 / 10점 )**
      - 평가 이유: ...
    - **총점 및 총평 ( [Total]점 / 30점 )**: ...
    """
    prompt = PromptTemplate.from_template(eval_prompt_template)

    # case: 22년 10월, 미국 상무부 대중 수출규제
    TARGET_REPORT_FILE = "RiskReport_Target_중국_2022-10.md" # (또는 미국_2022-10.md)
    
    GROUND_TRUTH = """
    2022년 10월, 미국 상무부가 중국의 첨단 반도체 및 반도체 생산 장비에 대한 포괄적인 수출 통제 조치를 발표함.
    이로 인해 중국 내 반도체 수입이 급감하고, 글로벌 공급망의 디커플링(탈동조화)이 가속화됨.
    """
    
    RETRIEVED_CONTEXT = """
    - 기사 발행일(2022-10-07): 미국 정부가 중국 반도체 산업을 겨냥해 광범위한 수출 통제 조치를 내렸다.
    - 기사 발행일(2022-10-12): 미국의 제재로 인해 글로벌 반도체 장비 업체들이 중국 내 직원들을 철수시키고 있다.
    """
    # =====================================================================

    report_path = os.path.join("../reports", TARGET_REPORT_FILE)
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            generated_report = f.read()
    except FileNotFoundError:
        print(f"에러: {report_path} 파일을 찾을 수 없습니다.")
        return

    print(f"\n[{TARGET_REPORT_FILE}] 리포트 채점ing...\n")

    final_prompt = prompt.format(
        ground_truth_text=GROUND_TRUTH,
        retrieved_context_text=RETRIEVED_CONTEXT,
        generated_report_text=generated_report
    )
    
    try:
        response = evaluator_llm.invoke(final_prompt)
        result_text = response.content
        print("="*60)
        print("[LLM as a Judge] 평가 결과:")
        print("="*60)
        print(result_text)
        print("="*60)
        
        eval_filepath = os.path.join("../reports", f"EvalResult_{TARGET_REPORT_FILE.replace('.md', '.txt')}")
        with open(eval_filepath, "w", encoding="utf-8") as f:
            f.write(result_text)
            
        print(f"평가 결과가 저장되었습니다: {eval_filepath}")

    except Exception as e:
        print(f"API 에러 발생: {e}")

if __name__ == "__main__":
    evaluate_report()