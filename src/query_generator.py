import pandas as pd
from dateutil.relativedelta import relativedelta

RELATED_COUNTRIES = {
    '대만': ['Taiwan', 'China', 'United States'],
    '중국': ['China', 'United States', 'Taiwan'],
    '미국': ['United States', 'China', 'Taiwan'],
    '러시아 연방': ['Russia', 'Ukraine', 'United States'],
    '일본': ['Japan', 'China', 'Taiwan', 'United States'],
    '인도': ['India', 'China']
}

def generate_dynamic_queries(trade_path='../data/total_trade.csv', acled_path='../data/clean_ACLED.csv'):
    trade_df = pd.read_csv(trade_path, encoding='utf-8-sig')
    acled_df = pd.read_csv(acled_path, encoding='utf-8-sig')
    
    if 'Trigger_발동' not in trade_df.columns:
        print("error: Trigger_발동' not in trade_df.columns")
        return []
        
    anomalies = trade_df[trade_df['Trigger_발동'] == True].copy()
    acled_df['event_date'] = pd.to_datetime(acled_df['기간'], format='%Y-%m')
    
    generated_queries = []
    
    for _, row in anomalies.iterrows():
        target_month = pd.to_datetime(row['기간'], format='%Y-%m')
        target_month_int = int(target_month.strftime('%Y%m')) # DB 필터링용 정수
        kor_country = row['국가']
        
        search_countries = RELATED_COUNTRIES.get(kor_country, [])
        month_minus_1 = target_month - relativedelta(months=1)
        month_minus_2 = target_month - relativedelta(months=2)
        
        # AND: trigger (Trade AND Geopolitics)
        geo_trigger = acled_df[
            (acled_df['country'].isin(search_countries)) & 
            (acled_df['event_date'].isin([month_minus_1, month_minus_2]))
        ]
        
        # query
        if not geo_trigger.empty:
            event_types = ", ".join(geo_trigger['event_type'].unique())
            
            #ACLED notes 내용의 앞100자 정도를 추출하여 쿼리 컨텍스트 강화
            notes_sample = " / ".join(
                geo_trigger['notes'].dropna().astype(str).apply(lambda x: x[:100] + "...").tolist()[:3]
            )
            
            query_str = (
                f"{row['기간']} 시점, {kor_country}의 반도체 무역 수치가 15% 이상 급감했습니다. "
                f"직전 1~2개월 동안 {', '.join(search_countries)} 지역에서 발생한 "
                f"지정학적 이슈({event_types}) 관련 정황[{notes_sample}]을 바탕으로 공급망 타격 뉴스를 찾아주세요."
            )
            
            generated_queries.append({
                'trade_period': row['기간'],
                'target_month_int': target_month_int,
                'country': kor_country,
                'query': query_str
            })
            
    print(f"total {len(generated_queries)} Data-Triggered query")
    return generated_queries