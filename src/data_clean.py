import pandas as pd
import re

print("===관세청(TRASS) 데이터 전처리===")
try:
    trass_df = pd.read_csv('../data/total_trade.csv', encoding='utf-8-sig')
    trass_df = trass_df[trass_df['기간'] != '총계'].copy()
    
    trass_df['품목명'] = trass_df['HS코드'].apply(
        lambda x: '다이오드/트랜지스터' if str(x).startswith('8541') else '집적회로'
    )
    
    print(f"-> 관세청 데이터 정제 완료: 총 {len(trass_df)}행")
    print(trass_df.head(3))
except FileNotFoundError:
    print("Error: trade data not found")


print("\n=== 로이터 경제 뉴스 필터링 ===")
try:
    news_df = pd.read_csv('../data/reuters.csv', encoding='utf-8-sig')
    news_df = news_df.dropna(subset=['main_text', 'date']).copy()
    
    # datetime: str -> datetime
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df = news_df.dropna(subset=['date']).copy()

    tech_kw = 'semiconductor|microchip|ai chip|tsmc|samsung|intel|asml|smic|foundry|wafer'
    trade_kw = 'supply chain|shortage|export control|sanction|tariff|trade war|embargo'
    geo_kw = 'geopolitic|taiwan strait|sino-us|us-china|hegemony'
    all_keywords = f"{tech_kw}|{trade_kw}|{geo_kw}"
    
    filtered_news = news_df[news_df['main_text'].str.contains(all_keywords, case=False, na=False)].copy()

    def clean_text(text):
        text = str(text)
        text = re.split(r'Breakingviews\s*Reuters Breakingviews', text)[0]
        #remove: The author is a Reuters...
        text = re.split(r'\(The author is a Reuters.*?columnist.*?\)', text)[0]
        return text.strip()
        
    filtered_news['main_text'] = filtered_news['main_text'].apply(clean_text)
    
    print(f"complete reuter {len(news_df)}column -> after {len(filtered_news)}column")
    
    filtered_news.to_csv('../data/clean_reuters.csv', index=False, encoding='utf-8-sig')
except FileNotFoundError:
    print("Error: reuters data not found")

    import pandas as pd

print("\n===ACLED 지정학적 분쟁 데이터 전처리===")
try:
    acled_df = pd.read_csv('../data/ACLED_data.csv', encoding='utf-8-sig')
    
    essential_cols = ['event_date', 'disorder_type', 'event_type', 'location', 'notes', 'fatalities']
    if 'country' in acled_df.columns:
        essential_cols.insert(1, 'country')
        
    acled_df = acled_df[essential_cols].copy()
    acled_df['event_date'] = pd.to_datetime(acled_df['event_date'], errors='coerce')
    acled_df = acled_df.dropna(subset=['event_date']).copy()
    acled_df['기간'] = acled_df['event_date'].dt.strftime('%Y-%m')
    
    # 결측치 제거
    acled_df = acled_df.dropna(subset=['notes']).copy()
    
    if 'country' in acled_df.columns:
        target_countries = ['China', 'Taiwan', 'Japan', 'South Korea', 'Russia', 'India']
        acled_df = acled_df[acled_df['country'].isin(target_countries)].copy()
        
    print(f"-> ACLED 데이터 정제 완료: 총 {len(acled_df)}행")
    
    acled_df.to_csv('../data/clean_ACLED.csv', index=False, encoding='utf-8-sig')
    
except FileNotFoundError:
    print("Error: ACLED data not found")