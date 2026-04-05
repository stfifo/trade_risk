import pandas as pd
import re

print("...trass data filtering...")
# trass data
try:
    trass_df = pd.read_csv('../data/merged_trade_data.csv', encoding='utf-8-sig')
    
    trass_df = trass_df[trass_df['기간'] != '총계'].copy()
    
    trass_df['품목명'] = trass_df['HS코드'].apply(lambda x: '다이오드/트랜지스터' if x == 8541 else '집적회로')
    
    print(f"ttrade data clean: total {len(trass_df)} col")
    # 정제된 데이터 확인
    print(trass_df.head(3))
except FileNotFoundError:
    print("error: trass data not found")


print("...reuters news filtering...")
try:

    news_df = pd.read_csv('../data/reuters.csv', encoding='utf-8-sig')

    news_df = news_df.dropna(subset=['main_text', 'date']).copy()
    # keyword filtering
    tech_kw = 'semiconductor|microchip|ai chip|tsmc|samsung|intel|asml|smic|foundry|wafer'
    trade_kw = 'supply chain|shortage|export control|sanction|tariff|trade war|embargo'
    geo_kw = 'geopolitic|taiwan strait|sino-us|us-china|hegemony'
    all_keywords = f"{tech_kw}|{trade_kw}|{geo_kw}"
    
    filtered_news = news_df[news_df['main_text'].str.contains(all_keywords, case=False, na=False)].copy()
    
    def clean_text(text):
        text = str(text)
        text = text.split("Breakingviews\nReuters Breakingviews")[0]
        text = text.split("(The author is a Reuters Breakingviews columnist")[0]
        return text.strip()
        
    filtered_news['main_text'] = filtered_news['main_text'].apply(clean_text)
    
    print(f"로이터 뉴스 정제 완료: 원본 {len(news_df)}행 -> 정제 후 {len(filtered_news)}행")
    
    # 전처리 완료된 파일: reuters_filter.csv
    filtered_news.to_csv('../data/reuters_filter.csv', index=False, encoding='utf-8-sig')
    print(filtered_news.head(2))
except FileNotFoundError:
    print("error: reuters news data not found")