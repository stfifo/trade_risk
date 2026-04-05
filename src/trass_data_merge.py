import pandas as pd
import glob
import os

file_pattern = "../data/trade_*.csv"
file_list = glob.glob(file_pattern)

if not file_list:
    print("error: trade data not found")
else:
    print(f"total {len(file_list)} file found")


df_list = []

for file in file_list:
    try:
        df = pd.read_csv(file, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='utf-8') 
    
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)
merged_df = merged_df[merged_df['기간'] != '총계']
print(f"\n병합 완료... 총 데이터 행 개수: {len(merged_df)}")

save_path = "../data/merged_trade_data.csv"
merged_df.to_csv(save_path, index=False, encoding='utf-8-sig')