import pandas as pd
import numpy as np

def detect_trade_anomalies(data_path, date_col='기간', group_cols=['국가', 'HS코드'], value_cols=['수입 중량', '수출 중량'], threshold=-0.15):
    """
    관세청 무역 데이터를 로드하여 전월 대비 변동률(MoM)을 계산하고,
    임계치(Threshold)를 초과하는 이상 변동(Trigger)을 탐지합니다.
    """
    
    try:
        df = pd.read_csv(data_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: '{data_path}' not found")
        return None, None

    if date_col in df.columns:
        df = df[df[date_col] != '총계'].copy()
        
    # sort
    date_dt_col = f'{date_col}_dt'
    df[date_dt_col] = pd.to_datetime(df[date_col], format='%Y-%m', errors='coerce')
    
    sort_columns = group_cols + [date_dt_col]
    existing_sort_cols = [col for col in sort_columns if col in df.columns]
    
    df = df.sort_values(by=existing_sort_cols).reset_index(drop=True)
    
    # 전월 대비 증감률 계산
    existing_group_cols = [col for col in group_cols if col in df.columns]
    trigger_conditions = []
    
    for col in value_cols:
        mom_col = f'{col}_MoM_변동률'
        if col in df.columns:
            df[mom_col] = df.groupby(existing_group_cols)[col].pct_change()
            # 0으로 나누어 발생한 inf 값을 NaN으로 처리
            df[mom_col] = df[mom_col].replace([np.inf, -np.inf], np.nan)
            # 임계치 이하 조건 저장
            trigger_conditions.append(df[mom_col] <= threshold)
    
    # trigger
    if trigger_conditions:
        combined_condition = np.logical_or.reduce(trigger_conditions)
        df['Trigger_발동'] = combined_condition
        triggered_df = df[df['Trigger_발동']].copy()
    else:
        df['Trigger_발동'] = False
        triggered_df = pd.DataFrame()
    
    return df, triggered_df


if __name__ == "__main__":
    DATA_PATH = '../data/total_trade.csv'

    full_data, trigger_alerts = detect_trade_anomalies(
        data_path=DATA_PATH, 
        value_cols=['수입 중량', '수출 중량'], 
        threshold=-0.15
    )
    
    # # 코드 실행 확인용
    # if trigger_alerts is not None and not trigger_alerts.empty:
    #     print("\n[최근 트리거 사례 Top 5]")
        
    #     display_cols = ['기간', '국가', 'HS코드', '수입 중량', '수출 중량', '수입 중량_MoM', '수출 중량_MoM']
        
    #     temp_alerts = trigger_alerts[display_cols].copy()
        
    #     for col in ['수입중량_MoM', '수출 중량_MoM']:
    #         temp_alerts[col] = (temp_alerts[col] * 100).round(2).astype(str) + '%'
        
    #     print(temp_alerts.tail(5))
        
    #     full_data.to_csv(DATA_PATH, index=False, encoding='utf-8-sig')
    #     print(f"\n양방향 트리거 계산 결과가 '{DATA_PATH}'에 업데이트됨")