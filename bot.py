import time
import json
import os
import ccxt
import pandas as pd
import numpy as np

# ==========================
# 1. 從 GitHub Secrets 讀取金鑰 (合約模式)
# ==========================
exchange = ccxt.okx({
    'apiKey': os.environ.get('OKX_API_KEY'),
    'secret': os.environ.get('OKX_SECRET_KEY'),
    'password': os.environ.get('OKX_PASSWORD'),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}  # 強制指定為永續合約市場
})

exchange.load_markets()

# ==========================
# 2. 策略與合約參數設定 (已移除最大加碼次數限制)
# ==========================
SYMBOL = 'ETH/USDT:USDT'        # OKX 永續合約標準標的
TIMEFRAME = '5m'                # 5分鐘極短線
RSI_LENGTH = 3                  # RSI 週期 3
RSI_BUY_TARGET = 10.0           # RSI=10 預判開倉目標
RSI_CANBUY = 30.0               # RSI >= 30 冷卻解禁門檻
PROFIT_TARGET_RATIO = 1.003     # 固定 0.3% 停利賣出目標 (保本價 * 1.003)

LEVERAGE = 10                   # 10 倍槓桿
MARGIN_MODE = 'isolated'        # 逐倉模式 (Isolated)
MAINTENANCE_MARGIN_RATE = 0.005 # 維持保證金率 (約 0.5%)

BASE_MARGIN_USD = 1.0           # 基礎買入保證金 1U (買單A)
FEE_RATE = 0.0005               # 合約手續費率

STATE_FILE = "bot_state_swap.json"
LOG_FILE = "market_log_swap.csv"

def get_taiwan_time():
    """取得準確的台灣本地時間 (UTC+8)"""
    return (pd.Timestamp.now() + pd.Timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        "buyCount": 0,
        "totalContracts": 0.0,
        "totalMarginUSD": 0.0,
        "avgEntryPrice": 0.0,
        "breakevenPrice": 0.0,
        "liquidationPrice": 0.0,
        "rsi_reset": True,
        "active_order_ids": []      # 追蹤所有掛單 ID
    }

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def log_market_data(log_data):
    df_log = pd.DataFrame([log_data])
    if not os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
    else:
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

state = load_state()

def calculate_rsi(df, period=3):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_exact_price_for_rsi_with_current(df, current_price, period=3, target_rsi=10.0):
    alpha = 1.0 / period
    temp_df = df.copy()
    delta = temp_df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain_series = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss_series = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    
    prev_avg_gain = avg_gain_series.iloc[-1]
    prev_avg_loss = avg_loss_series.iloc[-1]
    
    target_rs = target_rsi / (100.0 - target_rsi)
    
    target_avg_loss_A = (1.0 - alpha) * prev_avg_loss
    target_avg_gain_A = target_rs * target_avg_loss_A
    price_case_up = current_price + (target_avg_gain_A - (1.0 - alpha) * prev_avg_gain) / alpha
    
    target_avg_gain_B = (1.0 - alpha) * prev_avg_gain
    target_avg_loss_B = target_avg_gain_B / target_rs
    price_case_down = current_price - (target_avg_loss_B - (1.0 - alpha) * prev_avg_loss) / alpha
    
    if target_rsi > 50:
        return price_case_up if price_case_up > current_price else price_case_down
    else:
        return price_case_down if price_case_down < current_price else price_case_up

def cancel_orders_safely(order_ids):
    for oid in order_ids:
        try:
            if oid:
                exchange.cancel_order(
