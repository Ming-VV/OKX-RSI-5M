import time
import json
import os
import ccxt
import pandas as pd
import numpy as np

# ==========================
# 1. 從環境變數讀取 OKX API 金鑰 (合約模式)
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
# 2. 策略與合約參數設定
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

BASE_MARGIN_USD = 5.0           # 基礎買入保證金 5U (買單A)
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
                exchange.cancel_order(oid, SYMBOL)
                print(f"🗑️ 已成功撤銷舊掛單: {oid}")
        except Exception as e:
            print(f"⚠️ 撤銷訂單失敗 ({oid}) (可能已成交或不存在): {e}")

def run_strategy():
    global state
    
    try:
        current_time = get_taiwan_time()
        print(f"[{current_time}] 🔄 預先掛單機器人啟動 (保證金: {BASE_MARGIN_USD}U) | 標的: {SYMBOL} | 單向持倉模式")
        
        # ==========================
        # 設定保證金模式與槓桿 (單向持倉模式專用)
        # ==========================
        try:
            exchange.set_margin_mode(MARGIN_MODE, SYMBOL)
        except Exception as e:
            print(f"ℹ️ 保證金模式設定提示 (若已有掛單或持倉可能需先清空): {e}")

        try:
            exchange.set_leverage(int(LEVERAGE), SYMBOL, {'marginMode': MARGIN_MODE})
        except Exception as e:
            try:
                exchange.set_leverage(int(LEVERAGE), SYMBOL)
            except Exception as e2:
                print(f"❌ 槓桿設定最終失敗: {e2}")
                raise e2

        print(f"✅ 成功套用: 逐倉 (Isolated) / {LEVERAGE}x 槓桿")
        
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        
        df['rsi'] = calculate_rsi(df, RSI_LENGTH)
        current_rsi = df['rsi'].iloc[-1]
        
        print(f"現價: {current_price} | 當前動態 RSI(3): {current_rsi:.2f} | 持倉合約: {state['totalContracts']} | 解禁狀態: {state['rsi_reset']}")

        action_taken = "HOLD"

        # ==========================
        # 3. 交易所真實狀態與訂單成交檢查
        # ==========================
        open_orders = exchange.fetch_open_orders(SYMBOL)
        open_order_ids = [order['id'] for order in open_orders]
        
        tracked_active_ids = state.get('active_order_ids', [])
        missing_orders = [oid for oid in tracked_active_ids if oid not in open_order_ids]
        
        if missing_orders:
            print(f"🔍 偵測到有掛單已成交或取消，正在向交易所查詢真實狀態...")
            
            for oid in missing_orders:
                try:
                    order_info = exchange.fetch_order(oid, SYMBOL)
                    order_status = order_info.get('status')
                    
                    if order_status == 'closed':
                        print(f"🎉 確認掛單 {oid} 已真實成交！")
                        side = order_info.get('side')
                        
                        if side == 'sell':  # 平倉賣單成交 -> 清倉重置
                            state['buyCount'] = 0
                            state['totalContracts'] = 0.0
                            state['totalMarginUSD'] = 0.0
                            state['avgEntryPrice'] = 0.0
                            state['breakevenPrice'] = 0.0
                            state['liquidationPrice'] = 0.0
                            state['rsi_reset'] = True
                            print(f"✨ 固定 0.3% 止盈平倉單完全成交！已清倉並重置狀態。")
                            
                        elif side == 'buy':  # 買單成交 -> 更新持倉
                            positions = exchange.fetch_positions([SYMBOL])
                            actual_contracts = 0.0
                            actual_entry_price = 0.0
                            for pos in positions:
                                if pos['symbol'] == SYMBOL and float(pos['contracts']) > 0:
                                    actual_contracts = float(pos['contracts'])
                                    actual_entry_price = float(pos['entryPrice'])
                                    break
                            
                            # 【已修正】直接透過真實成交合約數與均價計算保證金，避免 cost 欄位換算錯誤
                            if actual_contracts > 0 and actual_entry_price > 0:
                                actual_notional = actual_contracts * actual_entry_price
                                exact_total_margin = actual_notional / LEVERAGE
                                
                                state['buyCount'] += 1
                                state['totalContracts'] = actual_contracts
                                state['totalMarginUSD'] = exact_total_margin
                                state['avgEntryPrice'] = actual_entry_price
                            else:
                                state['buyCount'] += 1
                                state['totalMarginUSD'] += BASE_MARGIN_USD
                                state['avgEntryPrice'] = current_price
                            
                            state['breakevenPrice'] = state['avgEntryPrice'] / (1.0 - (FEE_RATE * 2))
                            state['liquidationPrice'] = state['avgEntryPrice'] * (1.0 - (1.0 / LEVERAGE) + MAINTENANCE_MARGIN_RATE)
                            
                            state['rsi_reset'] = False
                            print(f"🛒 買單成交！總次數: {state['buyCount']} | 總合約: {state['totalContracts']} | 均價: {state['avgEntryPrice']:.2f} | 總保證金: {state['totalMarginUSD']:.2f}U")
                            
                            state['active_order_ids'] = []
                        
                    elif order_status == 'canceled':
                        print(f"⚠️ 掛單 {oid} 被取消。")
                        
                except Exception as e:
                    print(f"⚠️ 查詢訂單 {oid} 失敗: {e}")
            
            save_state(state)

        elif not tracked_active_ids and len(open_orders) > 0:
            state['active_order_ids'] = [o['id'] for o in open_orders]
            print(f"⚠️ 【狀態同步】自動同步現有掛單群組")
            save_state(state)

        # ==========================
        # 4. RSI 冷卻解禁判定 (>= 30 解禁)
        # ==========================
        if not state['rsi_reset'] and current_rsi >= RSI_CANBUY:
            state['rsi_reset'] = True
            print(f"📈 【RSI 冷卻解禁】當前 RSI 已回升至 {current_rsi:.2f} (>= 30)，解禁下次加碼開關！")
            save_state(state)

        # ==========================
        # 5. 核心判定與預先掛單佈署
        # ==========================
        is_safe_from_liq = (state['buyCount'] == 0) or (current_price > state['liquidationPrice'] * 1.02)
        is_allowed_by_cost = (state['buyCount'] == 0) or (state['breakevenPrice'] > current_price)

        should_buy = is_allowed_by_cost and is_safe_from_liq and state['rsi_reset']
        has_inventory = state['totalContracts'] > 0

        new_active_ids = []

        # --- A. 預先掛賣單 (內建防呆安全閥：若賣價不高於成本均價則不掛單) ---
        if has_inventory:
            target_sell_price = state['breakevenPrice'] * PROFIT_TARGET_RATIO
            formatted_sell_price = float(exchange.price_to_precision(SYMBOL, target_sell_price))
            formatted_contracts = float(exchange.amount_to_precision(SYMBOL, state['totalContracts']))
            
            if formatted_sell_price <= state['avgEntryPrice']:
                print(f"⚠️ 【安全機制攔截】預測賣單價 ({formatted_sell_price}) 低於或等於成本均價 ({state['avgEntryPrice']:.2f})，為避免虧損，暫不掛賣單！")
            else:
                sell_order_exists = False
                for o in open_orders:
                    if o['side'] == 'sell' and abs(float(o['price']) - formatted_sell_price) < 0.1:
                        new_active_ids.append(o['id'])
                        sell_order_exists = True
                    else:
                        if o['side'] == 'sell':
                            cancel_orders_safely([o['id']])
                
                if not sell_order_exists:
                    params = {'reduceOnly': True}
                    order_sell = exchange.create_limit_sell_order(SYMBOL, formatted_contracts, formatted_sell_price, params)
                    new_active_ids.append(order_sell['id'])
                    print(f"🎯 【預先掛賣單】獲利停利價: {formatted_sell_price} | 庫存: {formatted_contracts}")
                    action_taken = "SWAP_PRE_PLACE_TP_SELL"

        # --- B. 預先掛買單 (使用 5U 保證金計算) ---
        if should_buy:
            buy_orders_to_cancel = [o['id'] for o in open_orders if o['side'] == 'buy']
            if buy_orders_to_cancel:
                cancel_orders_safely(buy_orders_to_cancel)

            # 【預先買單 1】RSI=10 預判價格買單 (保證金 5U)
            rsi10_price = calculate_exact_price_for_rsi_with_current(df, current_price, RSI_LENGTH, RSI_BUY_TARGET)
            margin_1 = BASE_MARGIN_USD
            notional_1 = margin_1 * LEVERAGE
            contracts_1 = notional_1 / rsi10_price
            
            f_contracts_1 = float(exchange.amount_to_precision(SYMBOL, contracts_1))
            f_price_1 = float(exchange.price_to_precision(SYMBOL, rsi10_price))
            
            order1 = exchange.create_limit_buy_order(SYMBOL, f_contracts_1, f_price_1)
            new_active_ids.append(order1['id'])
            print(f"🎯 【預先掛買單A (RSI=10 預判)】掛單價: {f_price_1} | 保證金: {margin_1}U (委託價值: {notional_1}U)")

            # 【預先買單 2】0.97 折數巨幅加碼單 (庫存保證金 + 買單A保證金) * 2
            estimated_total_margin_including_a = state['totalMarginUSD'] + margin_1
            margin_2 = estimated_total_margin_including_a * 2.0
            
            if state['buyCount'] == 0:
                estimated_avg_price = f_price_1
            else:
                curr_contracts = state['totalContracts']
                estimated_total_contracts = curr_contracts + f_contracts_1
                estimated_avg_price = (state['avgEntryPrice'] * curr_contracts + f_price_1 * f_contracts_1) / estimated_total_contracts if estimated_total_contracts > 0 else f_price_1

            target_price_2 = estimated_avg_price * 0.97
            notional_2 = margin_2 * LEVERAGE
            contracts_2 = notional_2 / target_price_2
            
            f_contracts_2 = float(exchange.amount_to_precision(SYMBOL, contracts_2))
            f_price_2 = float(exchange.price_to_precision(SYMBOL, target_price_2))
            
            order2 = exchange.create_limit_buy_order(SYMBOL, f_contracts_2, f_price_2)
            new_active_ids.append(order2['id'])
            print(f"⚡ 【預先買單B (0.97折數)】掛單價: {f_price_2:.2f} | 保證金: {margin_2:.2f}U (委託價值: {notional_2}U)")
            
            action_taken = "SWAP_PRE_PLACE_DOUBLE_BUYS"

        state['active_order_ids'] = list(set(new_active_ids))
        save_state(state)

        # ==========================
        # 6. 寫入記錄至 CSV
        # ==========================
        last_candle = df.iloc[-1]
        log_data = {
            "timestamp": current_time,
            "symbol": SYMBOL,
            "open": round(last_candle['open'], 2),
            "high": round(last_candle['high'], 2),
            "low": round(last_candle['low'], 2),
            "close": round(last_candle['close'], 2),
            "rsi": round(current_rsi, 2),
            "buy_count": state['buyCount'],
            "total_margin_usd": round(state['totalMarginUSD'], 2),
            "avg_entry_price": round(state['avgEntryPrice'], 2),
            "breakeven_price": round(state['breakevenPrice'], 2),
            "liquidation_price": round(state['liquidationPrice'], 2),
            "rsi_reset": state['rsi_reset'],
            "action": action_taken
        }
        log_market_data(log_data)

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    run_strategy()
