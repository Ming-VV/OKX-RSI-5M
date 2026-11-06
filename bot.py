import time                                   # 時間模組 (目前程式沒用到，保留備用)
import json                                   # 用來讀寫 JSON 狀態檔
import os                                     # 讀環境變數、檢查檔案是否存在
import ccxt                                   # 加密貨幣交易所統一介面套件
import pandas as pd                           # 處理 K 線資料表 (DataFrame)
import numpy as np                            # 數值運算 (目前沒直接用到，保留備用)

# ==========================
# 1. 從環境變數讀取 OKX API 金鑰 (合約模式)
# ==========================
exchange = ccxt.okx({                         # 建立 OKX 交易所物件
    'apiKey': os.environ.get('OKX_API_KEY'),          # API Key，從環境變數讀 (避免寫死在程式)
    'secret': os.environ.get('OKX_SECRET_KEY'),       # API Secret
    'password': os.environ.get('OKX_PASSWORD'),       # OKX 建立 API 時設定的 passphrase
    'enableRateLimit': True,                          # 開啟 ccxt 內建的請求頻率限制，避免被封
    'options': {'defaultType': 'swap'}                # 預設市場類型為永續合約
})

exchange.load_markets()                       # 載入所有交易對資訊 (精度、合約大小等)，後面換算要用

# ==========================
# 2. 策略與合約參數設定 (保守防禦 4 檔金字塔配置)
# ==========================
SYMBOL = 'ETH/USDT:USDT'        # 交易標的：ETH 對 USDT 永續合約 (U 本位)
TIMEFRAME = '5m'                # K 線週期：5 分鐘
RSI_LENGTH = 3                  # RSI 計算週期：3 根 K 線 (非常敏感)
RSI_BUY_TARGET = 10.0           # 預計在 RSI 跌到 10 的價位掛買單
RSI_CANBUY = 30.0               # RSI 回升到 30 以上，才解除「加碼冷卻」

# 4 檔分批止盈設定：ratio 是相對「回本價」的倍數，weight 是該檔賣出的持倉比例
PROFIT_TARGETS = [
    {"ratio": 1.001, "weight": 0.50},  # 第 1 檔：回本價 +0.1%，賣 50%
    {"ratio": 1.003, "weight": 0.25},  # 第 2 檔：回本價 +0.3%，賣 25%
    {"ratio": 1.006, "weight": 0.15},  # 第 3 檔：回本價 +0.6%，賣 15%
    {"ratio": 1.01,  "weight": 0.10}   # 第 4 檔：回本價 +1.0%，賣 10%
]

LEVERAGE = 10                   # 槓桿倍數 (沒有持倉時才會設定到交易所)
DEFAULT_MARGIN_MODE = 'isolated'  # 沒有持倉時，新倉位使用逐倉
MAINTENANCE_MARGIN_RATE = 0.005 # 維持保證金率 0.5% (用來估算強平價)

POSITION_MODE = 'net'           # 持倉模式：'net' 買賣模式；'long_short' 開平倉模式

TAKEOVER_KEEP_LEVERAGE = True   # 接管既有倉位時，是否保留交易所現有槓桿 (True=不改)

BASE_MARGIN_USD = 1.0          # 每次基礎買單投入的保證金 (USDT)
FEE_RATE = 0.0005               # 單邊手續費率 0.05%

STATE_FILE = "bot_state_swap.json"   # 機器人狀態存檔位置
LOG_FILE = "market_log_swap.csv"     # 行情與動作記錄檔


# ==========================
# 下單參數：tdMode 一律跟著「目前倉位的模式」走
# ==========================
def buy_params(td_mode):                      # 產生買單的額外參數
    params = {'tdMode': td_mode}              # tdMode 決定這張單是逐倉(isolated)還是全倉(cross)
    if POSITION_MODE == 'long_short':         # 若帳戶是開平倉模式
        params['posSide'] = 'long'            # 必須指定方向為多單
    return params                             # 回傳參數字典


def sell_params(td_mode):                     # 產生賣單的額外參數
    params = {'tdMode': td_mode}              # 賣單的保證金模式必須和倉位一致，否則會被拒
    if POSITION_MODE == 'long_short':         # 開平倉模式
        params['posSide'] = 'long'            # 平多單指定 long，且不能帶 reduceOnly
    else:                                     # 買賣模式
        params['reduceOnly'] = True           # 只減倉，避免賣單誤開出空單
    return params                             # 回傳參數字典


def get_taiwan_time():                        # 取得台灣時間字串
    """取得準確的台灣本地時間 (UTC+8)"""
    return pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M:%S')
    # 直接指定 Asia/Taipei 時區，不論主機是 UTC 或其他時區，都會得到正確的台灣時間


def load_state():                             # 讀取機器人狀態
    if os.path.exists(STATE_FILE):            # 若狀態檔存在
        with open(STATE_FILE, 'r') as f:      # 開檔讀取
            return json.load(f)               # 解析 JSON 並回傳
    return {                                  # 狀態檔不存在 -> 回傳全新初始狀態
        "buyCount": 0,                        # 已成交買單次數
        "totalContracts": 0.0,                # 目前持有張數
        "totalMarginUSD": 0.0,                # 累計投入保證金 (USDT)
        "avgEntryPrice": 0.0,                 # 持倉均價
        "breakevenPrice": 0.0,                # 含手續費的回本價
        "liquidationPrice": 0.0,              # 強平價
        "rsi_reset": True,                    # 加碼開關：True=可以掛新買單
        "active_order_ids": []                # 機器人目前追蹤的掛單 ID
    }


def save_state(state):                        # 儲存機器人狀態
    with open(STATE_FILE, 'w') as f:          # 以覆寫模式開檔
        json.dump(state, f, indent=4)         # 寫成排版好的 JSON


def log_market_data(log_data):                # 把一筆記錄寫進 CSV
    df_log = pd.DataFrame([log_data])         # 轉成單列 DataFrame
    if not os.path.exists(LOG_FILE):          # CSV 不存在 -> 第一次寫入
        df_log.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')   # 含表頭；utf-8-sig 讓 Excel 不亂碼
    else:                                     # CSV 已存在 -> 附加
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')  # 不重複寫表頭


state = load_state()                          # 程式啟動時載入狀態 (全域變數)


def calculate_rsi(df, period=3):              # 計算 RSI 指標
    delta = df['close'].diff()                # 每根 K 線收盤價與前一根的差
    gain = delta.clip(lower=0)                # 只保留上漲部分 (下跌變 0)
    loss = -delta.clip(upper=0)               # 只保留下跌幅度 (轉成正數)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()  # 平均漲幅 (Wilder 平滑)
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()  # 平均跌幅
    rs = avg_gain / avg_loss                  # 相對強度 RS
    rsi = 100 - (100 / (1 + rs))              # RSI 公式
    return rsi                                # 回傳整條 RSI 序列


def calculate_exact_price_for_rsi_with_current(df, current_price, period=3, target_rsi=10.0):
    """反推：價格要跌(或漲)到多少，RSI 才會剛好等於目標值"""
    alpha = 1.0 / period                      # 平滑係數
    temp_df = df.copy()                       # 複製資料，避免改到原表
    delta = temp_df['close'].diff()           # 價差
    gain = delta.clip(lower=0)                # 上漲部分
    loss = -delta.clip(upper=0)               # 下跌部分

    avg_gain_series = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()  # 平均漲幅序列
    avg_loss_series = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()  # 平均跌幅序列

    prev_avg_gain = avg_gain_series.iloc[-1]  # 最新一根的平均漲幅
    prev_avg_loss = avg_loss_series.iloc[-1]  # 最新一根的平均跌幅

    target_rs = target_rsi / (100.0 - target_rsi)   # 由目標 RSI 反推需要的 RS

    # 情境 A：假設下一個價格是「上漲」(此時 loss 只做衰減)
    target_avg_loss_A = (1.0 - alpha) * prev_avg_loss                 # 上漲時平均跌幅只會衰減
    target_avg_gain_A = target_rs * target_avg_loss_A                 # 需要的平均漲幅
    price_case_up = current_price + (target_avg_gain_A - (1.0 - alpha) * prev_avg_gain) / alpha  # 對應價格

    # 情境 B：假設下一個價格是「下跌」(此時 gain 只做衰減)
    target_avg_gain_B = (1.0 - alpha) * prev_avg_gain                 # 下跌時平均漲幅只會衰減
    target_avg_loss_B = target_avg_gain_B / target_rs                 # 需要的平均跌幅
    price_case_down = current_price - (target_avg_loss_B - (1.0 - alpha) * prev_avg_loss) / alpha  # 對應價格

    if target_rsi > 50:                       # 目標 RSI 偏高 -> 價格應在現價之上
        return price_case_up if price_case_up > current_price else price_case_down
    else:                                     # 目標 RSI 偏低 -> 價格應在現價之下
        return price_case_down if price_case_down < current_price else price_case_up


def cancel_orders_safely(order_ids):          # 批次撤單 (失敗不中斷)
    for oid in order_ids:                     # 逐一處理每個訂單 ID
        try:
            if oid:                           # ID 有效才撤
                exchange.cancel_order(oid, SYMBOL)           # 呼叫交易所撤單
                print(f"🗑️ 已成功撤銷舊掛單: {oid}")
        except Exception as e:                # 撤單失敗 (可能已成交或已不存在)
            print(f"⚠️ 撤銷訂單失敗 ({oid}) (可能已成交或不存在): {e}")


def find_long_position():                     # 查詢目前多單持倉
    """取得目前多單持倉資訊 (dict)；無持倉回傳 None"""
    positions = exchange.fetch_positions([SYMBOL])           # 向交易所查該標的持倉
    for pos in positions:                     # 逐筆檢查
        contracts = float(pos.get('contracts') or 0)         # 持有張數 (None 視為 0)
        side = pos.get('side')                               # 方向：long / short
        if pos['symbol'] == SYMBOL and contracts > 0 and side in (None, 'long'):  # 是此標的、有持倉、且是多單
            return {                          # 回傳整理後的持倉資訊
                'contracts': contracts,                                  # 張數
                'entryPrice': float(pos.get('entryPrice') or 0),         # 開倉均價
                'marginMode': pos.get('marginMode'),                     # 保證金模式 cross / isolated
                'leverage': pos.get('leverage'),                         # 目前槓桿
                'liquidationPrice': float(pos.get('liquidationPrice') or 0),  # 交易所計算的強平價
                'initialMargin': float(pos.get('initialMargin') or 0),   # 佔用的初始保證金
            }
    return None                               # 找不到持倉


def run_strategy():                           # 主策略：每次執行跑一輪
    global state                              # 使用全域的 state

    try:
        current_time = get_taiwan_time()      # 取得當前台灣時間
        print(f"[{current_time}] 🔄 保守防禦 4 檔金字塔機器人啟動 (保證金: {BASE_MARGIN_USD}U) | 標的: {SYMBOL}")

        # ==========================
        # 3-0. 先讀交易所真實持倉，決定本輪使用的保證金模式
        # ==========================
        pos_info = find_long_position()       # 查目前持倉
        if pos_info:                          # 有持倉
            td_mode = pos_info['marginMode'] or DEFAULT_MARGIN_MODE   # 沿用倉位的模式 (例如 cross)
            actual_contracts = pos_info['contracts']                  # 實際張數
            actual_entry_price = pos_info['entryPrice']               # 實際均價
        else:                                 # 沒有持倉
            td_mode = DEFAULT_MARGIN_MODE     # 使用預設的逐倉
            actual_contracts = 0.0            # 張數為 0
            actual_entry_price = 0.0          # 均價為 0

        # ==========================
        # 設定槓桿
        # ==========================
        if pos_info and TAKEOVER_KEEP_LEVERAGE:      # 接管中且選擇保留槓桿
            print(f"ℹ️ 接管模式：沿用現有槓桿 {pos_info['leverage']}x，不更動")
        else:                                 # 其他情況才設定槓桿
            lever_params = {'marginMode': td_mode}   # 槓桿設定需指定保證金模式
            if POSITION_MODE == 'long_short':        # 開平倉模式要指定方向
                lever_params['posSide'] = 'long'
            try:
                exchange.set_leverage(int(LEVERAGE), SYMBOL, lever_params)   # 呼叫交易所設定槓桿
            except Exception as e:
                print(f"❌ 槓桿設定失敗: {e}")
                raise e                       # 槓桿失敗就中止本輪 (避免用錯槓桿下單)

        # 計算本輪實際使用的槓桿：有持倉用倉位的槓桿，否則用設定值
        eff_leverage = float(pos_info['leverage']) if (pos_info and pos_info.get('leverage')) else float(LEVERAGE)
        print(f"✅ 本輪保證金模式: {td_mode} | 槓桿: {eff_leverage}x | 持倉模式: {POSITION_MODE}")

        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)    # 抓最近 50 根 5 分 K
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # 轉成表格

        ticker = exchange.fetch_ticker(SYMBOL)    # 抓最新報價
        current_price = ticker['last']            # 最新成交價

        df['rsi'] = calculate_rsi(df, RSI_LENGTH) # 計算 RSI 並存入表格
        current_rsi = df['rsi'].iloc[-1]          # 取最新一根的 RSI (尚未收盤的動態值)

        print(f"現價: {current_price} | 當前動態 RSI(3): {current_rsi:.2f} | 持倉合約: {state['totalContracts']} | 解禁狀態: {state['rsi_reset']}")

        action_taken = "HOLD"                     # 本輪動作記錄，預設為不動作

        # ==========================
        # 3. 狀態同步 / 接管既有倉位
        # ==========================
        if actual_contracts == 0 and state['totalContracts'] > 0:    # 交易所已無倉位，但機器人以為有
            print("✨ 偵測到持倉已全部清空（止盈完成），重置機器人狀態。")
            state['buyCount'] = 0                 # 以下全部歸零，回到初始狀態
            state['totalContracts'] = 0.0
            state['totalMarginUSD'] = 0.0
            state['avgEntryPrice'] = 0.0
            state['breakevenPrice'] = 0.0
            state['liquidationPrice'] = 0.0
            state['rsi_reset'] = True             # 加碼開關重新開啟
            state['active_order_ids'] = []        # 清空追蹤的掛單
            save_state(state)                     # 存檔

        elif actual_contracts > 0:                # 交易所有倉位
            # --- 接管：機器人狀態是空的，但交易所有倉位 ---
            if state['totalContracts'] == 0 and state['buyCount'] == 0:
                print(f"🤝 【接管既有倉位】模式: {td_mode} | 張數: {actual_contracts} | 均價: {actual_entry_price}")
                state['buyCount'] = 1             # 視為已買過 1 次 (讓成本與強平判斷生效)
                init_margin = pos_info['initialMargin']      # 優先用交易所回報的保證金
                if init_margin <= 0:              # 若交易所沒回報，自己估算
                    contract_size = exchange.market(SYMBOL)['contractSize'] or 1.0   # 每張合約大小
                    init_margin = actual_contracts * contract_size * actual_entry_price / eff_leverage  # 名目價值 / 槓桿
                state['totalMarginUSD'] = init_margin        # 記錄累計保證金
                state['rsi_reset'] = False        # 先鎖住加碼，等 RSI>=30 才解禁
                state['active_order_ids'] = []    # 清空追蹤掛單

            state['totalContracts'] = actual_contracts       # 同步最新張數
            if actual_entry_price > 0:            # 均價有效時更新相關價位
                state['avgEntryPrice'] = actual_entry_price  # 更新均價
                state['breakevenPrice'] = state['avgEntryPrice'] / (1.0 - (FEE_RATE * 2))  # 回本價 = 均價加上來回手續費
                if pos_info['liquidationPrice'] > 0:         # 交易所有提供強平價 -> 直接用
                    state['liquidationPrice'] = pos_info['liquidationPrice']
                else:                             # 沒有 -> 用公式估算 (逐倉近似)
                    state['liquidationPrice'] = state['avgEntryPrice'] * (1.0 - (1.0 / eff_leverage) + MAINTENANCE_MARGIN_RATE)
            save_state(state)                     # 存檔

        open_orders = exchange.fetch_open_orders(SYMBOL)             # 查交易所目前所有未成交掛單
        open_order_ids = [order['id'] for order in open_orders]      # 取出訂單 ID 清單

        tracked_active_ids = state.get('active_order_ids', [])       # 機器人上一輪記錄的掛單
        missing_orders = [oid for oid in tracked_active_ids if oid not in open_order_ids]  # 不在掛單列表 = 可能已成交/被撤

        if missing_orders:                        # 有消失的訂單 -> 逐一確認
            for oid in missing_orders:
                try:
                    order_info = exchange.fetch_order(oid, SYMBOL)   # 查該訂單的最終狀態
                    if order_info.get('status') == 'closed' and order_info.get('side') == 'buy':   # 是已成交的買單
                        state['buyCount'] += 1                       # 買入次數 +1

                        # 依「這張單實際成交」計算投入的保證金：成交張數 × 每張大小 × 成交均價 ÷ 槓桿
                        filled_qty = float(order_info.get('filled') or 0)                    # 實際成交張數
                        fill_price = float(order_info.get('average') or order_info.get('price') or 0)  # 成交均價 (沒有就用掛單價)
                        cs = exchange.market(SYMBOL).get('contractSize') or 1.0              # 每張合約大小
                        if filled_qty > 0 and fill_price > 0:                                # 資料完整 -> 用實際值
                            filled_margin = filled_qty * cs * fill_price / eff_leverage
                        else:                                                                # 資料缺失 -> 退回基礎保證金
                            filled_margin = BASE_MARGIN_USD
                        state['totalMarginUSD'] += filled_margin                             # 累計到總保證金
                        print(f"💰 本次成交保證金約 {filled_margin:.2f}U | 累計 {state['totalMarginUSD']:.2f}U")

                        p = find_long_position()                     # 重新查持倉
                        if p:                                        # 更新持倉資料
                            state['totalContracts'] = p['contracts']
                            state['avgEntryPrice'] = p['entryPrice']
                            if p['liquidationPrice'] > 0:            # 優先用交易所強平價
                                state['liquidationPrice'] = p['liquidationPrice']
                            else:                                    # 否則估算
                                state['liquidationPrice'] = state['avgEntryPrice'] * (1.0 - (1.0 / eff_leverage) + MAINTENANCE_MARGIN_RATE)

                        state['breakevenPrice'] = state['avgEntryPrice'] / (1.0 - (FEE_RATE * 2))   # 重算回本價
                        state['rsi_reset'] = False                   # 買進後鎖住加碼，等 RSI 回升
                        print(f"🛒 買單成交！總次數: {state['buyCount']} | 總合約: {state['totalContracts']} | 均價: {state['avgEntryPrice']:.2f}")
                except Exception as e:
                    print(f"⚠️ 查詢訂單 {oid} 失敗: {e}")

            state['active_order_ids'] = [o['id'] for o in open_orders]   # 追蹤清單更新為目前真實掛單
            save_state(state)

        # ==========================
        # 4. RSI 冷卻解禁判定 (>= 30 解禁)
        # ==========================
        if not state['rsi_reset'] and current_rsi >= RSI_CANBUY:     # 目前被鎖住，且 RSI 已回升到 30
            state['rsi_reset'] = True             # 解除鎖定，允許下次加碼
            print(f"📈 【RSI 冷卻解禁】當前 RSI 已回升至 {current_rsi:.2f} (>= 30)，解禁下次加碼開關！")
            save_state(state)

        # ==========================
        # 5. 核心判定與保守防禦 4 檔掛單佈署
        # ==========================
        # 安全條件1：沒買過，或現價高於強平價 2% 以上才允許買
        is_safe_from_liq = (state['buyCount'] == 0) or (current_price > state['liquidationPrice'] * 1.02)
        # 安全條件2：沒買過，或現價仍低於回本價 (只在虧損區加碼，不追高)
        is_allowed_by_cost = (state['buyCount'] == 0) or (state['breakevenPrice'] > current_price)

        should_buy = is_allowed_by_cost and is_safe_from_liq and state['rsi_reset']   # 三條件同時成立才掛買單
        has_inventory = state['totalContracts'] > 0      # 是否持有倉位 (決定要不要掛賣單)

        new_active_ids = []                       # 本輪新掛出的訂單 ID

        market = exchange.market(SYMBOL)          # 取得此標的的市場資訊
        contract_size = market.get('contractSize') or 1.0    # 每張合約大小 (ETH 約 0.1)

        # --- A. 4 檔分批止盈掛賣單 ---
        if has_inventory:                         # 有倉位才掛賣單
            sell_orders_to_cancel = [o['id'] for o in open_orders if o['side'] == 'sell']   # 找出舊賣單
            if sell_orders_to_cancel:
                cancel_orders_safely(sell_orders_to_cancel)   # 先撤掉舊賣單，再用最新均價重掛

            total_contracts = state['totalContracts']         # 目前總張數

            for idx, target in enumerate(PROFIT_TARGETS, start=1):    # 依序處理 4 檔
                raw_qty = total_contracts * target["weight"]          # 該檔應賣張數
                formatted_qty = float(exchange.amount_to_precision(SYMBOL, raw_qty))   # 依交易所精度截斷
                target_price = state['breakevenPrice'] * target["ratio"]              # 該檔目標價
                formatted_sell_price = float(exchange.price_to_precision(SYMBOL, target_price))  # 依價格精度處理

                if formatted_sell_price <= state['avgEntryPrice']:    # 目標價不高於均價 -> 會虧損，不掛
                    print(f"⚠️ 【安全機制】第 {idx} 檔止盈價 ({formatted_sell_price}) 低於成本均價，暫不掛此單")
                    continue

                if formatted_qty <= 0:            # 張數被精度截成 0 -> 無法下單
                    print(f"ℹ️ 第 {idx} 檔數量經精度處理後為 0 (原始 {raw_qty})，略過")
                    continue

                try:
                    order_sell = exchange.create_limit_sell_order(    # 掛限價賣單
                        SYMBOL, formatted_qty, formatted_sell_price, sell_params(td_mode)
                    )
                    new_active_ids.append(order_sell['id'])           # 記錄訂單 ID
                    print(f"🛡️ 【止盈掛單 #{idx}】目標價: {formatted_sell_price} (+{(target['ratio']-1)*100:.1f}%) | 數量: {formatted_qty} ({int(target['weight']*100)}%) | 模式: {td_mode}")
                    action_taken = "SWAP_PRE_PLACE_DEFENSIVE_4_SELLS" # 更新動作記錄
                except Exception as e:
                    print(f"❌ 第 {idx} 檔掛單失敗: {e}")

        # --- B. 預先掛買單 ---
        if should_buy:                            # 符合加碼條件
            buy_orders_to_cancel = [o['id'] for o in open_orders if o['side'] == 'buy']     # 找出舊買單
            if buy_orders_to_cancel:
                cancel_orders_safely(buy_orders_to_cancel)    # 撤掉舊買單，重新計算價位

            # 【買單 A】掛在「RSI 剛好等於 10」的預判價格
            rsi10_price = calculate_exact_price_for_rsi_with_current(df, current_price, RSI_LENGTH, RSI_BUY_TARGET)
            margin_1 = BASE_MARGIN_USD            # A 單保證金
            notional_1 = margin_1 * eff_leverage  # 名目價值 = 保證金 × 槓桿
            contracts_1 = notional_1 / rsi10_price / contract_size   # 換算成張數 (除以每張大小)

            f_contracts_1 = float(exchange.amount_to_precision(SYMBOL, contracts_1))   # 張數精度處理
            f_price_1 = float(exchange.price_to_precision(SYMBOL, rsi10_price))        # 價格精度處理

            try:
                order1 = exchange.create_limit_buy_order(SYMBOL, f_contracts_1, f_price_1, buy_params(td_mode))  # 掛限價買單
                new_active_ids.append(order1['id'])
                print(f"🎯 【預先掛買單A (RSI=10)】掛單價: {f_price_1} | 張數: {f_contracts_1} | 保證金: {margin_1}U | 模式: {td_mode}")
            except Exception as e:
                print(f"❌ 買單A 掛單失敗: {e}")

            # 【買單 B】掛在預估新均價 ×0.97 的位置，保證金為累計保證金 ×2 (馬丁式加碼)
            estimated_total_margin_including_a = state['totalMarginUSD'] + margin_1    # A 成交後的累計保證金
            margin_2 = estimated_total_margin_including_a * 2.0                        # B 單投入兩倍

            if state['buyCount'] == 0:            # 還沒持倉：新均價就是 A 單價格
                estimated_avg_price = f_price_1
            else:                                 # 已有持倉：加權平均估算 A 成交後的新均價
                curr_contracts = state['totalContracts']
                estimated_total_contracts = curr_contracts + f_contracts_1
                estimated_avg_price = (state['avgEntryPrice'] * curr_contracts + f_price_1 * f_contracts_1) / estimated_total_contracts if estimated_total_contracts > 0 else f_price_1

            target_price_2 = estimated_avg_price * 0.97      # B 單價格：比預估均價再低 3%
            notional_2 = margin_2 * eff_leverage             # B 單名目價值
            contracts_2 = notional_2 / target_price_2 / contract_size   # 換算成張數

            f_contracts_2 = float(exchange.amount_to_precision(SYMBOL, contracts_2))   # 張數精度處理
            f_price_2 = float(exchange.price_to_precision(SYMBOL, target_price_2))     # 價格精度處理

            try:
                order2 = exchange.create_limit_buy_order(SYMBOL, f_contracts_2, f_price_2, buy_params(td_mode))  # 掛限價買單
                new_active_ids.append(order2['id'])
                print(f"⚡ 【預先掛買單B (0.97折)】掛單價: {f_price_2:.2f} | 張數: {f_contracts_2} | 保證金: {margin_2:.2f}U | 模式: {td_mode}")
            except Exception as e:
                print(f"❌ 買單B 掛單失敗: {e}")

            action_taken = "SWAP_PRE_PLACE_DOUBLE_BUYS"      # 更新動作記錄

        state['active_order_ids'] = list(set(new_active_ids))   # 追蹤清單 = 本輪新掛出的單 (去重)
        save_state(state)                         # 存檔

        # ==========================
        # 6. 寫入記錄至 CSV
        # ==========================
        last_candle = df.iloc[-1]                 # 最新一根 K 線
        log_data = {                              # 組裝本輪記錄
            "timestamp": current_time,            # 時間
            "symbol": SYMBOL,                     # 標的
            "open": round(last_candle['open'], 2),     # 開
            "high": round(last_candle['high'], 2),     # 高
            "low": round(last_candle['low'], 2),       # 低
            "close": round(last_candle['close'], 2),   # 收
            "rsi": round(current_rsi, 2),         # RSI
            "buy_count": state['buyCount'],       # 買入次數
            "total_margin_usd": round(state['totalMarginUSD'], 2),     # 累計保證金
            "avg_entry_price": round(state['avgEntryPrice'], 2),       # 均價
            "breakeven_price": round(state['breakevenPrice'], 2),      # 回本價
            "liquidation_price": round(state['liquidationPrice'], 2),  # 強平價
            "rsi_reset": state['rsi_reset'],      # 加碼開關狀態
            "margin_mode": td_mode,               # 本輪保證金模式
            "action": action_taken                # 本輪動作
        }
        log_market_data(log_data)                 # 寫入 CSV

    except Exception as e:                        # 整輪的最外層保護：任何錯誤都只印出，不讓程式崩潰
        print(f"❌ 發生錯誤: {e}")


if __name__ == "__main__":                        # 直接執行此檔案時
    run_strategy()                                # 跑一輪策略 (需搭配排程如 cron 每 5 分鐘執行)
