# pattern_detection/layers/layer2_pattern_confirmation.py

import pandas as pd
import numpy as np
import scipy.stats # for linear regression
import math # for angle calculation
from datetime import datetime, timedelta

def _count_valid_touches(pattern_data_df: pd.DataFrame, trendline_slope, trendline_intercept, start_date, is_upper_line=True, tolerance_pct=0.015):
    """
    Counts valid touches on a trendline within a pattern.
    A 'touch' is defined as the relevant price (High for upper, Low for lower) being within a tolerance band
    of the trendline, and showing a potential reversal.
    
    Parameters:
        pattern_data_df (pd.DataFrame): The DataFrame containing OHLCV data for the pattern duration.
        trendline_slope (float): The slope of the trendline.
        trendline_intercept (float): The y-intercept of the trendline.
        start_date (datetime): The start date of the pattern, used to normalize x-axis (days).
        is_upper_line (bool): True if checking the upper trendline, False for the lower.
        tolerance_pct (float): The percentage tolerance around the trendline for a touch.
        
    Returns:
        int: The number of valid touches.
    """
    touch_count = 0
    
    # Ensure 'Date' column exists (if index is datetime, convert it to a column)
    if 'Date' not in pattern_data_df.columns:
        indexed_data = pattern_data_df.reset_index().rename(columns={'index': 'Date'})
    else:
        indexed_data = pattern_data_df.copy() # Use a copy to avoid modifying original

    indexed_data['Days_Since_Start'] = (indexed_data['Date'] - start_date).dt.days

    # Select the correct price series based on whether it's upper or lower line
    price_series_name = 'High' if is_upper_line else 'Low'

    # Track last touch date to avoid counting consecutive noisy points as multiple touches
    last_touch_date = None
    MIN_DAYS_BETWEEN_TOUCHES = 5 # Require at least 5 days between distinct touches

    for i in range(len(indexed_data)):
        current_date = indexed_data.loc[i, 'Date']
        days_since_start = indexed_data.loc[i, 'Days_Since_Start']
        actual_price = indexed_data.loc[i, price_series_name] # Use 'High' or 'Low' price
        current_open = indexed_data.loc[i, 'Open']
        current_close = indexed_data.loc[i, 'Close']

        # Calculate trendline price for the current day
        trendline_price = trendline_slope * days_since_start + trendline_intercept
        
        # Define the acceptable band around the trendline
        # Allow price to be slightly above for upper line, or slightly below for lower line
        lower_band = trendline_price * (1 - tolerance_pct)
        upper_band = trendline_price * (1 + tolerance_pct)

        # Check if current price point is within the tolerance band
        is_within_band = lower_band <= actual_price <= upper_band

        if is_upper_line:
            # For upper line: price should touch or slightly exceed from below/within the band
            # And ideally, the candle closes below its open or below the trendline (indicating resistance)
            if is_within_band and actual_price >= trendline_price * (1 - tolerance_pct/2): # More focused check: high touches above lower bound of band
                # Ensure enough time has passed since the last counted touch
                if last_touch_date is None or (current_date - last_touch_date).days >= MIN_DAYS_BETWEEN_TOUCHES:
                    # Basic check for reversal tendency (current close vs open, or next day's close)
                    # This is still a simplification; true reversal detection is more complex.
                    
                    if current_close < current_open or current_close < trendline_price: # Candle closed lower or below trendline
                        touch_count += 1
                        last_touch_date = current_date
                    elif i < len(indexed_data) - 1: # If not a clear reversal on current candle, check next candle's close
                        next_close = indexed_data.loc[i+1, 'Close']
                        if next_close < current_close: # Next close is lower than current close
                            touch_count += 1
                            last_touch_date = current_date

        else: # Lower line
            # For lower line: price should touch or slightly fall below from above/within the band
            # And ideally, the candle closes above its open or above the trendline (indicating support)
            if is_within_band and actual_price <= trendline_price * (1 + tolerance_pct/2): # More focused check: low touches below upper bound of band
                # Ensure enough time has passed since the last counted touch
                if last_touch_date is None or (current_date - last_touch_date).days >= MIN_DAYS_BETWEEN_TOUCHES:
                    
                    if current_close > current_open or current_close > trendline_price: # Candle closed higher or above trendline
                        touch_count += 1
                        last_touch_date = current_date
                    elif i < len(indexed_data) - 1: # If not a clear reversal on current candle, check next candle's close
                        next_close = indexed_data.loc[i+1, 'Close']
                        if next_close > current_close: # Next close is higher than current close
                            touch_count += 1
                            last_touch_date = current_date
    
    return touch_count


def confirm_pattern_quality(candidate_pattern: dict, full_data: pd.DataFrame) -> dict | None:
    """
    第二層：對候選型態進行更深入的幾何形狀品質分析和技術指標綜合分析。
    輸入:
        candidate_pattern: 從 Layer 1 傳來的潛在型態字典。
        full_data: 完整的原始股票 K 線數據 DataFrame (未還原權息)。
    輸出: 如果型態被確認，返回一個包含詳細特徵的字典；否則返回 None。
    此層旨在對初步型態進行嚴格的幾何驗證和量價、指標的合理性檢查。
    """
    print("  > Layer 2: 正在執行型態確認 (基於原始 K 線數據)...")
    symbol = candidate_pattern.get("symbol")
    start_date_str = candidate_pattern.get("start_date")
    end_date_str = candidate_pattern.get("end_date")
    
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    # 提取型態區間的數據
    # 確保包含 Open, High, Low, Close, Capacity (成交量)
    pattern_data = full_data.loc[start_date:end_date].copy() 
    
    # 確保有必要的欄位
    required_cols = ['Open', 'High', 'Low', 'Close', 'Capacity']
    if not all(col in pattern_data.columns for col in required_cols):
        print(f"    > 數據缺少必要欄位 (需 {required_cols})，型態不通過。")
        return None

    # 這個判斷條件導致了「數據不足以進行第二層確認」的輸出
    # 我們需要重新思考這個條件的邏輯，或者將其放寬
    # 至少要有足夠的 K 線數量才能稱為一個「型態」
    # 參考 layer1 的 MIN_PEAK_TROUGH_DIST, pattern_data 應該至少是 MIN_PEAK_TROUGH_DIST * 2 或更多
    # 目前設置為 30 似乎太高，尤其是對 10, 20 天的 window size 來說
    if pattern_data.empty or len(pattern_data) < 15: # 調整為至少 15 天數據
        print("    > 數據不足以進行第二層確認。")
        return None

    # 1. 幾何形狀品質分析
    # 從候選模式中獲取初步的趨勢線點。
    upper_points_info = candidate_pattern.get("potential_trendline_points", {}).get("upper", [])
    lower_points_info = candidate_pattern.get("potential_trendline_points", {}).get("lower", [])

    if len(upper_points_info) < 2 or len(lower_points_info) < 2:
        print("    > 趨勢線觸及點不足 (Layer 1 傳入)，型態不通過。")
        return None

    # 提取日期索引和價格
    # 將日期轉換為自模式開始以來的天數，以便進行線性回歸
    # 確保轉換為 NumPy 陣列以便 linregress 處理
    upper_dates_idx = np.array([(pd.to_datetime(p["date"]) - start_date).days for p in upper_points_info])
    upper_prices = np.array([p["price"] for p in upper_points_info])
    lower_dates_idx = np.array([(pd.to_datetime(p["date"]) - start_date).days for p in lower_points_info])
    lower_prices = np.array([p["price"] for p in lower_points_info])

    # 擬合趨勢線 (使用線性回歸)
    try:
        # 檢查是否有足夠的獨特點來進行回歸，避免 NaN 或 inf 斜率
        if len(np.unique(upper_dates_idx)) < 2: # 如果日期都是一樣的，無法計算斜率
             upper_slope, upper_intercept, r_value_upper = 0.0, upper_prices.mean(), 0.0 
        else:
            upper_slope, upper_intercept, r_value_upper, _, _ = scipy.stats.linregress(upper_dates_idx, upper_prices)

        if len(np.unique(lower_dates_idx)) < 2: # 如果日期都是一樣的，無法計算斜率
             lower_slope, lower_intercept, r_value_lower = 0.0, lower_prices.mean(), 0.0 
        else:
            lower_slope, lower_intercept, r_value_lower, _, _ = scipy.stats.linregress(lower_dates_idx, lower_prices)

    except ValueError: 
        print("    > 無法擬合趨勢線 (點太少或值無變化)，型態不通過。")
        return None
    except Exception as e:
        print(f"    > 擬合趨勢線時發生未知錯誤: {e}，型態不通過。")
        return None

    # 趨勢線直線度驗證 (R^2 值接近 1 表示擬合度高，趨勢線越直)
    # 放寬 R^2 閾值，允許更大的波動
    if r_value_upper**2 < 0.6 or r_value_lower**2 < 0.6: # 閾值調整為 0.6
        print("    > 趨勢線直線度不佳 (R^2 不足)，型態不通過。")
        return None

    # 收斂性檢查：判斷三角類型和斜率方向
    SLOPE_THRESHOLD_FOR_FLAT = 0.005 # 定義斜率接近水平的閾值

    is_upper_down = upper_slope < -SLOPE_THRESHOLD_FOR_FLAT
    is_lower_up = lower_slope > SLOPE_THRESHOLD_FOR_FLAT
    is_upper_flat = abs(upper_slope) < SLOPE_THRESHOLD_FOR_FLAT
    is_lower_flat = abs(lower_slope) < SLOPE_THRESHOLD_FOR_FLAT

    # 必須形成收斂型態：
    # 對稱三角: 上軌向下 AND 下軌向上
    # 上升三角: 上軌平坦 AND 下軌向上
    # 下降三角: 上軌向下 AND 下軌平坦
    if not ( (is_upper_down and is_lower_up) or \
             (is_upper_flat and is_lower_up) or \
             (is_upper_down and is_lower_flat) ):
        print("    > 趨勢線未形成明確的三角收斂形態 (斜率方向不符)。")
        return None

    # 收斂角度驗證 (用斜率計算夾角)
    # 避免除以零或極端斜率導致的 math domain error
    angle_upper = math.degrees(math.atan(upper_slope)) if not np.isclose(upper_slope, 0) else 0.0
    angle_lower = math.degrees(math.atan(lower_slope)) if not np.isclose(lower_slope, 0) else 0.0
    
    convergence_angle = abs(angle_upper - angle_lower) # 兩條線的夾角

    # 稍微放寬角度範圍，以捕捉更多型態
    if not (5 <= convergence_angle <= 85): # 將上限提高到 85 度
        print(f"    > 收斂角度 {convergence_angle:.2f} 不在合理範圍，型態不通過。")
        return None
    
    # 觸及點數量驗證 (至少 2 個，使用新實現的 _count_valid_touches)
    min_touches_required = 2 # 將要求降低到 2 個有效觸及點
    
    # 直接傳遞整個 pattern_data 給 _count_valid_touches 函數
    upper_touches = _count_valid_touches(pattern_data, upper_slope, upper_intercept, start_date, is_upper_line=True)
    lower_touches = _count_valid_touches(pattern_data, lower_slope, lower_intercept, start_date, is_upper_line=False) # 修正了 is_lower_line 為 is_upper_line

    if upper_touches < min_touches_required or lower_touches < min_touches_required:
        print(f"    > 有效觸及點數量不足 (上軌: {upper_touches}, 下軌: {lower_touches})，型態不通過。")
        return None


    # 2. 技術指標綜合分析 (基於**未還原權息數據**計算)
    if len(pattern_data) < 15: # 至少 15 天數據，對 RSI 也夠用
        print("    > 數據量不足以計算完整的技術指標，跳過 Layer 2。")
        return None
    
    # 在 full_data 上計算，然後取 pattern_data 範圍的 RSI 和 MACD
    temp_full_data = full_data.copy() # 確保操作在副本上，避免 SettingWithCopyWarning
    
    # RSI (Relative Strength Index)
    delta = temp_full_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = np.where(loss == 0, np.inf, gain / loss) 
    temp_full_data['RSI'] = 100 - (100 / (1 + rs))
    # 使用 .bfill() 代替 fillna(method='bfill', inplace=True)
    temp_full_data['RSI'] = temp_full_data['RSI'].bfill() # 向後填充NaN

    # MACD (Moving Average Convergence Divergence)
    ema_12 = temp_full_data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = temp_full_data['Close'].ewm(span=26, adjust=False).mean()
    temp_full_data['MACD'] = ema_12 - ema_26
    temp_full_data['Signal_Line'] = temp_full_data['MACD'].ewm(span=9, adjust=False).mean()
    temp_full_data['MACD_Hist'] = temp_full_data['MACD'] - temp_full_data['Signal_Line'] # MACD 柱狀圖
    # 使用 .bfill() 代替 fillna(method='bfill', inplace=True)
    temp_full_data['MACD'] = temp_full_data['MACD'].bfill()
    temp_full_data['Signal_Line'] = temp_full_data['Signal_Line'].bfill()
    temp_full_data['MACD_Hist'] = temp_full_data['MACD_Hist'].bfill()


    # 提取型態區間的指標數據
    pattern_data_with_indicators = temp_full_data.loc[start_date:end_date].copy()


    # 成交量分析 (通常在收斂期間萎縮)
    # 獲取型態前一段時間的成交量均值
    vol_start_before_pattern = start_date - timedelta(days=60)
    # 確保數據範圍有效，避免索引錯誤
    # 這裡的 volume_before_pattern_data 應從 full_data 而非 pattern_data_with_indicators 獲取
    volume_before_pattern_data = full_data['Capacity'].loc[vol_start_before_pattern:start_date - timedelta(days=1)]
    
    volume_mean_in_pattern = pattern_data_with_indicators['Capacity'].mean()
    volume_mean_before_pattern = volume_before_pattern_data.mean()
    
    volume_decreasing_in_pattern = False
    if not np.isnan(volume_mean_in_pattern) and not np.isnan(volume_mean_before_pattern) and volume_mean_before_pattern > 0:
        volume_decreasing_in_pattern = volume_mean_in_pattern < volume_mean_before_pattern * 0.95 # 稍微放寬閾值，下降 5% 視為遞減

    # 檢查 RSI/MACD 是否有收斂或背離現象 (這些分析結果將作為特徵傳給 Layer 3)
    # 此處僅為示意，實際需要實現複雜的背離偵測邏輯，例如：
    # - 判斷價格創高/新低時，指標是否未創高/新低
    rsi_analysis_result = "NORMAL" # 預設為正常
    macd_analysis_result = "NORMAL" # 預設為正常
    
    # 這裡可以加入更複雜的 RSI/MACD 背離判斷邏輯
    # ...

    # 如果通過所有檢查，返回確認後的特徵
    print("  > Layer 2: 型態通過確認，提取特徵。")
    return {
        "symbol": symbol,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "upper_slope": upper_slope,
        "upper_intercept": upper_intercept,
        "lower_slope": lower_slope,
        "lower_intercept": lower_intercept, # 修正了這裡的錯誤
        "convergence_angle": convergence_angle,
        "volatility_decreasing_flag": candidate_pattern.get("volatility_decreasing", False), 
        "volume_decreasing_in_pattern": volume_decreasing_in_pattern,
        "last_rsi": pattern_data_with_indicators['RSI'].iloc[-1] if not pattern_data_with_indicators['RSI'].empty and not pattern_data_with_indicators['RSI'].isnull().all() else np.nan,
        "last_macd": pattern_data_with_indicators['MACD'].iloc[-1] if not pattern_data_with_indicators['MACD'].empty and not pattern_data_with_indicators['MACD'].isnull().all() else np.nan,
        "macd_signal_diff": pattern_data_with_indicators['MACD_Hist'].iloc[-1] if not pattern_data_with_indicators['MACD_Hist'].empty and not pattern_data_with_indicators['MACD_Hist'].isnull().all() else np.nan,
        "rsi_divergence_detected": rsi_analysis_result,
        "macd_convergence_detected": macd_analysis_result,
        "upper_touches": upper_touches, 
        "lower_touches": lower_touches, 
        # 更多特徵...
    }