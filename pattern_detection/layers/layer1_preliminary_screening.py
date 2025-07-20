# pattern_detection/layers/layer1_preliminary_screening.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.stats # 可能用於未來更穩健的線性擬合

def find_potential_patterns(data: pd.DataFrame) -> list[dict]:
    """
    第一層：初步篩選潛在的三角收斂型態。
    輸入: 股票的 K 線數據 DataFrame (包含 'Open', 'High', 'Low', 'Close', 'Capacity'，索引為日期)。
    注意：這裡處理的是**未還原權息的原始股價**。
    輸出: 潛在型態的列表，每個型態是一個字典，包含初步的趨勢線點和區間。
    此層旨在快速濾除大量非潛在型態，提高後續層次的效率。
    判斷依據包括但不限於：潛在的高低點序列、波動率收縮等。
    """
    print("  > Layer 1: 正在執行初步篩選 (基於原始 K 線數據)...")
    potential_patterns = []

    # 定義要測試的視窗大小列表
    # 按照您的建議，新增了較小的視窗 (10, 20)，並延伸到 180 (約 9 個月)
    # 從 30 開始是為了更穩定，然後再擴大範圍
    WINDOW_SIZES_TO_TEST = [10, 20, 30, 60, 90, 120, 150, 180] 
    MIN_PEAK_TROUGH_DIST = 5 # 減少此值以允許更接近的峰谷點，適用於更緊密的型態

    if data.empty or len(data) < max(WINDOW_SIZES_TO_TEST): # 確保有足夠數據用於最大視窗
        print("    > 數據不足，無法進行初步篩選。")
        return potential_patterns

    # 針對整個數據或足夠長的周期計算 ATR
    # 確保 'Capacity' 被正確用於成交量，如同之前 'Volume' 的用法
    temp_close_shifted = data['Close'].shift(1).fillna(data['Close']) 
    data['TR'] = np.maximum(data['High'] - data['Low'], 
                             np.abs(data['High'] - temp_close_shifted), 
                             np.abs(data['Low'] - temp_close_shifted))
    data['ATR'] = data['TR'].ewm(span=14, adjust=False).mean() # 14 日 EMA ATR

    # 迭代不同的視窗大小
    for window_size in WINDOW_SIZES_TO_TEST:
        if len(data) < window_size:
            continue # 如果數據長度小於當前視窗，則跳過

        # 處理最近 `window_size` 天的數據
        # 確保視窗有足夠的數據點
        if len(data) < window_size:
            continue
            
        recent_data = data.iloc[-window_size:].copy() # 使用 .copy() 避免 SettingWithCopyWarning

        # --- 1. 改進的相對高點/低點偵測 ---
        # 使用比簡單滾動最大/最小值更穩健的方法來尋找峰谷點。
        # 這仍然是簡化版本；建議在生產環境中使用 `scipy.signal.find_peaks`。
        
        # 識別潛在的峰值 (局部最大值) 和谷值 (局部最小值)
        # 我們可以使用滾動視窗來查找當前 High/Low 在較小的周圍視窗 (例如，MIN_PEAK_TROUGH_DIST) 中是否是最高/最低點。
        # 這仍然是基礎方法；對於穩健的峰谷偵測，考慮 `scipy.signal.find_peaks` 
        # 或基於百分比價格波動的 Zig-Zag 類型演算法。

        # 簡化方法：查找 High 是其視窗中最大值，Low 是其視窗中最小值的點
        is_peak = (recent_data['High'] == recent_data['High'].rolling(window=MIN_PEAK_TROUGH_DIST * 2 + 1, center=True).max())
        is_trough = (recent_data['Low'] == recent_data['Low'].rolling(window=MIN_PEAK_TROUGH_DIST * 2 + 1, center=True).min())

        potential_high_points = recent_data[is_peak]['High'].dropna()
        potential_low_points = recent_data[is_trough]['Low'].dropna()

        # 過濾掉距離過近的點，或重複的點 (保留第一個)
        potential_high_points = potential_high_points[~potential_high_points.index.duplicated(keep='first')]
        potential_low_points = potential_low_points[~potential_low_points.index.duplicated(keep='first')]
        
        # 確保有足夠的點
        if len(potential_high_points) < 2 or len(potential_low_points) < 2:
            # print(f"    > Window {window_size}: 未找到足夠的高點/低點。")
            continue

        # --- 2. 波動率遞減分析 (ATR) ---
        volatility_decreasing = False
        if not recent_data['ATR'].empty and len(recent_data['ATR']) >= window_size * 0.5: # 需要足夠的數據點
            # 比較視窗後半段的平均 ATR 與前半段的平均 ATR
            recent_atr_mean = recent_data['ATR'].iloc[-int(window_size/2):].mean()
            older_atr_mean = recent_data['ATR'].iloc[0:int(window_size/2)].mean()
            
            # 放寬條件：檢查是否有任何程度的下降，不一定非得是 5%
            if not np.isnan(recent_atr_mean) and not np.isnan(older_atr_mean) and older_atr_mean > 0:
                # 波動率下降至少 1%
                volatility_decreasing = recent_atr_mean < older_atr_mean * 0.99 
            
        # --- 3. 初步趨勢線擬合與收斂傾向判斷 ---
        # 迭代高點組合以建立上軌，迭代低點組合以建立下軌。
        # 這是比簡單的「最近兩點」更需要改進的關鍵區域。
        # 為了這個例子簡化，我們仍然使用「最近相關點」的概念，但選擇上會稍微更穩健。

        # 選擇最近的 2 個顯示潛在下降趨勢的高點
        # 這仍然是啟發式方法；一個正確的方法會迭代並擬合線條。
        sorted_highs = potential_high_points.sort_index(ascending=False) # 按日期排序，最近的在前
        valid_upper_points = []
        for i in range(len(sorted_highs)):
            for j in range(i + 1, len(sorted_highs)):
                # 確保較晚的點低於或大致等於較早的點 (下降或平坦斜率)
                if sorted_highs.iloc[i] <= sorted_highs.iloc[j] + (sorted_highs.iloc[j] * 0.01): # 允許平坦的輕微容差
                    valid_upper_points.append(sorted_highs.index[i])
                    valid_upper_points.append(sorted_highs.index[j])
                    break # 找到兩個點用於潛在的上軌
            if len(valid_upper_points) >= 2:
                break
        
        # 選擇最近的 2 個顯示潛在上升趨勢的低點
        sorted_lows = potential_low_points.sort_index(ascending=False) # 按日期排序，最近的在前
        valid_lower_points = []
        for i in range(len(sorted_lows)):
            for j in range(i + 1, len(sorted_lows)):
                # 確保較晚的點高於或大致等於較早的點 (上升或平坦斜率)
                if sorted_lows.iloc[i] >= sorted_lows.iloc[j] - (sorted_lows.iloc[j] * 0.01): # 允許平坦的輕微容差
                    valid_lower_points.append(sorted_lows.index[i])
                    valid_lower_points.append(sorted_lows.index[j])
                    break # 找到兩個點用於潛在的下軌
            if len(valid_lower_points) >= 2:
                break

        if len(valid_upper_points) < 2 or len(valid_lower_points) < 2:
            # print(f"    > Window {window_size}: 未能找到合適的高點/低點趨勢點。")
            continue

        # 獲取這些選定點的實際價格
        current_upper_points = recent_data.loc[valid_upper_points]['High']
        current_lower_points = recent_data.loc[valid_lower_points]['Low']

        # 簡化的收斂斜率檢查：
        # 粗略檢查線條是否相互靠近，或者一條線平坦，另一條線靠近。
        upper_slope_check = (current_upper_points.iloc[0] - current_upper_points.iloc[1]) / ((current_upper_points.index[0] - current_upper_points.index[1]).days + 1)
        lower_slope_check = (current_lower_points.iloc[0] - current_lower_points.iloc[1]) / ((current_lower_points.index[0] - current_lower_points.index[1]).days + 1)
        
        # 基本收斂檢查：上軌斜率必須 <= 0，下軌斜率必須 >= 0。
        # 且它們不能同時朝同一個方向移動。
        converging = (upper_slope_check <= 0.01 and lower_slope_check >= -0.01) # 允許非常輕微的正/負斜率來表示平坦
        
        # 確保它們不是平行或發散的 (例如，都上升，都下降，或者上軌 > 0 且下軌 < 0)
        # 此外，檢查在周期結束時，上軌是否通常位於下軌之上
        if not converging or recent_data['High'].iloc[-1] <= recent_data['Low'].iloc[-1]: # 線條過早交叉的基本檢查
            # print(f"    > Window {window_size}: 線條未收斂或已交叉。")
            continue

        # 如果條件滿足，則添加一個潛在型態
        # 根據選定點的範圍確定型態的開始和結束日期
        pattern_start_date = min(current_upper_points.index.min(), current_lower_points.index.min())
        pattern_end_date = recent_data.index.max() # 型態通常結束於視窗中的最新數據點

        # 確保型態持續時間合理 (例如，至少 30 天)
        if (pattern_end_date - pattern_start_date).days < 30:
            # print(f"    > Window {window_size}: 型態持續時間過短。")
            continue

        potential_patterns.append({
            "symbol": data.name if hasattr(data, 'name') and data.name is not None else 'UNKNOWN',
            "start_date": pattern_start_date.strftime('%Y-%m-%d'),
            "end_date": pattern_end_date.strftime('%Y-%m-%d'),
            "window_size": window_size, # 記錄是哪個視窗找到的
            "potential_trendline_points": {
                "upper": [{"date": p_date.strftime('%Y-%m-%d'), "price": price} for p_date, price in current_upper_points.items()],
                "lower": [{"date": p_date.strftime('%Y-%m-%d'), "price": price} for p_date, price in current_lower_points.items()]
            },
            "volatility_decreasing": volatility_decreasing
        })
        # 如果我們在給定視窗中找到一個型態，我們可能會選擇不在其內部更小的子視窗中尋找
        # 目前，允許不同視窗大小的重疊型態。

    print(f"  > Layer 1: 初步篩選完成，發現 {len(potential_patterns)} 個潛在型態。")
    return potential_patterns