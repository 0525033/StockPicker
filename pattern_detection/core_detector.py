# pattern_detection/core_detector.py

import os
import pandas as pd
from datetime import datetime
import numpy as np
import glob

# 導入各層的模組 (標準套件匯入方式)
from pattern_detection.layers import layer1_pre_screening
from pattern_detection.layers import layer2_pattern_confirmation
from pattern_detection.layers import layer3_machine_learning_judgment

# 配置路徑 (請根據您的實際路徑調整)
STOCK_DATA_DIR = "C:\\Users\\my861\\OneDrive\\Desktop\\台股股價蒐集\\stock_data" # 已修正為 stock_data
MODEL_PATH = "C:\\Users\\my861\\OneDrive\\Desktop\\台股股價蒐集\\ml_model\\trained_model.pkl" # 您的模型路徑

def load_stock_data(symbol):
    """載入特定股票的 K 線數據（未還原權息）。"""
    file_path = os.path.join(STOCK_DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(file_path):
        print(f"  > 錯誤：找不到 {symbol} 的數據檔於 {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        # 確保必要的欄位存在
        required_cols = ['Open', 'High', 'Low', 'Close', 'Capacity']
        if not all(col in df.columns for col in required_cols):
            print(f"  > 警告：{symbol} 數據缺少必要的 OHLCV 欄位。")
            return None
        return df
    except Exception as e:
        print(f"  > 載入或處理 {symbol} 數據時發生錯誤: {e}")
        return None

def get_all_symbols():
    """從數據目錄讀取所有股票代碼。"""
    symbols = []
    # 使用 glob 更安全地遍歷 .csv 檔案
    for filepath in glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv")):
        filename = os.path.basename(filepath)
        symbol = os.path.splitext(filename)[0] # 獲取不帶副檔名的檔案名
        symbols.append(symbol)
    
    # 這裡可以根據需要進行排序或篩選，例如只包含數字代碼
    # symbols = sorted([s for s in symbols if s.isdigit()]) # 如果您只處理數字代碼
    return sorted(symbols)


def main_detection_system():
    """核心型態檢測系統的主函數。"""
    print("--- 啟動核心檢測系統 (基於未還原權息數據) ---")

    all_symbols = get_all_symbols()
    if not all_symbols:
        print("錯誤：未找到任何股票數據檔案。請檢查 STOCK_DATA_DIR 配置。")
        return

    print(f"已載入 {len(all_symbols)} 支股票代碼。")

    # --- 新增總結計數器 ---
    total_symbols_processed = 0
    total_potential_patterns_layer1 = 0
    total_confirmed_patterns_layer2 = 0
    total_final_patterns_layer3 = 0
    # ---

    for symbol in all_symbols:
        total_symbols_processed += 1
        print(f"\n--- 正在分析股票：{symbol} ---")
        stock_data = load_stock_data(symbol)

        if stock_data is None or stock_data.empty:
            print(f"  > 跳過 {symbol}，無效或空數據。")
            continue

        # 執行第一層：初步篩選
        print("  > 執行第一層：初步篩選...")
        potential_patterns = layer1_pre_screening.find_potential_patterns(stock_data)
        
        # --- 更新 Layer 1 總結 ---
        if potential_patterns:
            total_potential_patterns_layer1 += len(potential_patterns)
            print(f"  > {symbol} 初步篩選發現 {len(potential_patterns)} 個潛在型態。")
        else:
            print(f"  > {symbol} 未發現潛在型態。")
            continue # 如果沒有潛在型態，直接跳過 Layer 2, 3

        # 執行第二層：型態確認
        print("  > 執行第二層：型態確認...")
        confirmed_patterns_for_symbol = []
        for p_pattern in potential_patterns:
            confirmed_pattern_features = layer2_pattern_confirmation.confirm_pattern_quality(p_pattern, stock_data)
            if confirmed_pattern_features:
                confirmed_patterns_for_symbol.append(confirmed_pattern_features)
                # --- 更新 Layer 2 總結 ---
                total_confirmed_patterns_layer2 += 1 
                print(f"  > {symbol} 成功確認一個型態，準備進行機器學習判斷。")
                
                # 執行第三層：機器學習判斷
                print("  > Layer 3: 正在執行機器學習綜合判斷...")
                final_judgment = layer3_machine_learning_judgment.make_judgment(confirmed_pattern_features, MODEL_PATH)
                if final_judgment:
                    # --- 更新 Layer 3 總結 ---
                    total_final_patterns_layer3 += 1
                    print(f"  > {symbol} 機器學習判斷通過，確認為有效型態！")
                    # 在這裡可以添加將確認的型態儲存起來的邏輯
                else:
                    print(f"  > {symbol} 機器學習判斷未通過。")
            else:
                print(f"  > {symbol} 型態未通過第二層確認。")

    # --- 最終總結報告 ---
    print("\n" + "="*40)
    print("               檢測結果總結             ")
    print("="*40)
    print(f"處理股票總數： {total_symbols_processed} 支")
    print(f"Layer 1 初步篩選出的潛在型態總數： {total_potential_patterns_layer1} 個")
    print(f"Layer 2 確認為有效型態的總數： {total_confirmed_patterns_layer2} 個")
    print(f"Layer 3 機器學習最終確認的型態總數： {total_final_patterns_layer3} 個")
    print("="*40)

    if total_final_patterns_layer3 == 0:
        print("無三角收斂型態被檢測到，無需儲存。")
    # else:
    #     print("所有成功確認的型態已處理。") # 您可以在這裡添加儲存邏輯的提示
    
    print("--- 核心檢測系統執行完成 ---")

if __name__ == "__main__":
    main_detection_system()