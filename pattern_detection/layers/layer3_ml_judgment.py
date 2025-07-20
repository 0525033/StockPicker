# pattern_detection/layers/layer3_ml_judgment.py

import pickle # 用於載入預訓練模型
import pandas as pd
import numpy as np
import os

# 假設模型是基於 sklearn 訓練的 (例如 RandomForestClassifier, LogisticRegression 等)
# 在實際應用中，您可能需要導入訓練模型時用到的特定類別，例如：
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler # 如果您的模型需要標準化

# 定義一個簡單的分類特徵映射，實際中會更複雜，與模型訓練一致
def _map_categorical_to_numerical(value):
    if value == "NORMAL":
        return 0
    elif value == "DIVERGENCE" or value == "CONVERGENCE":
        return 1
    else:
        return 0 # 預設值

def predict_pattern_outcome(features: dict, model_path: str) -> dict | None:
    """
    第三層：利用預訓練的機器學習模型進行綜合判斷和突破預測。
    輸入:
        features: 從 Layer 2 傳來的確認型態的特徵字典。
        model_path: 預訓練機器學習模型的檔案路徑。
    輸出: 包含型態類型、置信度、突破方向和風險評估的字典，如果判斷失敗則為 None。
    此層旨在利用數據驅動的方法，給出更客觀和量化的型態預測。
    """
    print("  > Layer 3: 正在執行機器學習綜合判斷...")

    # 1. 載入預訓練的機器學習模型
    model = None
    try:
        if not os.path.exists(model_path):
            print(f"錯誤：機器學習模型檔案不存在於：{model_path}。請先訓練並儲存模型。")
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("    > 已成功載入機器學習模型。")
    except Exception as e:
        print(f"錯誤：載入機器學習模型失敗：{e}")
        return None

    # 2. 特徵準備
    # 確保特徵順序和模型訓練時一致，並處理缺失值。
    # 這裡需要將字典轉換為模型預期的輸入格式 (例如 NumPy array 或 Pandas DataFrame)。
    
    # 假設模型訓練時使用的特徵列表 (請務必與您實際訓練模型時的特徵名稱和順序一致)
    # 這是一個示例列表，您需要根據實際訓練的特徵來調整
    EXPECTED_FEATURES = [
        "upper_slope", "lower_slope", "convergence_angle",
        "volatility_decreasing_flag", "volume_decreasing_in_pattern",
        "last_rsi", "last_macd", "macd_signal_diff",
        "rsi_divergence_detected", "macd_convergence_detected"
    ]
    
    # 建立模型輸入的 DataFrame
    # 使用 pd.DataFrame([features]) 確保它是一個 DataFrame，即使只有一行數據
    input_df = pd.DataFrame([features]) 
    
    # 處理布林值特徵 (轉換為 int，某些模型可能需要)
    input_df['volatility_decreasing_flag'] = input_df['volatility_decreasing_flag'].astype(int)
    input_df['volume_decreasing_in_pattern'] = input_df['volume_decreasing_in_pattern'].astype(int)

    # 處理分類特徵 (例如 RSI/MACD 的分析結果，這裡使用一個簡單的映射)
    input_df['rsi_divergence_detected'] = input_df['rsi_divergence_detected'].apply(_map_categorical_to_numerical)
    input_df['macd_convergence_detected'] = input_df['macd_convergence_detected'].apply(_map_categorical_to_numerical)

    # 確保所有預期特徵都存在，如果缺少則填充 NaN (模型需要處理)
    for feat in EXPECTED_FEATURES:
        if feat not in input_df.columns:
            input_df[feat] = np.nan 

    # 確保特徵順序與模型訓練時一致
    # 實際應用中，通常會在訓練模型時保存一個特徵列表，並在預測時依此排序
    model_input = input_df[EXPECTED_FEATURES]
    
    # 處理缺失值 (與模型訓練時一致的策略，例如填充平均值或中位數)
    # 注意：這裡使用 input_df.mean() 填充，這可能與訓練數據的統計量不一致。
    # 更穩健的做法是保存訓練時的統計量 (如 mean, std) 並在此處使用。
    model_input = model_input.fillna(model_input.mean()) 
    
    # 3. 模型預測
    confidence = 0.0
    direction = "UNKNOWN"
    pattern_type = "UNKNOWN" 
    
    try:
        # 預測型態置信度。
        # 假設模型是分類模型，predict_proba 會返回每個類別的概率。
        # 您需要根據您模型訓練時的類別定義來解釋這些概率。
        prediction_proba = model.predict_proba(model_input)
        
        # 這裡的邏輯需要根據您實際訓練的機器學習模型輸出格式來填寫。
        # 假設您的模型預測多個類別，例如：0:無效型態, 1:向上突破型態, 2:向下突破型態, ...
        # 您需要知道 model.classes_ 的順序
        
        # 示例：假設模型預測的最高概率就是其最確信的結果
        predicted_class_idx = np.argmax(prediction_proba, axis=1)[0]
        predicted_confidence = prediction_proba[0, predicted_class_idx]
        
        # 假設您模型訓練時的 class_labels 如下：
        # class_labels_map = {
        #     0: {"type": "NO_PATTERN", "direction": "NONE"},
        #     1: {"type": "SYMMETRICAL_TRIANGLE", "direction": "UP"},
        #     2: {"type": "SYMMETRICAL_TRIANGLE", "direction": "DOWN"},
        #     3: {"type": "ASCENDING_TRIANGLE", "direction": "UP"},
        #     4: {"type": "DESCENDING_TRIANGLE", "direction": "DOWN"},
        #     # 更多類型...
        # }
        
        # 由於沒有實際模型和其 class_labels，這裡只能提供概念性的判斷。
        # 您需要根據 `model.classes_[predicted_class_idx]` 來判斷真實的類別。
        
        # 為了 Pseudo Code 簡化：
        if predicted_confidence > 0.6: # 置信度閾值可調整
            confidence = predicted_confidence
            
            # 這裡的型態類型和方向判斷是基於 features 進行的簡化示例，
            # 實際應由 ML 模型直接預測。
            # 您可以訓練模型來直接輸出型態類型和突破方向。
            SLOPE_THRESHOLD_FOR_FLAT = 0.005 # 與 layer2_pattern_confirmation 保持一致
            if features["upper_slope"] < -SLOPE_THRESHOLD_FOR_FLAT and features["lower_slope"] > SLOPE_THRESHOLD_FOR_FLAT:
                pattern_type = "SYMMETRICAL_TRIANGLE"
            elif abs(features["upper_slope"]) < SLOPE_THRESHOLD_FOR_FLAT and features["lower_slope"] > SLOPE_THRESHOLD_FOR_FLAT:
                pattern_type = "ASCENDING_TRIANGLE"
            elif features["upper_slope"] < -SLOPE_THRESHOLD_FOR_FLAT and abs(features["lower_slope"]) < SLOPE_THRESHOLD_FOR_FLAT:
                pattern_type = "DESCENDING_TRIANGLE"
            else:
                pattern_type = "UNKNOWN_TRIANGLE"
            
            # 假設模型預測的最高概率類別可以直接映射到方向
            # 例如：如果最高概率是 "向上突破型態"
            direction = "UP" if predicted_class_idx in [1,3] else "DOWN" if predicted_class_idx in [2,4] else "UNKNOWN" # 示例映射
        else:
            confidence = 0.0
            print("    > 模型判斷置信度不足，未確認型態。")
            return None

    except Exception as e:
        print(f"錯誤：機器學習模型預測時發生錯誤：{e}")
        return None

    # 4. 風險評估
    # 可以基於模型預測的置信度、歷史回測數據、當前市場情緒等因素進行評估。
    risk = "LOW" if confidence > 0.8 else ("MEDIUM" if confidence > 0.6 else "HIGH")

    print("  > Layer 3: 機器學習判斷完成。")
    return {
        "type": pattern_type,
        "confidence": float(confidence),
        "direction": direction,
        "risk": risk,
        "upper_trendline_params": {"slope": features["upper_slope"], "intercept": features["upper_intercept"]},
        "lower_trendline_params": {"slope": features["lower_slope"], "intercept": features["lower_intercept"]},
        "volume_analysis_result": features["volume_decreasing_in_pattern"], # 布林值，可以考慮轉換為 "True"/"False" 或其他描述
        "rsi_macd_analysis_result": {
            "rsi": features["last_rsi"], 
            "macd_diff": features["macd_signal_diff"], 
            "rsi_div": features["rsi_divergence_detected"], 
            "macd_conv": features["macd_convergence_detected"]
        }
    }