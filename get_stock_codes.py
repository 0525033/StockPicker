import twstock
import os
from datetime import datetime, timedelta # 雖然這個腳本用不到，但確保與之前的腳本一致

def get_and_save_stock_list(output_file="stock_list.txt"):
    """
    獲取所有台灣上市股票和 ETF 的代碼，並儲存到一個文字檔案中。

    Args:
        output_file (str): 儲存股票代碼的檔案名稱。
    """
    print("正在獲取所有台灣上市股票和 ETF 的代碼...")
    try:
        # 修正後的程式碼：直接使用 twstock.twse 字典來迭代
        # twstock.twse 包含了所有台灣證券交易所上市的證券資訊
        # 鍵為股票代碼，值為證券物件，其中包含 type (類別) 等資訊
        all_codes_info = twstock.twse # 注意這裡不再有 .codes

        stock_etf_codes = []
        for code, info_obj in all_codes_info.items(): # 迭代字典的 key (代碼) 和 value (資訊物件)
            # 篩選類別為 '股票' 或 'ETF' 的證券
            if info_obj.type == '股票' or info_obj.type == 'ETF':
                stock_etf_codes.append(code)
        
        # 將獲取到的代碼排序
        stock_etf_codes.sort()

        # 將代碼儲存到檔案中，每個代碼佔一行
        with open(output_file, 'w', encoding='utf-8') as f:
            for code in stock_etf_codes:
                f.write(f"{code}\n")
        
        print(f"成功將 {len(stock_etf_codes)} 個股票及 ETF 代碼儲存到 {output_file}")
        print("您可以在此檔案中查看代碼（預覽前10行）：")
        # 簡單預覽檔案內容
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 10: # 只印出前10行預覽
                    print(line.strip())
                else:
                    break
        print("...")

    except Exception as e:
        print(f"獲取代碼時發生錯誤: {e}")

# --- 執行函式 ---
if __name__ == "__main__":
    get_and_save_stock_list()

    # 以下是如何在 stock_downloader.py 中調用這些代碼的範例
    # 這部分邏輯不變，只需要確保 stock_list.txt 生成成功

    # print("\n--- 示範如何在 stock_downloader.py 中讀取代碼 ---")
    # try:
    #     stock_list_file = "stock_list.txt"
    #     if os.path.exists(stock_list_file):
    #         with open(stock_list_file, 'r', encoding='utf-8') as f:
    #             loaded_codes = [line.strip() for line in f if line.strip()]
    #         print(f"成功從 {stock_list_file} 讀取了 {len(loaded_codes)} 個代碼。")
    #         print("讀取到的前5個代碼:", loaded_codes[:5])
    #     else:
    #         print(f"股票代碼列表檔案 {stock_list_file} 不存在。請先運行此腳本來生成它。")
    # except Exception as e:
    #     print(f"讀取代碼列表時發生錯誤: {e}")