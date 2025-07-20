import twstock
import pandas as pd
import pyarrow.parquet as pq
import os
from datetime import datetime, timedelta
import concurrent.futures
import time

MAX_WORKERS = 8 # 最大執行緒數量

def _fetch_and_store_single_stock(code, output_dir, file_format):
    """
    用於多執行緒的輔助函式，負責單支股票的資料獲取和儲存。
    會從最早可獲取日期開始，逐月獲取到今天，以確保資料完整性。
    """
    output_file = os.path.join(output_dir, f"{code}.{file_format}")
    existing_df = pd.DataFrame()
    
    # 設置結束日期為今天 (台灣時間)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # 初始假設從證券最早日期開始獲取
    # twstock.Stock 內部可能沒有直接提供 "最早上市日期"
    # 所以我們需要一個策略來獲取。這裡暫時假設從很久以前開始迭代
    # 或者我們可以先嘗試獲取最遠的歷史資料點來確定最早日期
    
    # 用來追蹤從哪個日期開始抓取新的歷史數據
    fetch_start_date = datetime(2000, 1, 1) # 設定一個足夠早的日期作為默認開始 (可以根據實際情況調整)

    print(f"[{code}] 正在處理...")

    # 1. 檢查現有檔案並獲取最新日期，以實現增量更新
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_parquet(output_file)
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            
            latest_date_in_file = existing_df['Date'].max()
            
            # 如果現有資料的最新日期已經是今天，則無需更新
            if latest_date_in_file >= today:
                print(f"[{code}] 資料已是最新 ({latest_date_in_file.strftime('%Y-%m-%d')})，無需更新。")
                return f"[{code}] 已是最新"

            # 設置新的獲取起始日期為現有檔案最新日期的隔天
            fetch_start_date = latest_date_in_file + timedelta(days=1)
            
            print(f"[{code}] 現有資料最新日期: {latest_date_in_file.strftime('%Y-%m-%d')}，將從 {fetch_start_date.strftime('%Y-%m-%d')} 開始更新。")

        except Exception as e:
            print(f"[{code}] 讀取或處理現有資料時發生錯誤: {e}。將從 {fetch_start_date.strftime('%Y-%m-%d')} 開始重新獲取所有歷史資料。")
            existing_df = pd.DataFrame() # 清空，將重新下載

    try:
        stock_obj = twstock.Stock(code) # 初始化 Stock 物件

        all_new_data_points = []
        
        # 逐月迭代從 fetch_start_date 到 today
        current_month_dt = fetch_start_date.replace(day=1) # 從 fetch_start_date 的這個月的第一天開始
        
        # 循環直到當前月份超過今天 (考慮到今天可能在月中，所以要處理到今天所在的月份)
        while current_month_dt <= today.replace(day=1) + timedelta(days=31): # 加31天確保涵蓋當前月份
            try:
                # 獲取當前月份的資料
                monthly_data = stock_obj.fetch(current_month_dt.year, current_month_dt.month)
                
                # 過濾資料，只保留指定日期範圍內的資料
                for data_point in monthly_data:
                    data_date = data_point.date
                    # 判斷條件：日期必須在增量更新的起始日期之後，並且在今天之前或等於今天
                    if data_date >= fetch_start_date and data_date <= today:
                        all_new_data_points.append({
                            'Date': data_date,
                            'Capacity': data_point.capacity,
                            'Turnover': data_point.turnover,
                            'Open': data_point.open,
                            'High': data_point.high,
                            'Low': data_point.low,
                            'Close': data_point.close,
                            'Change': data_point.change,
                            'Transaction': data_point.transaction
                        })
                
            except twstock.TwstockError as e:
                # twstock.TwstockError 是 twstock 特定的錯誤，例如沒有該月份資料
                print(f"[{code}] 獲取 {current_month_dt.year}-{current_month_dt.month} 月資料時發生 twstock 錯誤: {e}")
                # 這裡可以決定是否跳過這個月還是重試。目前選擇跳過
            except Exception as e:
                print(f"[{code}] 獲取 {current_month_dt.year}-{current_month_dt.month} 月資料時發生未知錯誤: {e}")

            # 移動到下一個月
            if current_month_dt.month == 12:
                current_month_dt = current_month_dt.replace(year=current_month_dt.year + 1, month=1, day=1)
            else:
                current_month_dt = current_month_dt.replace(month=current_month_dt.month + 1, day=1)

            # 加入短暫延遲，避免過快請求，尤其是在大量獲取時
            # time.sleep(0.1) # 根據需要調整，太快可能被擋，太慢影響效率
            
        if not all_new_data_points:
            print(f"[{code}] 沒有找到任何新的資料可供更新。")
            if existing_df.empty: # 如果原本就沒資料，也沒有新的，則表示真的沒有資料
                return f"[{code}] 無任何資料"
            else:
                return f"[{code}] 無新資料"

        new_df = pd.DataFrame(all_new_data_points)
        new_df['StockCode'] = code
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        new_df = new_df.set_index('Date').sort_index()

        # 合併現有資料和新資料
        if not existing_df.empty:
            # 合併現有和新資料，避免重複日期
            # 使用 concat 並 drop_duplicates，確保以 Date 為基準去重，保留最新資料
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['Date', 'StockCode'], keep='last')
            combined_df = combined_df.sort_index() # 再次按日期排序
            print(f"[{code}] 已更新 {len(new_df)} 筆新資料。總資料筆數: {len(combined_df)}")
        else:
            combined_df = new_df
            print(f"[{code}] 下載了 {len(new_df)} 筆資料。")

        # 儲存到 Parquet 檔案
        if file_format == "parquet":
            combined_df.to_parquet(output_file, index=True)
            return f"[{code}] 資料已成功儲存到 {output_file}"
        else:
            return f"[{code}] 不支援的檔案格式。"

    except Exception as e:
        return f"[{code}] 獲取或處理資料時發生錯誤: {e}"

# --- fetch_and_store_stock_data_threaded 函式保持不變 ---
def fetch_and_store_stock_data_threaded(
    stock_codes,
    output_dir="stock_data",
    file_format="parquet",
):
    """
    使用多執行緒獲取股票/ETF 歷史資料，支援增量更新。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_code = {
            executor.submit(_fetch_and_store_single_stock, code, output_dir, file_format): code
            for code in stock_codes
        }

        for future in concurrent.futures.as_completed(future_to_code):
            code = future_to_code[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                results.append(f"[{code}] 生成了例外: {exc}")
    
    print("\n--- 所有任務完成報告 ---")
    for res in results:
        print(res)

# --- if __name__ == "__main__": 區塊保持不變 ---
if __name__ == "__main__":
    stock_list_file = "stock_list.txt"
    taiwan_stocks_etfs = []

    try:
        if os.path.exists(stock_list_file):
            with open(stock_list_file, 'r', encoding='utf-8') as f:
                taiwan_stocks_etfs = [line.strip() for line in f if line.strip()]
            print(f"成功從 {stock_list_file} 讀取了 {len(taiwan_stocks_etfs)} 個股票及 ETF 代碼。")
            print("讀取到的前5個代碼:", taiwan_stocks_etfs[:5])
        else:
            print(f"錯誤：股票代碼列表檔案 {stock_list_file} 不存在。請先執行 get_stock_codes.py 來生成它。")
            import sys
            sys.exit(1)
            
    except Exception as e:
        print(f"讀取股票代碼列表時發生錯誤: {e}")
        import sys
        sys.exit(1)

    if taiwan_stocks_etfs:
        print("\n開始多執行緒下載與更新資料...")
        fetch_and_store_stock_data_threaded(taiwan_stocks_etfs)
        print("\n資料獲取與儲存完成！")
    else:
        print("\n沒有股票或 ETF 代碼可供下載，程式中止。")

    print("\n--- 讀取 Parquet 檔案範例 ---")
    try:
        if taiwan_stocks_etfs:
            example_code = taiwan_stocks_etfs[0]
            example_file = os.path.join("stock_data", f"{example_code}.parquet")
            if os.path.exists(example_file):
                df_read = pd.read_parquet(example_file)
                print(f"成功從 {example_file} 讀取 {example_code} 的資料。資料頭部範例:")
                print(df_read.head())
                print(f"資料尾部範例 (最新資料):")
                print(df_read.tail())
            else:
                print(f"範例檔案 {example_file} 不存在。請確保資料獲取過程成功完成。")
        else:
            print("沒有可供讀取的股票代碼，跳過讀取範例。")
    except Exception as e:
        print(f"讀取 Parquet 檔案時發生錯誤: {e}")