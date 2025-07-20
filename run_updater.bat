@echo off
rem 設定批次檔的工作目錄為當前批次檔所在的目錄
cd /d "%~dp0"

echo --- 開始台股資料更新流程 (%date% %time%) ---

rem --- 執行 get_stock_codes.py ---
echo.
echo === 執行 get_stock_codes.py ===
"C:\Users\my861\AppData\Local\Microsoft\WindowsApps\python.exe" "%~dp0get_stock_codes.py"
rem 如果您沒有使用虛擬環境，請將上面一行替換為您的 Python 解譯器完整路徑：
rem "C:\Users\您的使用者名稱\AppData\Local\Programs\Python\Python39\python.exe" "%~dp0get_stock_codes.py"
if %errorlevel% neq 0 (
    echo 錯誤：get_stock_codes.py 執行失敗！請檢查錯誤訊息。
    goto :eof
)

rem --- 執行 stock_downloader.py ---
echo.
echo === 執行 stock_downloader.py ===
"C:\Users\my861\AppData\Local\Microsoft\WindowsApps\python.exe" "%~dp0stock_downloader.py"
rem 如果您沒有使用虛擬環境，請將上面一行替換為您的 Python 解譯器完整路徑：
rem "C:\Users\您的使用者名稱\AppData\Local\Programs\Python\Python39\python.exe" "%~dp0stock_downloader.py"
if %errorlevel% neq 0 (
    echo 錯誤：stock_downloader.py 執行失敗！請檢查錯誤訊息。
    goto :eof
)

echo.
echo --- 台股資料更新流程完成 (%date% %time%) ---

pause