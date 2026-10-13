@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动【知识库导入服务】 http://127.0.0.1:8000/import.html
".venv\Scripts\python.exe" -m uvicorn app.import_process.api.file_import_service:app --host 127.0.0.1 --port 8000 --reload
