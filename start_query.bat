@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动【知识库问答服务】 http://127.0.0.1:8001/chat.html
".venv\Scripts\python.exe" -m uvicorn app.query_process.api.query_service:app --host 127.0.0.1 --port 8001 --reload
