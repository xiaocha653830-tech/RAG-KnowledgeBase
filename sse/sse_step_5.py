import asyncio
import uuid
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()
# 跨域配置（保持不变）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ 替换列表缓存：用异步队列存储每个会话的消息（key: session_id, value: asyncio.Queue）
task_queues = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str = None

async def test_task(session_id:str, query:str):
    task_queues[session_id] = asyncio.Queue()
    queue = task_queues[session_id]
    for i in range(5):
        queue.put(
            {
                "event": "progress",
                "data": f"这是progress时事件的数据{i+1}"
            }
        )
        await asyncio.sleep(1)
    queue.put(
        {
            "event": "complete",
            "data": "事件完成"
        }
    )
    queue.put(None)

@app.post("/submit_query")
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    # 生成或使用用户传入的session_id
    session_id = request.session_id or str(uuid.uuid4())
    # 将耗时任务加入后台执行
    background_tasks.add_task(test_task, session_id, request.query)
    return {"message": "任务已经启动", "session_id": session_id}