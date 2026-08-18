import asyncio
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Body, Path, BackgroundTasks
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from query_process.main_graph import MainGraphRunner
from tool.mongo_client_tool import get_recent_history_list, clear_history
from tool.task_utils import create_queue, update_task_status, TASK_STATUS_PROCESSING, get_task_info, put_data, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, get_data

app = FastAPI(
    title="掌柜智库-查询模块API",
    description="掌柜智库-查询模块API",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health():
    return {"aa": "bb"}


@app.get("/history/{session_id}")
async def get_history(session_id: str = Path(..., description="会话ID")):
    history_list = get_recent_history_list(session_id)

    # 将_id从ObjectID对象转为字符串
    history_list = [
        {
            "_id": str(item.get("_id")),
            "role": item.get("role", ""),
            "text": item.get("text", ""),
            "rewritten_query": item.get("rewritten_query", ""),
            "item_names": item.get("item_names", ""),
            "ts": item.get("ts", ""),
            "session_id": item.get("session_id", "")
        }
        for item in history_list
    ]
    history_list.sort(key=lambda a: a.get("ts"))
    return {"items": history_list}


@app.delete("/history/{session_id}")
async def delete_history(session_id: str = Path(..., description="会话ID")):
    clear_history(session_id)
    return {"msg": "删除成功"}


class QueryParams(BaseModel):
    query: str = Field(..., description="查询问题")
    session_id: str = Field(..., description="会话ID")


def run_query_graph(task_id, original_query, session_id):
    create_queue(task_id)

    try:
        init_state = {
            "task_id": task_id,
            "original_query": original_query,
            "session_id": session_id
        }
        # 更新总状态，放到队列，sse后期就可以从队列当中取出更新的数据状态推送给前端
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        put_data(task_id, event="progress", data=get_task_info(task_id))

        MainGraphRunner.create_and_run(init_state)

        # 更新总状态，放到队列，sse后期就可以从队列当中取出更新的数据状态推送给前端
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        put_data(task_id, event="progress", data=get_task_info(task_id))
    except Exception as e:
        # 更新总状态，放到队列，sse后期就可以从队列当中取出更新的数据状态推送给前端
        update_task_status(task_id, TASK_STATUS_FAILED)
        put_data(task_id, event="error", data=get_task_info(task_id))
        raise e


@app.post("/query")
async def query(background_tasks: BackgroundTasks, query_params: QueryParams = Body(..., description="查询请求体参数")):
    task_id = str(uuid.uuid4())
    original_query = query_params.query
    session_id = query_params.session_id

    background_tasks.add_task(run_query_graph, task_id, original_query, session_id)

    return {
        "task_id": task_id,
        "original_query": original_query,
        "session_id": session_id
    }


def generate_stream(task_id):
    while True:
        item = get_data(task_id)
        time.sleep(0.1)
        yield f"event: {item.get("event")}\n"
        yield f"data: {json.dumps(item.get("data"), ensure_ascii=False)}\n\n"


@app.get("/stream/{task_id}")
async def stream(task_id: str = Path(..., description="任务ID")):
    return StreamingResponse(generate_stream(task_id), media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)
