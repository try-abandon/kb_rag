import shutil
import uuid
from datetime import datetime
from pathlib import Path

import fastapi
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware

from config.config import project_root, MinIoConfig
from import_process.main_graph import MainGraphRunner
from tool.logger import logger
from tool.minio_client_tool import minio_client, get_minio_client
from tool.task_utils import get_task_info, add_running_task, add_done_task, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

app = FastAPI(
    title="掌柜智库-导入模块API",
    description="掌柜智库-导入模块API",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def run_import_graph(task_id, local_dir, local_file_path):
    try:
        init_state = {
            "task_id": task_id,
            "local_dir": local_dir,
            "local_file_path": local_file_path
        }

        update_task_status(task_id, TASK_STATUS_PROCESSING)
        MainGraphRunner.create_and_run(init_state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        logger.error(f"执行graph异常，task_id={task_id}")
        update_task_status(task_id, TASK_STATUS_FAILED)


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(..., description="上传的pdf文件")):
    # 第一步生成task_id
    task_id = str(uuid.uuid4())

    # 创建目标目录
    local_dir = f"{project_root}/data/{datetime.now().strftime('%Y%m%d')}"
    local_dir_obj = Path(local_dir)
    if not local_dir_obj.exists():
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    local_file_path = local_dir_obj / file.filename

    with open(local_file_path, "wb") as f:
        shutil.copyfileobj(file.file, f, 1024 * 1024)
    logger.info(f"文件上传成功，保存路径为：{local_file_path}")

    # 上传文件的状态追踪
    add_running_task(task_id, "upload_file")

    # 备份文件到minIO中
    minio_client = get_minio_client()
    minio_client.fput_object(
        bucket_name=MinIoConfig.minio_bucket_name,
        object_name=f"upload_file/{datetime.now().strftime('%Y%m%d')}/{task_id}/{file.filename}",
        file_path=local_file_path
    )
    logger.info(
        f"文件上传到minio成功，保存路径为：pdf_file/{datetime.now().strftime('%Y%m%d')}/{task_id}/{file.filename}")

    # 上传文件的状态追踪
    add_done_task(task_id, "upload_file")

    # 进行后台任务
    background_tasks.add_task(run_import_graph, task_id=task_id, local_dir=local_dir, local_file_path=local_file_path)

    # 返回task_id
    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_status(task_id: str = fastapi.Path(..., description="任务ID")):
    return get_task_info(task_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
