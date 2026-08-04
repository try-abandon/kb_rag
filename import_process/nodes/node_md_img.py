import base64
import os
import re
import time
from collections import deque
from pathlib import Path

from langchain.chat_models import init_chat_model

from config.config import LLMConfig
from import_process.base import NodeBase
from import_process.state import ImportGraphState
from tool.json_format_tool import json_format
from tool.logger import logger


class NodeMDImg(NodeBase):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def get_md_content(self, state: ImportGraphState):
        # 判断文件路径是否存在
        md_path = state.get("md_path", "")
        if not md_path:
            logger.error(f"md文件路径必须提供")
            raise ValueError(f"md文件路径必须提供")

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            logger.error(f"md文件不存在")
            raise ValueError(f"md文件不存在")

        with open(md_path_obj, 'r', encoding="utf-8") as f:
            md_content = f.read()

        if not md_content:
            logger.error(f"md文件内容为空")
            raise ValueError(f"md文件内容为空")

        return md_content, md_path_obj

    def get_image_with_context_list(self, md_content, md_images_path_obj, images_name_list):
        # 所有的图片后缀类型
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        MAX_LENGTH = 300
        image_with_context_list = []
        for image_name in images_name_list:
            if Path(image_name).suffix.lower() not in IMAGE_EXTENSIONS:
                logger.error(f"图片格式错误")
                continue

            # 创建正则表达式
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r"\)")
            # 在md_content中匹配图片
            match = pattern.search(md_content)

            if not match:
                logger.error(f"图片{image_name}未找到,不在md文档的内容中")
                continue

            # 获得图片的起始位置
            start, end = match.span()
            # 获得前文内容
            pre_content = md_content[max(0, start - MAX_LENGTH):  start]
            # 获得后文内容
            post_content = md_content[end: min(len(md_content), end + MAX_LENGTH)]

            image_path = str(md_images_path_obj / image_name)

            image_with_context_list.append({
                "image_path": image_path,
                "pre_content": pre_content,
                "post_content": post_content,
                "image_name": image_name
            })
        return image_with_context_list

    def get_image_summary_list(self, image_with_context_list):
        llm = init_chat_model(
            model=LLMConfig.llm_default_model,
            model_provider="openai",
            base_url=LLMConfig.openai_api_base,
            api_key=LLMConfig.openai_api_key,
            temperature=LLMConfig.llm_default_temperature,
        )

        # 创建双向队列和时间戳
        dq = deque(maxlen=30)
        current_time = time.time()
        QUEUE_SAVE_TIME = 30
        image_with_summary_list = []

        # 滑动门算法确保请求输入不大于模型阈值
        for image_with_context in image_with_context_list:
            # 每一次都清除队列中超时的数据，出队
            while dq and current_time - dq[0] > QUEUE_SAVE_TIME:
                dq.popleft()

            # 判断队列是否满了
            if dq and len(dq) == dq.maxlen:
                sleep_time = QUEUE_SAVE_TIME - (current_time - dq[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    current_time = time.time()
                    while dq and current_time - dq[0] > QUEUE_SAVE_TIME:
                        dq.popleft()

            dq.append(current_time)

            # 对图片进行base_64编码
            with open(image_with_context.get("image_path"), 'rb') as r:
                image_data = r.read()
                base64_str = base64.b64encode(image_data).decode('utf-8')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64_str,
                            },
                        },
                        {"type": "text", "text": f"""
                                                这是一张图片，图片上文部分为"{image_with_context.get("pre_content")}"，
                                                下文部分为"{image_with_context.get("post_content")}"，
                                                请用中文简要总结这张图片的摘要,字数在50字以内。"""},
                    ],
                },
            ]

            result = llm.invoke(messages)
            image_with_summary_list.append({
                "image_name": image_with_context.get("image_name"),
                "image_path": image_with_context.get("image_path"),
                "summary": result.content
            })

        return image_with_summary_list

    def process(self, state: ImportGraphState):
        # 获得md文件内容
        md_content, md_path_obj = self.get_md_content(state)

        # 获得图片路径
        md_images_path_obj = md_path_obj.parent / "images"
        # 判断图片路径是否为空
        if not md_images_path_obj.exists():
            logger.error(f"图片文件夹不存在")
            return {
                "md_content": md_content,
            }

        # 列出图片文件夹中所有文件和文件夹的名字
        images_name_list = os.listdir(md_images_path_obj)
        # 判断图片文件夹内是否为空
        if not images_name_list:
            logger.error(f"图片文件夹为空")
            return {
                "md_content": md_content,
            }

        # 获得包含图片上下文信息的字典列表
        image_with_context_list = self.get_image_with_context_list(md_content, md_images_path_obj, images_name_list)

        # 获取图片摘要
        image_with_summary_list = self.get_image_summary_list(image_with_context_list)

        return image_with_summary_list


if __name__ == '__main__':
    node = NodeMDImg()
    init_state = {
        "md_path": "../../data/hak180产品安全手册/hak180产品安全手册.md"
    }
    result = node(init_state)
    logger.info(json_format(result))
