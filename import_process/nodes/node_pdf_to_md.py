import shutil
import time
import zipfile
from pathlib import Path

import requests

from config.config import MineruConfig
from import_process.base import NodeBase
from import_process.state import ImportGraphState
from tool.json_format_tool import json_format
from tool.logger import logger


class NodePDFToMD(NodeBase):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """
 
    name = "node_pdf_to_md"

    def check_path(self, state: ImportGraphState):
        pdf_file_path = state.get("pdf_path", "")
        # 检查输入路径是否为空
        if not pdf_file_path:
            logger.error(f"文件路径必须提供")
            raise ValueError(f"文件路径必须提供")

        pdf_file_path_obj = Path(pdf_file_path)
        # 检查输入文件是否存在
        if not pdf_file_path_obj.exists():
            logger.error(f"文件不存在")
            raise ValueError(f"文件不存在")

        local_dir = state.get("local_dir", "")
        # 检查输出路径是否存在
        if not local_dir:
            logger.error(f"输出路径必须提供")
            raise ValueError(f"输出路径必须提供")

        # 输出路径不存在则创建目录
        local_dir_obj = Path(local_dir)
        if not local_dir_obj.exists():
            local_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_file_path, pdf_file_path_obj, local_dir_obj

    def upload_pdf(self, pdf_file_path, pdf_file_path_obj):
        token = MineruConfig.mineru_token
        url = f"{MineruConfig.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": f"{pdf_file_path_obj.name}", "data_id": "abcd"}
            ],
            "model_version": "vlm"
        }
        file_path = [f"{pdf_file_path}"]

        # 获得请求
        response = requests.post(url, headers=header, json=data)

        # 判断文件是否上传成功
        if response.status_code != 200:
            logger.error(f"上传PDF文件失败")
            raise Exception(f"上传PDF文件失败")
        logger.info(f"上传PDF文件成功")
        result = response.json()

        # 判断数据是否请求成功
        if result["code"] != 0:
            logger.error(f"请求数据失败")
            raise Exception(f"请求数据失败")
        logger.info(f"请求数据成功")
        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]

        for i in range(0, len(urls)):
            with open(file_path[i], 'rb') as f:
                res_upload = requests.put(urls[i], data=f)
                if res_upload.status_code == 200:
                    logger.info(f"{urls[i]}上传成功")
                else:
                    logger.info(f"{urls[i]}上传失败")

        return batch_id

    def obtain_the_analysis_result(self, batch_id):
        token = MineruConfig.mineru_token
        batch_id = batch_id
        url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # 轮询获得PDF文件压缩后的文件地址
        total_time = 100
        use_time = 0
        while True:
            start_time = time.time()
            try:
                res = requests.get(url, headers=header)
                if res.status_code != 200:
                    logger.error(f"获取PDF文件解析结果请求失败")
                    raise Exception(f"获取PDF文件解析结果请求失败")

                result = res.json()
                if result["code"] != 0:
                    logger.error(f"获取PDF文件解析结果请求数据失败")
                    raise Exception(f"获取PDF文件解析结果请求数据失败")

                # 通过官网的接口文档获得数据
                data = result["data"]["extract_result"][0]
                if data["state"] != "done":
                    logger.error(f"PDF文件处理中尚未完成")
                    raise Exception(f"PDF文件处理中尚未完成")

                logger.info(f"PDF文件处理完成")
                zip_file_url = data["full_zip_url"]
                return zip_file_url
            except Exception as e:
                logger.error(f"获取PDF文件解析结果失败: {e}")
                end_time = time.time()
                use_time += end_time - start_time
                if use_time > total_time:
                    logger.error(f"获取PDF文件解析结果超时")
                    raise Exception(f"获取PDF文件解析结果超时")
                continue

    def download_and_extract(self, zip_file_url, pdf_file_path_obj, local_dir_obj):
        md_zip_res = requests.get(zip_file_url)
        if md_zip_res.status_code != 200:
            logger.error(f"下载PDF文件请求失败")
            raise Exception(f"下载PDF文件请求失败")

        # 获得文件内容和写入的路径
        md_zip_content = md_zip_res.content
        md_zip_file_path_obj = local_dir_obj / f"{pdf_file_path_obj.stem}.zip"

        # 写入
        with open(md_zip_file_path_obj, "wb") as f:
            f.write(md_zip_content)

        # 解压zip文件

        unzip_file_content = zipfile.ZipFile(md_zip_file_path_obj)

        # 解压到的路径
        unzip_file_path_obj = local_dir_obj / f"{pdf_file_path_obj.stem}"

        # 判断文件目录是否存在,如果存在就删除
        if unzip_file_path_obj.exists():
            shutil.rmtree(unzip_file_path_obj)

        # 创建目录
        unzip_file_path_obj.mkdir(parents=True, exist_ok=True)

        # 真正的把解压的内容，放到这个目录
        unzip_file_content.extractall(unzip_file_path_obj)

        # 解压完成后，重命名
        origin_md_path_obj = unzip_file_path_obj / "full.md"
        # 落盘
        new_md_path_obj = origin_md_path_obj.with_name(f"{pdf_file_path_obj.stem}.md")
        origin_md_path_obj.rename(new_md_path_obj)

        # 读取Markdown文件内容 存储state
        with open(new_md_path_obj, 'r', encoding="utf-8") as f:
            md_content = f.read()

        return md_content, new_md_path_obj

    def process(self, state: ImportGraphState):
        # 检查路径和文件是否存在
        pdf_file_path, pdf_file_path_obj, local_dir_obj = self.check_path(state)

        # 上传PDF文件获得batch_id
        batch_id = self.upload_pdf(pdf_file_path, pdf_file_path_obj)

        # 获得PDF文件的解析结果地址
        zip_file_url = self.obtain_the_analysis_result(batch_id)

        # 下载并解压zip文件到目标地址
        md_content, new_md_path_obj = self.download_and_extract(zip_file_url, pdf_file_path_obj, local_dir_obj)

        return {
            "md_path": str(new_md_path_obj),
            "md_content": md_content
        }


if __name__ == '__main__':
    node = NodePDFToMD()
    init_state = {
        "pdf_path": "../../data/hak180产品安全手册.pdf",
        "local_dir": "../.././data"
    }
    result = node(init_state)
    logger.info(json_format(result))
