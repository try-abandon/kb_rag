from pathlib import Path

from import_process.base import NodeBase
from import_process.state import ImportGraphState
from tool.logger import logger
from tool.json_format_tool import json_format


class NodeEntry(NodeBase):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        local_file_path = state.get("local_file_path", "")

        # 防御性编程
        if not local_file_path:
            logger.error(f"{local_file_path}必须提供")
            raise ValueError(f"{local_file_path}必须提供")

        local_file_path_obj = Path(local_file_path)
        if not local_file_path_obj.exists():
            logger.error(f"{local_file_path}文件不存在")
            raise ValueError(f"{local_file_path}文件不存在")

        # 判断文件是pdf类型还是md类型，暂不支持其余类型
        file_title = local_file_path_obj.stem
        suffix = local_file_path_obj.suffix
        if suffix.lower() == ".pdf":
            return {
                "is_pdf_read_enabled": True,
                "pdf_path": local_file_path,
                "file_title": file_title,
            }
        elif suffix.lower() == ".md":
            return {
                "is_md_read_enabled": True,
                "md_path": local_file_path,
                "file_title": file_title,
            }
        else:
            logger.error(f"{local_file_path}文件是不支持的文件类型")
            raise ValueError(f"{local_file_path}文件是不支持的文件类型")


if __name__ == '__main__':
    node = NodeEntry()
    init_state = {
        "local_file_path": "../../data/hak180产品安全手册.pdf"
    }
    result = node(init_state)
    logger.info(json_format(result))
