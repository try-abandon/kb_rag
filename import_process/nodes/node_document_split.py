import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from import_process.base import NodeBase
from import_process.nodes.node_md_img import NodeMDImg
from import_process.state import ImportGraphState
from tool.json_format_tool import json_format
from tool.logger import logger


class NodeDocumentSplit(NodeBase):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        # 判断md文件路径是否存在
        md_file_path = state.get("md_path", "")
        if not md_file_path:
            logger.error(f"md文件路径必须提供")
            raise ValueError(f"md文件路径必须提供")

        # 判断文件是否存在
        md_file_path_obj = Path(md_file_path)
        if not md_file_path_obj.exists():
            logger.error(f"md文件不存在")
            raise ValueError(f"md文件不存在")

        # 判断是否给了文件名称
        file_title = state.get("file_title", "")
        if not file_title:
            file_title = md_file_path_obj.stem

        # 读取文件中的内容
        with open(md_file_path_obj, 'r', encoding="utf-8") as f:
            md_content = f.read()

        # 统一不同系统中文件中的换行符号
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 按行切分文档
        md_line_list = md_content.split("\n")

        # 代码块正则
        code_pattern = r"^(`{3,}|~{3,})"
        # 标题正则
        title_pattern = r'^\s*#{1,6}\s+.+'
        # 判断每一行是否在代码块中
        is_in_code_block = False
        # 获得代码块的起始符号
        masker = None
        # 当前下标
        current_index = 0
        # 区块列表
        block_list = []

        for index, line in enumerate(md_line_list):
            line = line.strip()
            match = re.match(code_pattern, line)

            # 判断是否在代码块中
            if match:
                if not is_in_code_block:
                    is_in_code_block = True
                    masker = match.group(1)
                    logger.info(f"代码块开始")
                else:
                    if masker == match.group(1):
                        is_in_code_block = False
                        masker = None
                        logger.info(f"代码块结束")

            # 不在代码块，判断是否是标题
            if not is_in_code_block and re.match(title_pattern, line):
                # 将该标题上面的内容列表作为一部分
                temp_list = md_line_list[current_index: index]

                # 将列表拼合成字符串，形成了包含标题和文档内容的一个content或单独文档
                content = "\n".join(temp_list)

                # 更新坐标
                current_index = index

                block_list.append({
                    "title": temp_list[0] if content.startswith("#") else "无标题",
                    "content": content,
                    "file_title": file_title
                })

        # 最后一个区块单独处理，因为他的后续没有标题
        block_list.append({
            "title": md_line_list[current_index],
            "content": "\n".join(md_line_list[current_index:]),
            "file_title": file_title
        })

        MAX_LENGTH = 300
        OVER_RAP = 30
        final_block_list = []

        spliter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            chunk_size=MAX_LENGTH,
            chunk_overlap=OVER_RAP
        )

        for block in block_list:
            title = block.get("title", "")
            content = block.get("content", "")

            # 真正的内容是需要去掉标题的
            real_content = content[len(title):] if content.startswith("#") else content

            # 如果切分内容的长度小于切分长度就不需要切分
            if len(real_content) < MAX_LENGTH:
                final_block_list.append({
                    **block,
                    "part":0
                })
                continue

            # 如果遇到表格则不进行切分
            if "<table" in real_content:
                final_block_list.append({
                    **block,
                    "part": 0
                })
                continue

            split_block_list = spliter.split_text(real_content)
            for index, split_block in enumerate(split_block_list):
                final_block_list.append({
                    "title": title,
                    "content": title + "\n\n" + split_block,
                    "file_title": file_title,
                    "part": index
                })

        # 备份文件
        with open(md_file_path_obj.parent / "chunks.json", 'w', encoding='utf-8') as f:
            f.write(json_format(final_block_list))

        return {
            "chunks": final_block_list
        }


if __name__ == '__main__':
    node = NodeDocumentSplit()
    init_state = {
        "md_path": "../../data/hak180产品安全手册/hak180产品安全手册.md",
        "file_title": "hak180产品安全手册"
    }
    result = node(init_state)
    logger.info(json_format(result))
