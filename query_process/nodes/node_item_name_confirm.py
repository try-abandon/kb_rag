# atguigu/query_process/nodes/node_item_name_confirm.py

import json
from typing import Any

from langchain.chat_models import init_chat_model
from openai import api_key, base_url

from config.config import LLMConfig, MilvusConfig
from config.prompt import ITEM_NAME_EXTRACT_SYSTEM_PROMPT, ITEM_NAME_EXTRACT_TEMPLATE
from query_process.base import NodeBase
from query_process.state import QueryGraphState
from tool.bge_m3_client_tool import get_bge_m3_embedding
from tool.json_format_tool import json_format
from tool.logger import logger
from tool.milvus_client_tool import milvus_client, get_milvus_client, get_reqs, search_hybrid
from tool.mongo_client_tool import add_or_update_history, get_recent_history_list, update_item_names_and_query


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def backfill_historical_data(self, answer: str, final_item_names: list[Any], message_id, rewritten_query,
                                 session_id) -> Any:
        # 判断answer是否有值来确定接下来是否添加历史节点
        # 如果有值那么回填历史数据,没有值不进行处理
        if answer:
            message_id = add_or_update_history(session_id, "assistant", answer)

        history_list = get_recent_history_list(session_id)
        ids = [history.get("_id") for history in history_list]
        if ids:
            update_item_names_and_query(ids, final_item_names, rewritten_query)
        return message_id

    def get_align_item_names(self, final_match_item_names: list[Any], chat_item_names: list[Any]) -> tuple[str, list[Any]]:
        # 对齐名字
        answer = ""
        final_item_names = []
        if chat_item_names:
            confirm_item_names = [
                item.get("search_item_name")
                for item in final_match_item_names
                if item.get("score") >= 0.6
            ]

            option_item_names = [
                item.get("search_item_name")
                for item in final_match_item_names
                if 0.4 <= item.get("score") < 0.6
            ]

            if confirm_item_names:
                final_item_names = confirm_item_names
                answer = ""
            elif option_item_names:
                final_item_names = []
                answer = f"请选择一下景点中的一个进行推荐:{",".join(option_item_names)}"
            else:
                final_item_names = []
                answer = "我无法识别您选择的是什么景点"
        return answer, final_item_names

    def get_final_match_item_names(self, chat_item_names: list[Any]) -> list[Any]:
        # 将大模型得到的主体名字进行向量化，才能进行向量检索
        embedding = get_bge_m3_embedding(chat_item_names)
        collection_name = MilvusConfig.item_name_collection

        final_match_item_names = []
        # 可能生成多个item_name，所以进行遍历
        for idx, chat_item_name in enumerate(chat_item_names):
            dense_data = embedding.get("dense")[idx]
            sparse_data = embedding.get("sparse")[idx]

            reqs = get_reqs(
                dense_data,
                sparse_data,
                dense_anns_field="dense_vector",
                sparse_anns_field="sparse_vector"
            )

            res = search_hybrid(
                collection_name=collection_name,
                reqs=reqs,
                ranker=(0.8, 0.2),
                limit=10,
                output_fields=["item_name"]
            )

            match_item_names = [
                {
                    "original_query": chat_item_name,
                    "search_item_name": item.get("entity").get("item_name"),
                    "score": item.get("distance")
                }
                for item in res[0]
            ]
            final_match_item_names.extend(match_item_names)
        return final_match_item_names

    def get_chat_item_names(self, history_content_str: str, original_query: str) -> tuple[list[Any], Any]:
        llm = init_chat_model(
            model=LLMConfig.item_model,
            model_provider="openai",
            api_key=LLMConfig.openai_api_key,
            base_url=LLMConfig.openai_api_base,
            temperature=LLMConfig.llm_default_temperature
        )

        message = [
            {"role": "system", "content": ITEM_NAME_EXTRACT_SYSTEM_PROMPT},
            {"role": "user",
             "content": ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_content_str,
                                                          original_query=original_query)}
        ]

        result = llm.invoke(message)

        # 对模型的输出进行整理
        res_json = result.content
        # 将代码块转为正常文本
        if res_json.startswith("```json"):
            res_json = res_json.replace("```json", "").replace("```", "")

        # 反序列化，将json转化为dict
        res_dict = json.loads(res_json)

        chat_item_names = res_dict.get("item_names")
        rewritten_query = res_dict.get("rewritten_query")

        # 如果有item_names那么清晰其中的空白字符
        if chat_item_names:
            chat_item_names = [
                chat_item_name.replace(" ", "").replace("\n", "").replace("\t", "")
                for chat_item_name in chat_item_names
            ]
        else:
            chat_item_names = []

        # 如果不存在重写的问题，那么将原始问题传给rewritten_query
        if not rewritten_query:
            rewritten_query = original_query
        return chat_item_names, rewritten_query

    def get_history_str(self, state: QueryGraphState) -> tuple[str, str, str, Any]:
        session_id = state.get("session_id")
        if not session_id:
            logger.error("session_id必须提供")
            raise ValueError("session_id必须提供")

        original_query = state.get("original_query")
        if not original_query:
            logger.error("original_query必须传递")
            raise ValueError("original_query必须传递")

        message_id = add_or_update_history(session_id, "user", original_query)
        # 获取历史聊天记录十条
        history_list = get_recent_history_list(session_id)
        history_content_str = ""

        for history in history_list:
            role = history.get("role")
            text = history.get("text")
            content = f"{role}:{text}\n"
            history_content_str += content
        return history_content_str, message_id, original_query, session_id

    def process(self, state: QueryGraphState):
        # 获得历史消息的字符串
        history_content_str, message_id, original_query, session_id = self.get_history_str(state)

        # 获得ai识别出来的item_names与rewritten_query
        chat_item_names, rewritten_query = self.get_chat_item_names(history_content_str, original_query)

        # 获得向量化检索后匹配的item_names
        final_match_item_names = self.get_final_match_item_names(chat_item_names)

        # 获得对齐后的item_names
        answer, final_item_names = self.get_align_item_names(final_match_item_names, chat_item_names)

        # 回填历史数据
        message_id = self.backfill_historical_data(answer, final_item_names, message_id, rewritten_query, session_id)

        return {
            "message_id": message_id,
            "original_query": original_query,
            "answer": answer,
            "item_names": final_item_names,
            "rewritten_query": rewritten_query,
            "history": get_recent_history_list(session_id, limit=10)
        }


if __name__ == "__main__":
    # 模拟会话历史
    session_id = "test_001"
    add_or_update_history(session_id, "user", "咨询下烫金机。")
    add_or_update_history(session_id, "assistant", "您好。请问是哪个型号")
    add_or_update_history(session_id, "user", "hak180")
    add_or_update_history(session_id, "assistant", "具体有什么问题呢？")

    # 初始化图状态
    init_state = {
        "session_id": "test_001",
        "original_query": "咋用？"
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    logger.info(json_format(result))
