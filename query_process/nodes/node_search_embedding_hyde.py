import json

from langchain.chat_models import init_chat_model

from config.config import MilvusConfig, LLMConfig
from config.prompt import HYDE_PROMPT
from query_process.base import NodeBase
from query_process.nodes.node_search_embedding import NodeSearchEmbedding
from query_process.state import QueryGraphState
from tool.bge_m3_client_tool import get_bge_m3_embedding
from tool.json_format_tool import json_format
from tool.logger import logger
from tool.milvus_client_tool import get_reqs, search_hybrid


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        if not rewritten_query:
            logger.error("rewritten_query必须存在")
            raise ValueError("rewritten_query必须存在")
        if not item_names:
            logger.error("item_names必须有值")
            raise ValueError("item_names必须有值")

        llm = init_chat_model(
            model=LLMConfig.item_model,
            model_provider="openai",
            api_key=LLMConfig.openai_api_key,
            base_url=LLMConfig.openai_api_base,
            temperature=LLMConfig.llm_default_temperature
        )

        message = [
            {"role": "user", "content": HYDE_PROMPT.format(rewritten_query=rewritten_query)},
        ]

        result = llm.invoke(message)
        # logger.info(result.content)

        merged_answer = f"{rewritten_query}{result.content}"

        embedding = get_bge_m3_embedding([merged_answer])
        collection_name = MilvusConfig.chunks_collection
        dense_data = embedding.get("dense")[0]
        sparse_data = embedding.get("sparse")[0]

        # 整理item_names
        item_names = [
            item.replace("\\", '\\\\').replace("'", "\\'").replace('"', '\\"')
            for item in item_names
        ]

        # 混合搜索添加过滤字段,in后面必须是字符串
        expr = f"item_name in {json.dumps(item_names)}"

        reqs = get_reqs(
            dense_data,
            sparse_data,
            dense_anns_field="dense_vector",
            sparse_anns_field="sparse_vector",
            expr=expr
        )

        res = search_hybrid(
            collection_name=collection_name,
            reqs=reqs,
            ranker=(0.8, 0.2),
            output_fields=["id", "content", "item_name", "title", "file_title"]
        )
        # logger.info(json_format(res[0]))

        hyde_embedding_chunks = [
            {
                **item.get("entity"),
                "score": item.get("distance"),
                "source": "local"
            }
            for item in res[0]
        ]

        return {
            "hyde_embedding_chunks": hyde_embedding_chunks
        }


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于BrotherHAK180烫金机如何使用",
        "item_names": ["BrotherHAK180烫金机"]
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(json_format(result))
