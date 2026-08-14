import json

from config.config import MilvusConfig
from query_process.base import NodeBase
from query_process.state import QueryGraphState
from tool.bge_m3_client_tool import get_bge_m3_embedding
from tool.json_format_tool import json_format
from tool.logger import logger
from tool.milvus_client_tool import get_reqs, search_hybrid


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState):
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names")
        if not rewritten_query:
            logger.error("rewritten_query必须存在")
            raise ValueError("rewritten_query必须存在")

        if not item_names:
            logger.error("item_names必须有值")
            raise ValueError("item_names必须有值")

        embedding = get_bge_m3_embedding([rewritten_query])
        collection_name = MilvusConfig.chunks_collection
        dense_data = embedding.get("dense")[0]
        sparse_data = embedding.get("sparse")[0]

        # 混合搜索添加过滤字段,in后面必须是字符串
        expr = f"item_name in {json.dumps(item_names, ensure_ascii=False)}"

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
        # logger.info(json_format(res))

        embedding_chunks = [
            {
                **item.get("entity"),
                "score": item.get("distance"),
                "source": "local"
            }
            for item in res[0]
        ]

        return {
            "embedding_chunks": embedding_chunks
        }


if __name__ == "__main__":
    init_state = {
        "rewritten_query": "关于BrotherHAK180烫金机如何使用",
        "item_names": ["BrotherHAK180烫金机"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(json_format(result))
