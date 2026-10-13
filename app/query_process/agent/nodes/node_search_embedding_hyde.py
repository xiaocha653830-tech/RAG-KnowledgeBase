# HyDE节点
import sys

from app.core.logger import step_log
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

@step_log("step_1_create_hyde_doc")
def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    阶段1：利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
    HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
    从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。

    :param rewritten_query: 用户改写后的查询语句
    :return: LLM生成的假设性文档内容
    """
    #获取提示词
    prompt = load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    #获取大模型对象
    llm = get_llm_client()
    #调用大模型
    response = llm.invoke(prompt)
    #获取模型所返回的结果，即假设性文档
    hyde_doc = response.content
    return hyde_doc

@step_log("step_2_search_embedding_hyde")
def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10, #稠密向量和稀疏向量检索的数据量
    limit: int = 5, #混合检索的数据量
    ranker_weights=(0.8, 0.2),  # 调整默认权重以偏向稠密向量 (0.8, 0.2)
    norm_score: bool = True,    # 默认开启归一化
    output_fields=["chunk_id", "content", "item_name"],
):
    """
    阶段2：利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。
    
    :param rewritten_query: 改写后的查询
    :param hyde_doc: Step 1 生成的假设性文档
    :param item_names: 商品名称列表，用于元数据过滤 (item_name in [...])
    :param req_limit: Milvus 搜索时的候选召回数量
    :param top_k: 最终返回的 Top K 结果数量
    :param ranker_weights: 混合检索权重 (Dense, Sparse)
    :param norm_score: 是否对分数进行归一化
    :param output_fields: 返回结果中包含的字段
    :return: 检索结果列表
    """
    #拼接rewritten_query和hyde_doc
    text = rewritten_query + " " + hyde_doc
    #获取text所对应的稠密稀疏向量
    embeddings = generate_embeddings([text])
    dense_vector = embeddings["dense"][0]
    sparse_vector = embeddings["sparse"][0]
    #拼接item_name作为检索条件
    expr_data = ",".join(f"'{item}'" for item in item_names)
    expr = f"item_name in [{expr_data}]"
    #设置稠密和稀疏的检索方式
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=req_limit,
    )
    #获取milvus客户端
    milvus_client = get_milvus_client()
    #进行混合检索
    result = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=ranker_weights,
        norm_score=norm_score,
        limit=limit,
        output_fields=output_fields,
    )
    return result


def node_search_embedding_hyde(state):
    """
    HyDE (Hypothetical Document Embedding) 检索节点
    核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

    执行步骤：
    1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
    2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
    3. 混合检索 (Step 2)：
       - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
       - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
    4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。

    :param state: 会话状态字典，包含 session_id, rewritten_query, item_names 等
    :return: 包含 hyde_embedding_chunks (检索结果) 和 hyde_doc (假设文档) 的字典
    """
    logger.info("---HyDE (假设文档检索) 节点开始处理---")
    # 记录任务开始状态
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    #判断rewritten_query是否为空
    if not rewritten_query:
        rewritten_query = state.get("original_query")
    if not rewritten_query:
        logger.warning("假设性文档缺失用户问题")
        return {"hyde_embedding_chunks": []}
   #判断item_names是否为空
    if not item_names:
        logger.warning("假设性文档缺失item_names")
        return {"hyde_embedding_chunks": []}
    #步骤1：通过rewritten_query生成假设性文档
    hyde_doc = ""
    try:
        hyde_doc = step_1_create_hyde_doc(rewritten_query)
    except Exception as e:
        logger.warning("获取假设性文档失败，{}", e)
    try:
        #步骤2：通过假设性文档和初始问题转化为向量检索数据
        result = step_2_search_embedding_hyde(
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            limit=5,
        )
        return {"hyde_embedding_chunks": result[0] if result else []}
    except Exception as e:
        logger.error("获取假设性文档检索结果失败，{}", e)
        return {"hyde_embedding_chunks": []}
    finally:
        # 无论成功失败，均标记任务结束
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        logger.info("---HyDE 节点处理结束---")


# if __name__ == "__main__":
#     # 本地测试代码
#     print("\n" + "=" * 50)
#     print(">>> 启动 node_search_embedding_hyde 本地测试")
#     print("=" * 50)
#
#     # 模拟输入状态
#     mock_state = {
#         "session_id": "test_hyde_session_001",
#         "original_query": "HAK 180 烫金机怎么操作？",
#         "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
#         "item_names": ["HAK 180 烫金机"],
#         "is_stream": False
#     }
#
#     try:
#         # 运行节点
#         result = node_search_embedding_hyde(mock_state)
#
#         print("\n" + "="*50)
#         print(">>> 测试结果摘要:")
#         print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
#         if result.get("hyde_doc"):
#             print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")
#
#         chunks = result.get("hyde_embedding_chunks", [])
#         print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
#         if chunks:
#             print(f"Top Chunk Score: {chunks[0].get('distance')}")
#         print("="*50)
#
#     except Exception as e:
#         logger.exception("测试运行期间发生未捕获异常: {}", e)
