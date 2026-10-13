import sys
import os

from app.conf.milvus_config import milvus_config
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


def node_search_embedding(state):
    """
    核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
    流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
    :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                  {
                      "session_id": str,        # 会话唯一标识
                      "rewritten_query": str,   # step3改写后的完整用户问题（含商品名）
                      "item_names": list[str],  # step6已确认的标准化商品名列表
                      "is_stream": bool/None    # 是否为流式响应，可选
                  }
    :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
             {
                 "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                 # 每个元素为一条匹配的向量数据，含业务字段
             }
    """
    logger.info("---search_milvus 开始处理---")
    add_running_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])
    #分别获取rewritten_query和item_names
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    #判断item_names是否为空
    if not item_names:
        logger.warning()
    #将重写问题转换为向量
    embeddings = generate_embeddings(rewritten_query)
    #分别获取稠密向量和稀疏向量
    dense_vector = embeddings["dense"][0]
    sparse_vector = embeddings["sparse"][0]
    #将item_names中的所有产品主体拼接为字符串
    item_str = ",".join(f"'{item}'"for item in item_names)
    #将item_name作为检索条件，拼接
    expr = f"item_names in [{item_str}]"
    #设置稠密向量和稀疏向量的检索方式
    reqs = create_hybrid_search_requests(
        expr=expr,
        sparse_vector=sparse_vector,
        dense_vector=dense_vector,
        limit=5
    )
    #获取milvus客户端
    milvus_client = get_milvus_client()
    #进行混合检索
    results = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.8,0.2),
        norm_score=True,
        limit=3,
        output_fields=["chunk_id","item_name","content"]
    )
        
    # 标记当前任务完成，更新任务状态
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 6. 构造并返回结果：若检索结果非空，取res[0]（适配Milvus批量搜索格式），否则返回空列表
    # res[0]为当前单条查询的检索结果，包含TOP5匹配的向量数据及业务字段
    return {"embedding_chunks": results[0] if results else []}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")
        
        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")
            
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)

