import sys
from typing import List, Dict, Any

from torch._dynamo import source

from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log, step_log


# RRF节点 将向量检索结果进行处理，转换为只包含实体信息的列表
def _as_entity_list(chunks) -> List[Dict[str, Any]]:
    """
    将上游节点输出统一规整为 entity dict 列表。
    兼容：
    - dict: {"entity": {..属性名和对应的字.}, "distance": ...} 或直接就是 {...}
    - pymilvus Hit: 不是 dict，但通常支持 hit.get("entity") 或 hit.entity
    - 其他：当作 chunk_id
    """
    #创建存储转换之后实体字典的列表
    out: List[Dict[str, Any]] = []
    #对chunks进行遍历
    for chunk in chunks:
        #判断chunk是否为空
        if not chunk:
            continue
        #创建存储转换之后实体字典的变量
        final_entity = {}
        # 情况A：处理 Milvus 返回的 Hit 对象（含 entity、chunk_id、distance）
        if hasattr(chunk, "entity") and hasattr(chunk, "chunk_id"):
            #获取hit对象的entity属性值
            entity = chunk.entity
            #判断entity是否有to_dict属性
            if hasattr(entity, "to_dict"):
                final_entity = entity.to_dict()
            elif isinstance(entity, dict):
                final_entity = entity.copy()
            else:
                try:
                    final_entity = dict(entity)
                except:
                    pass
            #在实体字典中补充chunk_id
            if "chunk_id"  not in final_entity:
                final_entity["chunk_id"] = chunk.chunk_id
            # 在实体字典中补充score
            if hasattr(chunk, "distance"):
                final_entity["score"] = chunk.distance
        # 情况B：doc已经是字典（模拟数据 / 已格式化数据）
        elif isinstance(chunk, dict):
            #获取entity
            entity = chunk["entity"]
            #判断entity是否是字典
            if isinstance(entity, dict):
                final_entity = entity.copy()
                # 补充 ID 和分数
                if "chunk_id"  not in final_entity:
                    final_entity["chunk_id"] = chunk["chunk_id"]
                # 在实体字典中补充score
                if "distance" not in final_entity:
                    final_entity["score"] = chunk["distance"]
            else:
                final_entity = chunk

        # 情况C：支持 .get() 方法的其他对象
        elif hasattr(chunk, "get"):
            entity = chunk.get("entity") or chunk
            if isinstance(entity, dict):
                final_entity = entity
        # 判断final_entity是否为空，是否为字典
        if final_entity and isinstance(final_entity, dict):
            out.append(final_entity)
    return out


@step_log("step_2_rrf")
def step_2_rrf(
        source_weights: list,
        k: int = 60,
        max_results: int = None,
) -> List[tuple]:
    """
    通用带权重的RRF算法实现
    :param source_weights:  列表，每个元素是(来源文档列表, 权重)的元组
                            例如: [([doc1, doc2], 1.0), ([doc2, doc3], 0.8)]
    :param k:     RRF 常数，默认 60。用于平滑排名影响，避免高排名文档占据过大优势。
    :param max_results: 只返回前 N 个，None 表示全部
    :return:      [(元素, RRF 得分), ...] 按得分降序排列
    """
    #创建存储分数和数据的字典
    score_map = {}
    chunk_map = {}
    #对source_weights进行遍历
    for chunks,weight in source_weights:
        #对chunks进行遍历,rank表示排名
        for rank,chunk in enumerate(chunks,start=1):
            #获取chunk_id
            chunk_id = chunk["chunk_id"]
            #计算chunk_id所对应数据的分数
            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (rank + k))
            #存储chunk_id所对应的数据
            chunk_map.setdefault(chunk_id, chunk)
    #创建存储数据以及所对应分数的列表
    merged_chunks = []
    #对score_map进行遍历
    for chunk_id,score in score_map.items():
        #从chunk_map中通过chunk_id获取所对应的分数
        chunk = chunk_map.get(chunk_id)
        #存储数据以及所对应的分数
        merged_chunks.append((chunk,score))
    #对merged_chunks进行排序
    merged_chunks.sort(key=lambda item: item[1], reverse=True)
    #对merged_chunks进行截断，保留max_results个结果
    merged_chunks = merged_chunks[:max_results]
    return merged_chunks



@node_log("node_rrf")
def node_rrf(state):
    """
    RRF (Reciprocal Rank Fusion) 倒数排名融合节点
    
    功能：
    将来自不同检索源（如 Embedding 检索、HyDE 检索、知识图谱检索等）的结果进行融合排序。
    RRF 是一种无需训练的算法，仅根据文档在不同列表中的排名来计算最终得分。
    
    步骤：
    1. 提取各路检索结果：从 state 中获取 embedding_chunks 和 hyde_embedding_chunks。
    2. 结果标准化：将不同格式的检索结果统一转换为包含 chunk_id 的实体列表。
    3. 设置权重：为不同来源分配权重（当前配置：Embedding=1.0, HyDE=1.0）。
    4. 执行 RRF：计算融合分数并重新排序。
    5. 结果截断：保留 Top K 个结果。
    6. 更新状态：将融合后的结果存入 state["rrf_chunks"]。
    """
    logger.info("---RRF (倒数排名融合) 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 第一步：获取上游检索节点返回的文档
    # 上游检索节点（Milvus hybrid_search）返回的通常是 hit 列表：
    #  {"entity": {...fields...}, "distance": ...}
    # RRF 需要使用 chunk_id 做去重与计分，因此这里必须保留 entity（而不是仅抽取 content 字符串）。
    # 步骤1：将向量检索结果进行处理，转换为只包含实体信息的列表
    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))

    #设置多路检索的结果的权重
    source_weights = [
        (embedding_chunks,1.0),
        (hyde_embedding_chunks,1.0)
    ]
    #步骤2：进行rrf融合以及排序
    rrf_result = step_2_rrf(source_weights,k=60,max_results=None)

    #步骤3：获取rrf融合排序之后的数据
    rrf_chunks = [chunk for chunk,score in rrf_result]

    # 记录任务结束
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    return {"rrf_chunks": rrf_chunks}


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_rrf 本地测试")
    print("="*50)

    # 1. 构造假数据 (模拟真实数据库字段)
    # 模拟 Embedding 检索结果 
    mock_embedding_chunks = [
        {
            "id": "doc_1", 
            "pk": "pk_1", 
            "file_title": "操作手册_v1.pdf", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容1：打开电源开关...", 
            "score": 0.9
        },
        {
            "id": "doc_2", 
            "pk": "pk_2", 
            "file_title": "维修指南.pdf", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容2：遇到故障请联系...", 
            "score": 0.8
        },
        {
            "id": "doc_3", 
            "pk": "pk_3", 
            "file_title": "参数表.xlsx", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容3：电压220V...", 
            "score": 0.7
        }
    ]
    
    # 模拟 HyDE 检索结果 (包含 3 个文档，顺序不同，且有新文档 doc_4)
    mock_hyde_chunks = [
        {
            "id": "doc_3", 
            "pk": "pk_3", 
            "file_title": "参数表.xlsx", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容3：电压220V...", 
            "score": 0.85
        }, 
        {
            "id": "doc_1", 
            "pk": "pk_1", 
            "file_title": "操作手册_v1.pdf", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容1：打开电源开关...", 
            "score": 0.82
        }, 
        {
            "id": "doc_4", 
            "pk": "pk_4", 
            "file_title": "安全须知.docx", 
            "item_name": "HAK 180 烫金机", 
            "content": "内容4：操作时请佩戴手套...", 
            "score": 0.75
        }
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        # 运行节点
        result = node_rrf(mock_state)
        
        # 验证结果
        rrf_chunks = result.get("rrf_chunks", [])
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)
        
        # 打印详细排名
        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            # 注意：返回结果中可能没有 chunk_id 字段，而是 id
            doc_id = doc.get('chunk_id') or doc.get('id')
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")

        # 验证预期逻辑：
        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]
        
        if "doc_1" in ids and "doc_3" in ids:
            print("\n[PASS] 交叉文档 (doc_1, doc_3) 成功融合保留")
        else:
            print("\n[FAIL] 交叉文档丢失")
            
        if len(ids) == 4:
            print("[PASS] 并集数量正确 (3+3-2重叠=4)")
        else:
            print(f"[FAIL] 并集数量错误: 期望4, 实际{len(ids)}")
            
        print("="*50)

    except Exception as e:
        logger.exception("测试运行期间发生未捕获异常: {}", e)
