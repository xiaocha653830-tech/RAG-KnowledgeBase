from app.utils.task_utils import *
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger, step_log, node_log
import sys

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对） 分比例
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对） 分值
RERANK_GAP_ABS: float = 0.5

# Rerank节点（工作流入口）
@step_log("step_1_merge_docs")
def step_1_merge_docs(state):
    """
    阶段一：文档合并与标准化
    
    目标：将多路召回（本地知识库 + 联网搜索）的异构数据，统一合并为 Reranker 模型可处理的标准格式。
    
    输入来源：
    1. rrf_chunks (List[Dict]): 本地知识库检索结果（经 RRF 融合排序）。
       - 结构：包含 Milvus entity 信息的复杂字典或对象。
       - 关键字段：chunk_id, content, title/item_name。
    2. web_search_docs (List[Dict]): 联网搜索结果（经 MCP 搜索返回）。
       - 结构：包含搜索摘要的扁平字典。
       - 关键字段：snippet, title, url。
       
    输出结果 (List[Dict]):
    - 标准化文档列表，每项包含：
      - text: 用于重排序的核心文本（content 或 snippet）
      - title: 标题（用于增强语义或展示）
      - doc_id/chunk_id: 唯一标识（本地文档有，联网文档为 None）
      - url: 来源链接（本地为空，联网文档有）
      - source: 来源标记 ("local" 或 "web")
    """
    
    #分别获取状态中web_search_doc和rrf_chunks
    web_search_docs = state.get("web_search_docs") or []
    rrf_chunks = state.get("rrf_chunks") or []
    #创建存储最终合并数据的列表
    doc_items = []
    #遍历rrf_chunks将其中数据转换为固定格式
    for chunk in rrf_chunks:
        #获取chunk中存储的entity
        entity = chunk.get("entity") if isinstance(chunk, dict) and "entity" in chunk else chunk
        #判断entity是否是字典
        if not isinstance(entity, dict):
            continue
        #获取content
        content = entity.get("content")
        #判断content是否为空
        if not content:
            continue
        #分别获取chunk_id和title(item_name)
        chunk_id = entity.get("chunk_id") or entity.get("id")
        title = entity.get("title") or entity.get("item_name")

        #将数据转换为固定的结构并存储到doc_items
        doc_items.append(
            {
                "text": content,
                "doc_id": chunk_id,
                "chunk_id": chunk_id,  # 兼容旧逻辑保留字段
                "title": title,
                "url": "",
                "source": "local",
            }
        )

    #遍历web_search_docs将其中数据转换为固定格式
    for doc in web_search_docs:
        #分别获取网络搜索结果中的snippet摘要title标题url网址
        snippet = (doc.get("snippet") or doc.get("content")).strip()
        title = (doc.get("title") or "").strip()
        url = (doc.get("url") or "").strip()
        #使用固定结构存储数据
        doc_items.append(
            {
                "text": snippet,
                "doc_id": "",
                "chunk_id": "",  # 兼容旧逻辑保留字段
                "title": title,
                "url": url,
                "source": "web",
            }
        )
    return doc_items


@step_log("step_2_rerank_docs")
def step_2_rerank_docs(state, doc_items):
    """
    阶段二：对文档进行重排序
    - 输入 doc_items：[{ text,doc_id}, ...]（由第一阶段产出）
    - 输出：在 state 中写入 reranked_docs（结构化列表）
    """
    #获取状态中的rewritten_query或者original_query
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    #判断rewritten_query和doc_items是否为空
    if not rewritten_query or not doc_items:
        return []
    #获取doc_items中的文本
    texts = [item["text"] for item in doc_items]
    try:
        #获取reranker模型对象
        reranker_model = get_reranker_model()
        #将数据转化为[[query,content]...]
        sentence_pairs = [[rewritten_query,text] for text in texts]
        #对数据进行重排序,返回的是每条数据的分数组成的列表
        scores = reranker_model.compute_score(sentence_pairs)
        #创建存储最终结果的列表
        score_docs = []
        #将scores、texts、doc_items进行压缩遍历
        for score,text,item in zip(scores,texts,doc_items):
            score_docs.append(
                {
                    "text": text,
                    "score": float(score),
                    "doc_id": item["doc_id"],
                    "chunk_id": item["chunk_id"],
                    "title": item["title"],
                    "url": item["url"],
                    "source": item["source"],
                }
            )
        #将最终的结果进行排序
        score_docs.sort(key=lambda item: item["score"], reverse=True)
        return score_docs
    except Exception as e:
        logger.error("使用reranker模型重排序失败，{}", e)
        for item in doc_items:
            item["score"] = 0.0
        return doc_items

@step_log("step_3_topk")
def step_3_topk(scored_docs):
    """
    阶段三：动态 TopK（最多 10）
    基于 scored_docs（已按 score 降序排序）进行智能截断，
    核心逻辑：结合固定上下限+断崖阈值判断，避免机械取前N条，保留语义相关的连续文档集合
    :param scored_docs: 列表，元素为带score的文档字典，已按score降序排列，格式如[{"doc": 文档对象, "score": 相关性分数}, ...]
    :return: 列表，动态截断后的TopK文档列表，数量≤10
    """
    # 硬上限：最多取前10条，取全局常量与实际文档数的较小值（避免索引越界）
    # 注：max_topk从全局常量读取，不依赖外部状态，保证逻辑一致性
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))
    min_topk = RERANK_MIN_TOPK  # 硬下限：至少保留的文档数量（全局常量配置）
    gap_ratio = RERANK_GAP_RATIO  # 相对断崖阈值：分数下降的相对比例阈值（全局常量配置）
    gap_abs = RERANK_GAP_ABS      # 绝对断崖阈值：分数下降的绝对差值阈值（全局常量配置）

    #创建表示真正动态topk的变量
    topk = max_topk
    #判断现有的数据是否能够满足硬下限
    #若能够满足，则获取动态的上限
    #若无法满足，则提供现有数据
    if topk > min_topk:
        #循环遍历相邻的数据的分数差以及分数差比例
        for i in range(min_topk -1,topk -1):
            #分别获取相邻的数据的分数
            score1 = scored_docs[i]["score"]
            score2 = scored_docs[i+1]["score"]
            #分别获取分数差值和分数差值的比例
            gap = score1 - score2
            rel = gap / (abs(score1) + 1e-6)
            #判断分数差值和分数差值的比例是否大于等于绝对断崖阈值和相对断崖阈值
            if gap >= gap_abs or rel >= gap_ratio:
                #获取动态topk的值
                topk = i + 1
                break
    scored_docs = scored_docs[:topk]
    return scored_docs


@node_log("node_rerank")
def node_rerank(state):
  """
  Rerank节点
  对检索到的文档进行重新排序，提高相关性
  """
  logger.info("---Rerank (重排序) 节点开始处理---")
  add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

  # 阶段一：合并文档
  doc_items = step_1_merge_docs(state)
  # 阶段二：对文档进行重排序
  scored_docs = step_2_rerank_docs(state, doc_items)
  # 阶段三：动态 TopK
  topk_docs = step_3_topk(scored_docs)

  logger.info(f"Rerank 节点处理结束, 最终输出 {len(topk_docs)} 条文档")

  add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
  return {"reranked_docs": topk_docs}


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_rerank 本地测试")
    print("="*50)
    
    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8},
        {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1} # 预期低分
    ]
    
    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"} # 预期低分
    ]
    
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？", # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])
        
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)
        
        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")
            
        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后
        
        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("="*50)

    except Exception as e:
        logger.exception("测试运行期间发生未捕获异常: {}", e)
