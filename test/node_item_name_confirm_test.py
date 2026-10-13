import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import AnnSearchRequest
from sqlalchemy.ext.asyncio import result
from sympy.polys.matrices import dense

from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger, step_log
from cleanup_milvus import collection_name

load_dotenv(find_dotenv())

@step_log("step_3_extract_info")
def step_3_extract_info(original_query, history_list):
    """
    利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
    若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
    :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
    :param history: 列表[字典] - 近期会话历史
    :return: 字典 - 提取结果，格式：{"item_names": [], "rewritten_query": ""}
    """
    #遍历history_list，将每个历史对话拼接，role(user/ai):text
    history_text = ""
    for history in history_list:
        history_text += f"{history['role']}:{history['text']}\n"
    #读取rewritten_query_and_itemnames.prompt获取提示词
    prompt = load_prompt("rewritten_query_and_itemnames",history_text=history_text,query=original_query)
    #组织用户提示词和系统提示词
    message = [
        SystemMessage("你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(prompt),
    ]
    try:
        #获取大模型对象
        llm = get_llm_client()
        #调用大模型，获取重写问题和产品主题
        response = llm.invoke(message)
        #获取大模型输出的内容
        result = response.content
        #判断result是否是json代码块，```json{key:value}```
        if result.startswith("```json"):
            result = result.replace("```json", "").replace("```","")
        #将字符串格式的json转换为python对象
        extract_result = json.loads(result)
        #判断结果中是否包含item_name
        if "item_names" not in extract_result:
            extract_result["item_names"] = []
        #判断结果中是否包含rewritten_query
        if "rewritten_query" not in extract_result:
            extract_result["rewritten_query"] = original_query
        return extract_result
    except Exception as e:
        logger.error(f"提取产品主体并且重写用户问题时出现了异常：{e}")
        return {"item_names": [], "rewritten_query": original_query}

@step_log("step_4_vectorize_and_query")
def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict]:
    #创建存储最终结果的列表
    results = []
    try:
        #获取milvus的客户端
        milvus_client = get_milvus_client()
        #判断milvus_client是否为空
        if not milvus_client:
            logger.error("获取Milvus客户端失败")
            return results
        #获取要检索的集合
        collection_name = milvus_config.item_name_collection
        #判断collection_name是否为空
        if not collection_name:
            logger.error("获取Milvus客户端失败")
            return results
        #获取item_names所对应的向量
        embeddings = generate_embeddings(item_names)
        #对item_names进行遍历，在向量数据库中进行检索
        for i,_ in enumerate(item_names):
            #分别获取每个item_name所对应的稠密向量和稀疏向量
            dense_vector = embeddings["dense"][i]
            sparse_vector = embeddings["sparse"][i]
            #设置稠密向量的检索方式设置稀疏向量的检索方式
            reqs = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector,limit=5)
            """hybrid_search_results
               data=[
                        [
                            {
                            'pk': 468868229533272140,
                            'distance': 0.9151462912559509,
                            'entity':{'item_name': 'HAK 180 烫金机'}
                            }
                        ]
                    ]
            """
            #进行混合检索
            hybrid_search_result = hybrid_search(
                client=milvus_client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.3,0.7),
                norm_score=True,
                limit=5,
                output_fields="item_names",
            )
            #创建存储检索结果的列表
            matches = []
            #判断检索结果是否为空
            if hybrid_search_result and len(hybrid_search_result) > 0:
                #对检索的结果进行遍历
                for result in hybrid_search_result[0]:
                    matches.append(
                        {
                            "item_names":result["entity"]["item_names"],
                            "score":result["distance"],
                        }
                    )
            #存储最终的结果
            results.append(
                {
                    "extracted_name":item_names[i],
                    "matches":matches,
                }
            )
        return results
    except Exception as e:
        logger.error(f"混合检索item_name失败，{e}")



@step_log("step_5_align_item_names")
def step_5_align_item_names(query_results: List[Dict]) -> Dict:
    # 创建存储已确认的item_name的列表
    confirmer_item_names: List[str] = []
    # 创建存储待确认的item_name的列表
    options: List[str] = []
    #遍历query_results
    for result in query_results:
        #获取从用户问题中提取的extracted_name
        extracted_name = result["extracted_name"]
        #获取extracted_name所检索的数据
        matches = result["matches"]
        #将检索到的数据根据分数进行倒叙排序
        matches.sort(key=lambda x: x["score"],reverse=True)
        #判断matches是否为空
        if not matches:
            logger.warning(f"{extracted_name}没有检索到任何数据")
            continue
        #分别获取高分数（>=0.85）和中间分数（>=0.6 and <0.85）
        high = [match for match in matches if match["score"] >= 0.85]
        middle = [match for match in matches if match["score"] >= 0.6]
        #判断high中数据的数量，若只有一条，则直接确认item_name
        #创建存储已确认的产品主体item_name变量
        picked = None
        if len(high) == 1:
            confirmer_item_names.append(high[0]["item_name"])
            continue
        # 判断high中数据的数量，若有多条，优先找检索的item name和extracted name一致的数据
        if len(high) > 1:
            for item in high:
                if item["item_name"] == extracted_name:
                    picked = item
                    continue
            #判断picked是否为None，若为None，表示0.85以上没有数据的item name和extracted name一致
            # 直接将0.85以上的数据中分数最高的作为已确认的item name
            if not picked:
                picked = high[0]
            #保存已确认的item_name
            confirmer_item_names.append(picked["item_name"])
            continue
        #表示没有已确认的item_name,即所检索的数据的score在0.6-0.85之间
        #将0.6-0.85的数据中的前三个作为待确认的item_name
        if len(middle) > 1:
            for item in middle[:3]:
                #保存待确认的item_name
                options.append(item["item_name"])
    return {
        "confirmer_item_names": list(set(confirmer_item_names)),
        "options": list(set(options)),
    }

@step_log("step_6_check_confirmation")
def step_6_check_confirmation(state: Dict, align_result: Dict, session_id: str, history_list: List[Dict], rewritten_query: str) -> Dict:
    """
    检查对齐结果，更新 State
    """
    #分别获取已确认和待确认的item_name的列表
    confirmed = align_result["confirmed"]
    options = align_result["options"]
    #分支1：有已确认的item_name
    if confirmed:
        #更新历史记录中item_names
        #先获取要修改的数据的_id
        ids = []
        #遍历历史记录
        for history in history_list:
            #若历史记录的item_name为空，进行更新，就需要记录历史记录的_id
            if not history["item_names"]:
                ids.append(history["_id"])
        #判断ids是否为空，若不为空则修改历史记录的item_name
        if ids:
            update_message_item_names(ids,confirmed)
        #更新状态
        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        # 判断state中是否有answer，若有则删除
        if state.get("answer"):
            del state["answer"]
        return state
    # 分支2：有待确认的item_name
    if options:
        #获取并拼接待确认的item_name
        options_str = "、".join(options)
        #拼接待确认信息
        answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
        #更新状态
        state["answer"] = answer
        state["item_names"] = []
        return state
    #分支3：既没有已确认也没有待确认
    # 拼接待确认信息
    answer = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
    # 更新状态
    state["answer"] = answer
    state["item_names"] = []
    return state

@step_log("step_7_write_history")
def step_7_write_history(state: Dict, session_id: str, history_list: List[Dict], rewritten_query: str, message_id: str) -> Dict:
    #判断状态中是否有answer,若有则保存历史记录，若没有，更新历史记录
    if state.get("answer"):
        save_chat_message(session_id, "assistant", state["answer"],item_names=state.get("item_names", []),rewritten_query=rewritten_query)
    #更新历史记录
    save_chat_message(
        session_id=session_id,  # 会话ID，关联所属会话
        role="user",  # 消息角色：用户
        text=state["original_query"],  # 消息内容：用户原始查询
        rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
        item_names=state.get("item_names", []),  # 补充关联的商品名列表
        message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
    )
    return state

def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    主节点函数：商品名称确认流程
    """
    logger.info(">>> node_item_name_confirm: 开始处理")
    
    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    # 标记任务开始
    add_running_task(session_id, "node_item_name_confirm", is_stream)

    # 1. 获取历史记录
    history_list = get_recent_messages(session_id, limit=10)
    # 2. 将当前用户的问题保存到MongoDB中,返回的message_id是添加的数据的唯一标识
    message_id = save_chat_message(state["session_id"], "user",state["original_query"],"",[])
    # 3. 从用户的问题中提取item_name并重写用户问题
    extract_result = step_3_extract_info(original_query, history_list)
    # 分别获取提取的item_name和重写之后的问题rewritten_query
    item_names = extract_result.get("item_names")
    rewritten_query = extract_result.get("rewritten_query")
    #更新状态中的rewritten_query
    state["rewritten_query"] = rewritten_query
    #创建存储对齐之后结果的字典
    align_result = {}
    #  如果有提取到商品名，进行搜索和对齐
    if len(item_names) > 0:
        #步骤4：通过item_names在向量数据库中进行检索
        query_results = step_4_vectorize_and_query(item_names)
        #步骤5：获取对齐的结果
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: 未提取到商品名，跳过向量检索")
    #步骤6：检查确认状态
    state = step_6_check_confirmation(state, align_result, session_id, history_list, rewritten_query)
    #步骤7：写入历史记录
    final_state = step_7_write_history(state, session_id, history_list, rewritten_query, message_id)
    #保存历史记录到状态中
    final_state["history"] = history_list
    # 标记任务完成
    add_done_task(session_id, "node_item_name_confirm", is_stream)
    
    return state


