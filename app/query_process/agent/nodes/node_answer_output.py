import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger, step_log, node_log
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000

@step_log("step_1_check_answer")
def step_1_check_answer(state) -> bool:
    """
    阶段一：检查 state 中是否已有 answer。
    - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
    - 若不存在：返回 False
    """
    #分别获取状态中的answer和is_stream
    answer = state.get("answer")
    is_stream = state.get("is_stream")
    #判断answer是否为空
    if answer:
      if is_stream:
          push_to_session(state["session_id"],SSEEvent.DELTA, {"delta": answer})
      else:
          set_task_result(state["session_id"], "answer", answer)
      return True
    else:
      return False


# 目标结构
# HAK 180 烫金机的操作面板位于机器正前方。开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
# 具体的按键位置请参考下图：
# 【图片】
# http://local-server/images/panel_view.jpg
# http://local-server/images/button_detail.jpg
@step_log("step_2_generate_response")
def step_2_construct_prompt(state: QueryGraphState) -> str:
    """
    第一阶段：构建 Prompt
    根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt
    """
    #从状态中获取所需数据
    rewritten_query = state.get("rewritten_query")
    original_query = state.get("original_query")
    question = rewritten_query if rewritten_query else original_query
    reranked_docs = state.get("reranked_docs")
    item_names = state.get("item_names")
    history_list = state.get("history")
    #处理上下文
    """
    构造"元数据头 + 正文"格式的字符串，例如：
        "[1] [local] [chunk_id=123] [score=0.95] [title=操作手册]
        这里是文档的正文内容..."
    """
    #处理上下文 创建存储处理之后的结果的列表
    docs = []
    #创建记录最大字符数
    used = 0
    #对reranked_docs进行遍历
    for num,chunk in enumerate(reranked_docs,start=1):
        #从chunk中获取所需要的数据
        text = chunk.get("text")
        if not text:
            continue
        data_list = [f"[{num}]"]
        source = chunk.get("source")
        if source:
            data_list.append(f"[{source}]")
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            data_list.append(f"[chunk_id={chunk_id}]")
        score = chunk.get("score")
        if score is not None:
            data_list.append(f"[score={float(score):.4f}]")
        title = chunk.get("title")
        if title:
            data_list.append(f"[title={title}]")
        #拼接各个数据为指定格式
        doc = " ".join(data_list) + "\n" + text
        #判断当前字符数是否超过最大字符数的阈值
        if used + len(doc) > MAX_CONTEXT_CHARS:
            break
        #存储每条数据转换的结果
        docs.append(doc)
        #记录本次循环的字节数
        used += len(doc) + 2
    #将每条数据转换的结果拼接为字符串
    context = "\n\n".join(docs) if docs else "无参考内容"
    """
    将历史对话转换为以下格式
    用户：xxx
    助手：xxx
    """
    #处理历史记录
    #判断历史记录是否为空
    history_str = ""
    if history_list:
        #对history_list进行遍历
        for history in history_list:
            #分别获取历史记录中role和text
            role = history.get("role")
            text = history.get("text")
            #判断role是否为user
            history_text = ""
            if role == "user" and text :
                history_str += f"用户：{text}\n"
            elif role == "assistant" and text :
                history_str += f"助手：{text}\n"
            if used + len(history_str) > MAX_CONTEXT_CHARS:
                break
            history_str += history_text
            #记录拼接之后的字符数
            used += len(history_text)
    else:
        history_str = "无历史记录1"
    """
    将产品主体转化为以下格式
    产品主体1，产品主体2，....
    """
    item_names_str = ", ".join(item_names or [])
    #读取提示词（answer_out.prompt：占位符为 context/history/item_names/question）
    prompt = load_prompt(
        "answer_out",
        context=context,
        history=history_str,
        item_names=item_names_str,
        question=question
    )
    #返回组装好的 prompt，供 step_3_generate_response 调用模型
    return prompt

@step_log("step_3_generate_response")
def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
    """
    第二阶段：生成回答
    调用llm生成答案，支持流式输出
    """
    #从状态中获取session_id和is_stream
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")
    #获取模型对象
    llm = get_llm_client()
    #调用模型 判断是否为流式
    if is_stream:
        #创建存储最终answer的变量
        answer = ""
        try:
            for chunk in llm.stream(prompt):
                #当使用流式调用大模型时，返回封装了token的对象
                #每次返回的具体的token
                content = chunk.content
                #将每次返回的token推送到队列中
                push_to_session(session_id, SSEEvent.DELTA,{"delta":content})
                #将每次的token拼接，获取最终的answer
                answer += content
        except Exception as e:
            push_to_session(session_id, SSEEvent.ERROR, {"error": e})
        #更新状态
        state["answer"] = answer
    else:
        try:
            #非流式，直接调用大模型
            response = llm.invoke(prompt)
            #获取最终的结果
            answer = response.content
            #设置当前任务的状态
            set_task_result(session_id,"answer",answer)
            # 更新状态
            state["answer"] = answer
        except Exception as e:
            state["answer"] = "抱歉，生成回答时出现错误"
    return state

#从最终的文档中提取图片的url
def _extract_images_from_docs(reranked_docs):
    """
    辅助方法：从文档列表中提取图片URL
    
    核心逻辑：
    1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）。
    2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果）。
       - 验证后缀名是否为图片格式 (.jpg, .png 等)。
    3. 策略二：使用正则表达式扫描文档 'text' 正文内容（常见于本地 Markdown 文档）。
       - 匹配 Markdown 图片语法: ![alt text](image_url)。
    4. 对提取到的 URL 进行去重处理，返回唯一图片列表。
    
    :param docs: 文档列表，每个文档为字典格式
    :return: 图片 URL 字符串列表
    """
    #创建存储提取的图片的url列表
    image_urls = []
    #创建正则表达式
    pattern = r"!\[.*?\]\((.*?)\)"
    #对文档进行遍历
    for doc in reranked_docs:
        #获取文档中的url
        url = doc.get("url")
        #判断url是否为空
        if url:
            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                if url not in image_urls:
                    image_urls.append(url)
        #获取文档中text
        text = doc.get("text")
        if text:
            #通过正则表达式提取文档中的图片url
            matches = re.findall(pattern, text)
            #判断matches是否为空
            if matches:
                for match in matches:
                    if match not in image_urls:
                        image_urls.append(match)
    return image_urls

@step_log("step_4_write_history")
def step_4_write_history(state: QueryGraphState, image_urls) -> QueryGraphState:
    """
    阶段四：把本轮答案写入 MongoDB history。
    利用 utils/mongo_history_utils.py 中的 save_chat_messages 方法。
    """
    session_id = state.get("session_id")
    item_names = state.get("item_names")
    answer = state.get("answer")
    #判断answer是否有值
    if answer:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            rewritten_query="",
            item_names=item_names,
            image_urls=image_urls,
            message_id=None
        )


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
    2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成答案
    3 阶段三：调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
    4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
    5 做最后一次push操作（主要是为了触发前端图片渲染)
     {
        "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
        "status": "completed",
        "image_urls": [
            "http://local-server/images/panel_view.jpg",
            "http://local-server/images/button_detail.jpg"
        ]
      }
    """
    logger.info("---node_answer_output (答案生成) 节点开始处理---")
    add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
    answer_exists = step_1_check_answer(state)

    # 阶段二  如果没有answer则 构建 Prompt
    if not answer_exists:
        prompt = step_2_construct_prompt(state)
        state["prompt"] = prompt

        # 阶段三：  如果没有answer则 调用大模型输出答案
        step_3_generate_response(state, prompt)

    # 提取图片URL（用于历史记录和前端展示）
    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])

    # 阶段四：把答案写入到mongodb的history中
    if state.get("answer"):
        logger.info("---写入MongoDB历史记录---")
        step_4_write_history(state, image_urls=image_urls)

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 阶段五: 流式输出结束，发送 final 事件 [最后兜底，确保图片都能争取渲染和结束]
    logger.info(f"---发送 final 事件---图片为：{image_urls}")
    if state.get("is_stream"):
        push_to_session(
            state['session_id'],
            SSEEvent.FINAL,
            {
                "answer": state["answer"],
                "status": "completed",
                "image_urls": image_urls  # 发送图片URL给前端
            }
        )

    logger.info("---node_answer_output 节点处理结束---")
    return state


if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_answer_output 本地测试")
    print("="*50)
    
    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)
            
            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg", # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False, # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)
        
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        
        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)
        
        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("="*50)

    except Exception as e:
        logger.exception("测试运行期间发生未捕获异常: {}", e)
