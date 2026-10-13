import sys
import json
import asyncio
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerSse, MCPServerStreamableHttp
from app.core.logger import logger

#调用mcp工具
async def mcp_call_streamable(query):
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": mcp_config.api_key},
            "timeout": 300,
            "sse_read_timeout": 300,
            "terminate_on_close": True,
        },
        max_retry_attempts=2,
    )
    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5},
        )
        return result
    finally:
        await search_mcp.cleanup()


def node_web_search_mcp(state):
    """
    LangGraph同步节点函数：处理MCP搜索逻辑，作为整个搜索流程的入口。
    
    该节点会调用 mcp_call 异步函数获取搜索结果，并将其解析为结构化数据存储到 state 中。
    
    :param state: LangGraph的全局状态对象，包含 session_id, rewritten_query 等信息
    :return: 字典，包含结构化的搜索结果 web_search_docs，供后续节点使用
    """
    logger.info("---node_web_search_mcp 开始处理---")
    
    # 1. 标记任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 获取rewritten_query
    rewritten_query = state["rewritten_query"]
    #存储最终结果的列表
    results = []
    #判断rewritten是否为空
    if rewritten_query:
        # 网络搜索属于「可降级旁路」：MCP 服务不可用（地址失效/鉴权失败/握手超时）时，
        # 仅记录警告并返回空结果，避免异常抛穿整个查询 graph 导致问答流程整体失败。
        try:
            #异步调用mcp服务
            result = asyncio.run(mcp_call_streamable(rewritten_query))
            #获取网络搜索的主要数据
            text = result.content[0].text
            #将json格式的字符串text转化为字典
            text_dict = json.loads(text)
            #遍历text_dict中的pages所对应的数据
            for page in text_dict.get("pages", []):
                results.append(
                    {
                        "title": page.get("title", "").strip(),
                        "url": page.get("url", "").strip(),
                        "snippet": page.get("snippet", "").strip(),
                    }
                )
        except Exception as e:
            # 降级处理：跳过网络搜索，其余三路检索（向量/HyDE/图谱）结果不受影响
            logger.warning("node_web_search_mcp 降级跳过（MCP 服务不可用）：{}", e)
    # 5. 标记任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"web_search_docs": results}


if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "="*50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("="*50)
    
    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("="*50)
        
    except Exception as e:
        logger.exception("测试运行期间发生未捕获异常: {}", e)
