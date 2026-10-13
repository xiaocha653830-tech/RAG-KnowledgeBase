from pathlib import Path

from app.core.logger import node_log,logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task


@node_log("node_entry")
def node_entry(state:ImportGraphState) -> ImportGraphState:
    """
        节点: 入口节点 (node_entry)
        为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
        未来要实现:
            1. 进行任务状态记录,开始和结束列表记录
            2. 根据state中 local_file_path属性判断数据类型进而修改
               相关参数, is_md_read_enabled 或者 is_pdf_read_enabled
                        md_path 或者 pdf_path
            3. 不可解析结果类型不可用,直接输出对应警告日志! 逻辑路由节点会自动处理
            4. 获取file_tile标识,用于后期识别pdf对应的主体(item_name)进行兜底
    """
    # 记录节点的状态
    add_running_task(state["task_id"],"node_entry")
    #获取上传文件的路径
    local_file_path = state.get("local_file_path")
    # 判断local_file_path是否为空
    if not local_file_path:
        #说明state中没有上传文件，健壮性处理
        logger.warning(f"当前状态中没有上传文件，请检查初始状态")
        add_done_task(state["task_id"], "node_entry")
        return state
    if local_file_path.endswith(".pdf"):
        # 更新is_pdf_read_enabled、pdf_path
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    elif local_file_path.endswith(".md"):
        # 更新is_md_read_enabled、md_path
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    else:
        #说明文件不是md、pdf
        logger.warning(f"虽然输出了local_file_path,但是无法识别文件类型,请检查输入文件类型是否正确,"
                       f"目前只支持md和pdf文件,请检查! {local_file_path}")
        add_done_task(state['task_id'], 'node_entry')
        return state
    # 获取文件标题，更新file_title
    file_tile = Path(local_file_path).stem
    state["file_title"] = file_tile
    # 记录节点的状态
    add_done_task(state["task_id"],"node_entry")
    return state



