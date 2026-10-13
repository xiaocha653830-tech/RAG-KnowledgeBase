# 加载环境变量：从 .env 文件读取配置（如Milvus地址、KG服务地址、BGE模型路径等）
from dotenv import load_dotenv
# 导入LangGraph核心依赖：StateGraph(状态图)、START/END(内置起始/结束节点常量)
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# 导入自定义状态类：统一管理工作流全程的所有数据（各节点共享/修改）
from app.import_process.agent.state import ImportGraphState, create_default_state
# 导入所有自定义业务节点：每个节点对应知识库导入的一个具体步骤
from app.import_process.agent.nodes.node_entry import node_entry  # 入口节点：初始化参数、校验输入
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF转MD：解析PDF文件为markdown格式
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD图片处理：提取/下载markdown中的图片、修复图片路径
from app.import_process.agent.nodes.node_document_split import node_document_split  # 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # 项目名识别：从分块中提取核心项目名称（业务定制化）
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE向量化：将文本分块转换为向量表示（适配Milvus向量库）
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # 导入Milvus：将向量数据写入Milvus向量数据库
# 创建工作流对象
workflow = StateGraph(ImportGraphState)

#添加节点
workflow.add_node("node_entry",node_entry)
workflow.add_node("node_pdf_to_md",node_pdf_to_md)
workflow.add_node("node_md_img",node_md_img)
workflow.add_node("node_document_split",node_document_split)
workflow.add_node("node_item_name_recognition",node_item_name_recognition)
workflow.add_node("node_bge_embedding",node_bge_embedding)
workflow.add_node("node_import_milvus",node_import_milvus)

# 创建条件边的路径函数
# 判断state中的is_md_read_enabled和is_pdf_read_enabled
# is_md_read_enabled = True -- >node_md_img
# is_pdf_read_enabled = True -- >node_pdf_to_md
def condition_fun(state: ImportGraphState):

    if state["is_md_read_enabled"]:
        # is_pdf_read_enabled 是否为True
        # 若为True,说明当前上传的是md文件--》node_md_img
        return "node_md_img"
    if state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
        #is_pdf_read_enabled 是否为True
        #若为True,说明当前上传的是pdf文件--》node_pdf_to_md
    else:
        #当接收到的文件既不是md也不是pdf时，直接打回
        return END

# 添加边
# 设置初始节点
workflow.set_entry_point("node_entry")
workflow.add_conditional_edges(
    "node_entry",
    condition_fun,
    {
        "node_md_img":"node_md_img",
        "node_pdf_to_md":"node_pdf_to_md",
        END:END
     }
)

workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF转MD完成 → MD图片处理
workflow.add_edge("node_md_img", "node_document_split")  # MD处理完成 → 文档分块
workflow.add_edge("node_document_split", "node_item_name_recognition")  # 分块完成 → 项目名识别
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # 项目名识别完成 → BGE向量化
workflow.add_edge("node_bge_embedding", "node_import_milvus")  # 向量化完成 → 导入Milvus向量库
workflow.add_edge("node_import_milvus", END)  # Milvus入库完成 → 工作流执行结束（END是内置结束节点）

#创建图对象
kb_import_app = workflow.compile()

