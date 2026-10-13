import json
import os
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
@step_log("step_1_get_content")
def step_1_get_content(state:ImportGraphState) :
    #获取md_content
    md_content = state["md_content"]
    #判断md_content是否为空
    if not md_content :
        logger.error("md文档内容获取失败，无法完成切分")
        raise RuntimeError("md文档内容获取失败，无法完成切分")
    #统一将md_content中的\r\n和\r替换成\n
    md_content = md_content.replace("\r\n","\n").replace("\r","\n")
    #获取file_title
    file_title = state["file_title"]

# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200 # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20

@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content, file_title):
    #创建匹配标题的正则表达式
    pattern = re.compile(r"^#{1,6}\s+.\s")
    #将md_content通过/n进行分割(列表中一行一行数据)
    lines = md_content.split("\n")
    #准备列表存储初切的切片
    sections = []
    #准备存储当前的标题行
    current_title = ""
    #准备存储当前行内容
    current_lines = []
    #是否是代码块
    is_code_clock = False
    #遍历每一行数据
    for line in lines:
        #去掉每一个数据两边的空格
        line = line.strip()
        #判断当前行是否为代码块中
        if lines.startswith("```") or lines.startswith("~~~"):

            #更新is_code_block的值，代码块的格式：第一行和最后一行以```或~~~开头
            #当前行为代码块的开始行，将is_code_block = True
            # 当前行为代码块的结束行，将is_code_block=False
            # 因此直接将is_code_block直接取反
            is_code_clock = not is_code_clock
            #存储此行数据
            current_lines.append(line)
            continue
        #不是代码块，判断是不是标题
        if not is_code_clock and pattern.match(line):
            #判断current_title是否为空，若不为空，则记录上一个标题所对应的切片
            if current_title:
                sections.append(
                    {
                        "title": current_title,
                        "lines": "\n".join(current_lines)
                        "file_title": file_title,
                    }
                )
            #遇到新的标题，记录新的标题的数据
            current_title = line
            current_lines = [line]
        else:
            # 表示是代码块或者不是标题，直接保存当前行
            current_lines.append(line)
    #保存md_content中最后一个标题以及后续的行
    if current_title:
        sections.append(
            {
                "title": current_title,
                "content": "\n".join(current_lines),
                "file_title": file_title,
            }
        )
        # 若遍历了md_content的每一行，sections仍然为空，表示md_content中没有任何一个标题
        # 此时直接将md_content作为一个切片
    if not sections:
        sections.append(
            {
                "title": "无主题",
                "content": md_content,
                "file_title": file_title,
            }
        )
    return sections



@step_log("step_3_refine_chunks")
def step_3_refine_chunks(sections):
    # 创建递归切分对象
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "；", " "]
    )
    # 创建存储最终结果的变量
    final_chunks = []
    # 遍历初切的切片
    for chunk in sections:
        # 使用递归切分对chunk进行二次切分
        sub_chunks = spliter.split_text(chunk["md_content"])
        # 判断二次切分之后的子切片的数量是否大于1
        has_multiple_chunks = len(sub_chunks) > 1
        # 对二次切分的结果进行遍历
        for idx, sub_chunk in enumerate(sub_chunks, start=1):
            # 获取当前子切片的标题
            # 若有多个子切片，current_title=title_1,title_2,title_3...
            # 若只有一个子切片，current_title=title
            current_title = f"{chunk['title']}_{idx}" if has_multiple_chunks else chunk["title"]
            # 保存每个切片
            final_chunks.append(
                {
                    "title": current_title,
                    "content": sub_chunk,
                    "file_title": chunk["file_title"],
                    "parent_title": chunk["title"],
                    "part":idx
                }
            )
    return final_chunks

@step_log("step_4_backup_chunks")
def step_4_backup_chunks(final_chunks, state):
    # 获取存储切片的json文件的路径
    chunks_backup_path = Path(state["md_path"]).parent / "backup.json"
    # 将final_chunks写出到json文件中
    with open(chunks_backup_path, "w", encoding="utf-8") as f:
        json.dump(
            final_chunks,
            f,
            ensure_ascii=False,  # 中文直接原文存储
            indent=4  # 表示json结构的缩进
        )

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    #记录节点运行状态为运行中
    add_running_task(state["task_id"],"node_document_split")
    #步骤1：获取与清洗内容，进行state中数据清晰(md_content / file_title (做标题兜底)
    md_content , file_title = step_1_get_content
    #步骤2：按标题进行初切，保证语义完整
    sections, title_count, lines_count = step_2_split_by_title(md_content, file_title)
    # 步骤3：使用RecursiveCharacterTextSplitter进行二次切分，控制切片大小
    final_chunks = step_3_refine_chunks(sections)
    # 步骤4：更新状态并备份切片数据到json文件中
    state['chunks'] = final_chunks
    step_4_backup_chunks(final_chunks, state)

    # 记录节点运行状态为已完成
    add_done_task(state["task_id"], "node_document_split")
    return state