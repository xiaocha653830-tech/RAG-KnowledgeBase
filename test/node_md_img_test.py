import base64
import re
from pathlib import Path

import requests
from aiohttp._websocket.reader_c import deque
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from openai.types.beta.threads import image_file
from win32gui import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_running_task, add_done_task

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(file_name: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return Path(file_name).suffix.lower() in IMAGE_EXTENSIONS

@step_log("step_1_get_content")
def step_1_get_content(state:ImportGraphState):
    #获取md_path并判断是否为空
    md_path = state["md_path"]
    if not md_path:
        raise ValueError("md_path为空，请检查流程")
    #获取md_path所对应的path对象
    md_path_obj = Path(md_path)
    #判断md_path_obj是否存在
    if not md_path_obj.exists():
        raise FileNotFoundError(f"{md_path}不存在")
    #判断md_content是否为空
    #若为空，说明上传的文件是md，节点执行node_entry -->node_md_img
    #若不为空，说明上传的文件是pdf
    if not state["md_content"]:
        state["md_content"] = md_path_obj.read_text(encoding="utf-8")
    #获取存储md文件所对应的图片的目录路径
    images_dir_obj = md_path_obj.parent / "images"
    return state["md_content"],md_path_obj,images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content:str, images_dir_obj:Path):
    #创建存储最终结果的列表：list[tuple(图片名，图片路径，tuple（上文，下文）)]
    images_targets = []
    #遍历images_dir_obj中的所有图片文件
    for image_file in images_dir_obj.iterdir():
        #获取每个图片文件的文件名
        image_name = image_file.name
        #判断文件名是否为支持
        if not is_supported_image(image_name):
            logger.warning(f"{image_name}不是支持的图片类型")
            continue
        #创建匹配md_content中图片的正则表达式
        # re.escape（）将字符串中的所有特殊字符进行转译
        pattern = re.compile("!/[.*?/]/(.*?"+ re.escape(image_name) +"/)")

        #在md_content中匹配正则
        match_list = list(pattern.finditer(md_content))
        #判断match_result是否为空
        if not match_list:
            logger.warning(f"图片{image_name}未在md文件中引用")
            continue
        #获取image_name在md_content中第一次出现的开始索引和结束索引
        start,end = match_list[0].span()
        #分别获取开始索引前100个字符和结束索引后的100个字符
        pre_text = md_content[max(0,start - 100):start]
        post_text = md_content[end:min(len(md_content),end + 100)]
        #将上文和下文组合成上下文
        context = (pre_text,post_text)
        #将遍历的图片的信息追加到image_target中
        images_targets.append((image_name,str(image_file),context))
        return images_targets


def step_3_image_summary(image_targets, stem) :
    #创建存储最终结果的字典
    summaries = {}
    # 创建一个双端队列
    requests_limiter = deque()
    #遍历image_targets，获取md文件中每个图片的摘要信息
    for image_name,image_path,_ in image_targets:
        #实现限流
        apply_api_rate_limit(requests_limiter, max_requests=100)
        # 获取提示词
        prompt = load_prompt("image_summary", root_folder=stem, image_content=image_file)
        # 获取视觉模型
        vl_model = get_llm_client(lm_config.lv_model)
        # 判断图片路径image_path是否是字符串
        if isinstance(image_path, Path):
            image_path = Path(image_path)
        # 将图片文件中内容转换为base64格式的字符串
        image_base64 = base64.b64encode(image_path.read_bytes()).decode(encoding="utf-8")
        # 准备用户提示词，以下是访问视觉模型解析图片的提示词的固定结构
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )
        #创建连对象
        chain = vl_model | StrOutputParser()
        #启动模型
        summary = chain.invoke([message])
        #匹配
        summaries[image_name] = summary
    return summaries


def step_4_upload_images_replace(image_summaries, image_targets, md_content, stem):
    # 获取minio的客户端对象
    minio_client = get_minio_client()
    # 将之前md文件中的图片查询
    object_list = minio_client.list_objects(
        bucket_name=minio_config.minio_bucket_name,
        prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
        recursive=True,
    )
    # 将object_list中的图片转换为DeleteObject对象
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    # 将object_list中的图片删除
    delete_errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    # 遍历删除的结果
    for error in delete_errors:
        logger.warning(f"{error}")
    # 创建存储图片上传到minio之后的url的字典
    image_urls = {}
    # 遍历image_targets，将图片上传到minio
    for image_name,image_path,_ in image_targets:
        try:
            # 上传图片
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_name}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            # 获取图片在minio中的地址
            image_urls[image_name] =f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}"
        except Exception as e:
            logger.error(f"{image_name}上传失败,{e}")
    # 创建存储图片所对应的摘要信息和minio中url的变量
    image_infos = {}
    # 遍历image_summaries
    for image_name,summary in image_summaries.items():
        image_infos[image_name] = (summary,image_urls[image_name])
    # 遍历image_infos，将md_content中的![]()-->![summary](url)
    for image_name,(summary,url) in image_infos.items():
        # 创建正则表达式，匹配md_content中的图片
        pattern = re.compile(r"!\[.*?\]\(.*?"+re.escape(image_name) + r"\)")
        # 替换md_content中的![]()-->![summary](url)
        md_content = pattern.sub(lambda _: f"![{summary}]({url})", md_content)
    return md_content

@step_log("step_5_backup_md_file")
def step_5_backup_md_file(md_path_obj: Path, new_md_content: str):
    # 创建新的md文件的路径
    new_md_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_new{md_path_obj.suffix}"
    # 向new_md_path_obj所对应的文件中写入new_md_content
    new_md_path_obj.write_text(new_md_content, encoding="utf-8")
    return str(new_md_path_obj)

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    #记录任务状态为运行中
    add_running_task(state["task_id"],"node_md_img")
    #步骤1：进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
    md_content,md_path_obj,images_dir_obj = step_1_get_content(state)
    # 步骤2：提取md文件中的图片
    image_targets = step_2_scan_images(md_content, images_dir_obj)
    #步骤3：进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结
    image_summaries = step_3_image_summary(image_targets, md_path_obj.stem)
    # 步骤4：上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
    new_md_content = step_4_upload_images_replace(image_summaries, image_targets, md_content, md_path_obj.stem)
    # 步骤5：备份新的md内容,改为原名称_new.md
    new_md_file_path_str = step_5_backup_md_file(md_path_obj, new_md_content)
    #记录任务状态为已完成
    add_done_task(state["task_id"],"node_md_img")

    return state