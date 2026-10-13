# -*- coding: utf-8 -*-
"""基于 ragas 的 RAG 问答评估测试程序（适配本项目）

程序流程：
    1. 从 eval/qa.csv 读取黄金问题集（question / ground_truth 两列）；
    2. 通过 RAG 查询流程（app.query_process.agent.main_graph.query_app）执行每个问题，
       从返回 state 的 reranked_docs 提取检索上下文、answer 提取生成答案；
    3. 使用 ragas 计算 5 项指标：Faithfulness、Answer Relevancy、Context Precision、
       Context Recall、Answer Correctness；
    4. 将评估结果保存到 eval/qa_result.csv（UTF-8 BOM 编码，共 9 列）。

本项目适配说明（与常见示例代码的差异）：
    - RAG 入口：本项目 main_graph.py 直接导出**已编译实例** query_app，不存在 create_query_graph()；
    - LLM 客户端：本项目无 AIClients 类，使用 app/lm/lm_utils.py 的 get_llm_client()；
    - 嵌入客户端：使用 app/lm/embedding_utils.py 的 get_bge_m3_ef()（pymilvus 的 BGEM3EmbeddingFunction）；
    - 检索上下文：reranked_docs 每项为 {"text", "score", "source"}，取 text 字段（非 content）。

运行前提：
    - 使用项目虚拟环境 .venv 运行本脚本（uv 管理，无 pip：uv pip install ragas）；
    - 已正确配置 .env 中的 OPENAI_API_KEY、OPENAI_BASE_URL、LLM_DEFAULT_MODEL、BGE_M3_PATH 等；
    - Milvus 已启动且目标集合 kb_chunks 中已有数据，否则检索为空、指标无实际参考意义。

依赖版本说明：
    - ragas 0.4.x 采用 instructor 架构，指标 LLM 需通过 ragas.llms.llm_factory 创建；
    - 嵌入使用 ragas.embeddings.base.BaseRagasEmbedding（需实现 embed_text / aembed_text）。
"""

import csv
import inspect
import os
import sys
import uuid
from typing import Any, Dict, List, Tuple

# ============================================================================
# 一、路径与依赖准备（必须在 import app 之前完成环境变量加载）
# ============================================================================

# 1. 计算项目根目录（eval 目录的上一级）
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)

# 2. 将项目根目录加入 sys.path，使 app 包可被导入
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 3. 显式加载项目根目录下的 .env，确保 app 内部读取到正确的环境变量
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass  # 若未安装 python-dotenv，则依赖外部已注入的环境变量


# ============================================================================
# 二、RAG 流程与客户端导入
# ============================================================================

# RAG 查询流程入口：main_graph.py 中已编译好的图实例 query_app（非工厂函数）
from app.query_process.agent.main_graph import query_app
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import get_bge_m3_ef

# ragas 现代 embedding 基类：若 ragas 未安装，先用 object 占位，避免模块级 import 报错，
# 真正的校验放在 step3 中进行
try:
    from ragas.embeddings.base import BaseRagasEmbedding
except ImportError:  # pragma: no cover - ragas 未安装时的兜底
    BaseRagasEmbedding = object


class BGEM3EmbeddingWrapper(BaseRagasEmbedding):
    """将 BGE-M3 嵌入客户端包装为 ragas 现代 embedding 接口。

    pymilvus 的 BGEM3EmbeddingFunction 提供 encode_documents(texts) 方法，返回 dict，
    其中 "dense" 字段为稠密向量（numpy 二维数组）。ragas 0.4.x 的 BaseRagasEmbedding
    则需要 embed_text / aembed_text 两个抽象方法，本类将前者适配为后者，
    供 AnswerRelevancy、AnswerCorrectness 等需要语义相似度的指标调用。

    说明：BGE-M3 对查询与文档使用同一套编码器（无 query 指令前缀差异），
    因此统一走 encode_documents，保证 embed_text 与 embed_texts 产出的向量口径一致。
    """

    def __init__(self, bge_m3_client):
        """初始化包装器。

        Args:
            bge_m3_client: get_bge_m3_ef() 返回的 BGEM3EmbeddingFunction 实例。
        """
        super().__init__()
        self._client = bge_m3_client

    @staticmethod
    def _dense_to_list(dense) -> List[List[float]]:
        """将 dense 向量统一转为 list-of-list 结构。

        Args:
            dense: encode_documents 返回的 dense 字段（numpy 二维数组或嵌套 list）。

        Returns:
            二维浮点数列表。
        """
        result = []
        for row in dense:
            # 兼容 numpy.ndarray（有 tolist）与普通 list 两种元素类型
            if hasattr(row, "tolist"):
                result.append(row.tolist())
            else:
                result.append(list(row))
        return result

    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        """单条文本向量化（同步）。

        Args:
            text: 单条文本。

        Returns:
            该文本的稠密向量。
        """
        result = self._client.encode_documents([text])
        return self._dense_to_list(result["dense"])[0]

    async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
        """单条文本向量化（异步）。BGE-M3 为本地模型无异步 API，复用同步实现。

        Args:
            text: 单条文本。

        Returns:
            该文本的稠密向量。
        """
        return self.embed_text(text, **kwargs)

    def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """批量文本向量化（同步，覆盖基类默认的逐条实现以提升性能）。

        Args:
            texts: 文本列表。

        Returns:
            每个文本对应的稠密向量列表。
        """
        result = self._client.encode_documents(list(texts))
        return self._dense_to_list(result["dense"])

    async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """批量文本向量化（异步），复用同步批量实现。

        Args:
            texts: 文本列表。

        Returns:
            每个文本对应的稠密向量列表。
        """
        return self.embed_texts(texts, **kwargs)


# ============================================================================
# 三、核心步骤函数（step1 ~ step6 按执行顺序编号）
# ============================================================================

def step1_load_qa_data() -> List[Dict[str, str]]:
    """步骤1：读取黄金问题集 qa.csv。

    读取 eval/qa.csv，列头兼容中英文（question/问题、ground_truth/参考答案）。

    Returns:
        问题-参考答案列表，每个元素形如 {"question": str, "ground_truth": str}。
    """
    qa_path = os.path.join(EVAL_DIR, "qa.csv")

    # 中英文列头 -> 标准字段名的映射，兼容不同的列头命名
    header_alias = {
        "question": "question",
        "问题": "question",
        "ground_truth": "ground_truth",
        "参考答案": "ground_truth",
    }

    qa_pairs: List[Dict[str, str]] = []
    with open(qa_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            # 将原始列头映射为标准字段名
            row = {header_alias.get(k, k): v for k, v in raw_row.items()}
            question = (row.get("question") or "").strip()
            ground_truth = (row.get("ground_truth") or "").strip()
            if question:
                qa_pairs.append({"question": question, "ground_truth": ground_truth})

    print(f"[step1] loaded {len(qa_pairs)} QA pairs from {qa_path}")
    return qa_pairs


def step2_init_rag_app():
    """步骤2：初始化 RAG 查询流程入口。

    本项目 main_graph.py 在模块导入时即完成编译并导出 query_app 实例，
    此处直接返回该实例，后续通过 query_app.invoke(state) 执行检索与生成。

    Returns:
        编译后的查询图（CompiledStateGraph）。
    """
    print("[step2] RAG query graph ready (query_app)")
    return query_app


def step3_init_eval_clients() -> Tuple[Any, Any, List[Any]]:
    """步骤3：初始化 ragas 评估所需的 LLM、Embedding 客户端与指标。

    - LLM：调用 get_llm_client() 获取 ChatOpenAI 客户端，提取其底层异步 OpenAI 客户端
      （root_async_client）与模型名（model_name），再用 ragas.llms.llm_factory 创建
      指标所需的 InstructorLLM（结构化输出）；
    - Embedding：调用 get_bge_m3_ef() 获取 BGE-M3 客户端，
      并用 BGEM3EmbeddingWrapper 包装为 ragas 的 BaseRagasEmbedding 接口；
    - 指标：从 ragas.metrics.collections 构建 5 项评估指标。

    Returns:
        (llm, embeddings, metrics) 三元组。
    """
    # 延迟导入 ragas 核心模块，若未安装则给出清晰的安装提示
    try:
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            AnswerCorrectness,
        )
    except ImportError as e:
        raise SystemExit(
            f"[step3] ragas not installed. Run:\n"
            f"  uv pip install ragas --python .venv/Scripts/python.exe\n"
            f"original error: {e}"
        )

    # 1. 获取 LLM 客户端（文本模式 ChatOpenAI），并提取底层异步客户端与模型名
    chat_llm = get_llm_client()
    model_name = chat_llm.model_name
    # 指标内部通过 agenerate() 异步调用，需使用 AsyncOpenAI 客户端（root_async_client），
    # 否则 InstructorLLM 会因同步 client 拒绝 agenerate()
    openai_client = chat_llm.root_async_client
    # 用 llm_factory 创建 ragas 指标所需的 InstructorLLM（结构化输出）
    # 注意：instructor 默认 max_tokens 偏小，而 faithfulness 需把答案拆解为 statements 列表
    # （含 reason/verdict），输出量约为答案的 2~3 倍，答案稍长即会触发截断报错，故显式调大。
    llm = llm_factory(model_name, client=openai_client, max_tokens=4096)

    # 2. 获取 BGE-M3 嵌入客户端并包装为 ragas 现代 embedding 接口
    bge_m3_client = get_bge_m3_ef()
    embeddings = BGEM3EmbeddingWrapper(bge_m3_client)

    # 3. 构建 5 项评估指标（llm 全部注入；embeddings 注入到需要语义相似度的指标）
    metrics = [
        Faithfulness(llm=llm),                              # 忠实度
        AnswerRelevancy(llm=llm, embeddings=embeddings),    # 答案相关性
        ContextPrecisionWithReference(llm=llm),             # 上下文精确度（带参考答案）
        ContextRecall(llm=llm),                             # 上下文召回率
        AnswerCorrectness(llm=llm, embeddings=embeddings),  # 答案正确性
    ]

    print(f"[step3] ragas ready | llm={model_name} | metrics={[m.name for m in metrics]}")
    return llm, embeddings, metrics


def step4_run_rag_for_question(question: str) -> Tuple[str, List[str]]:
    """步骤4：对单个问题执行 RAG 查询流程，获取答案与检索上下文。

    Args:
        question: 用户问题。

    Returns:
        (answer, contexts) 二元组：
            answer: 生成的答案字符串；
            contexts: 从 state["reranked_docs"] 中提取的上下文内容列表。
    """
    # 1. 构造查询流程的最小输入 state（与 query_service.py 的入参保持一致）
    state = {
        "original_query": question,
        "session_id": str(uuid.uuid4()),
        "is_stream": False,
    }

    try:
        # 2. 执行查询流程（query_app.invoke），返回最终 state
        result = query_app.invoke(state)
    except Exception as e:
        print(f"[step4] RAG failed: {type(e).__name__}: {e}")
        return "", []

    # 3. 从返回 state 中获取生成答案
    answer = result.get("answer", "") or ""

    # 4. 从返回 state 的 reranked_docs 中提取检索上下文
    #    每项结构为 {"text", "score", "source"}，text 为正文；兼容 content/snippet 字段
    contexts: List[str] = []
    for doc in (result.get("reranked_docs") or []):
        text = (doc.get("text") or doc.get("content") or doc.get("snippet") or "").strip()
        if text:
            contexts.append(text)

    return answer, contexts


def step5_evaluate_question(
    metrics: List[Any],
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, Any]:
    """步骤5：使用 ragas 对单个问答样本计算 5 项指标。

    ragas 0.4.x 的 collections 指标继承 BaseMetric（非旧版 Metric），其 score() 会
    内部 asyncio.run 包装 ascore()，因此逐个调用指标的 score() 方法，并根据每个指标
    ascore() 声明的参数名动态注入所需字段。

    Args:
        metrics: 已注入 llm/embeddings 的指标列表。
        question: 用户问题。
        answer: RAG 生成的答案。
        contexts: 检索上下文列表。
        ground_truth: 参考答案。

    Returns:
        包含原始字段与 5 项指标分数的字典，用于写入 qa_result.csv。
    """
    # 指标 metric.name -> 输出列名 的映射（对齐 qa_result.csv 的列头）
    metric_name_to_column = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision_with_reference": "context_precision",
        "context_recall": "context_recall",
        "answer_correctness": "answer_correctness",
    }

    # 初始化结果行，指标默认值为 None（评估失败时保留占位）
    row: Dict[str, Any] = {
        "question": question,
        "context": "\n".join(contexts),
        "answer": answer,
        "ground_truth": ground_truth,
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
        "answer_correctness": None,
    }

    # 公共字段池：各指标按需从中提取（字段名与各指标的 ascore 形参保持一致）
    base_fields = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": ground_truth,
    }

    # 逐个指标计算分数
    for metric in metrics:
        metric_name = getattr(metric, "name", "")
        column = metric_name_to_column.get(metric_name)
        if not column:
            continue
        try:
            # 1. 动态提取 ascore() 声明的参数名（排除 self），确定该指标需要的字段
            param_names = [p for p in inspect.signature(metric.ascore).parameters if p != "self"]
            # 2. 从公共字段池中筛选该指标所需字段
            kwargs = {p: base_fields[p] for p in param_names if p in base_fields}
            # 3. 调用 score()（同步包装，内部 asyncio.run 执行 ascore）
            result = metric.score(**kwargs)
            row[column] = getattr(result, "value", None)
        except Exception as e:
            print(f"[step5] metric {metric_name} failed: {type(e).__name__}: {e}")
            row[column] = None

    return row


def step6_save_result(rows: List[Dict[str, Any]]) -> None:
    """步骤6：将评估结果保存到 qa_result.csv（UTF-8 BOM 编码）。

    Args:
        rows: 评估结果行列表，每行包含 9 个字段。
    """
    # 固定输出列顺序
    fieldnames = [
        "question",
        "context",
        "answer",
        "ground_truth",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    result_path = os.path.join(EVAL_DIR, "qa_result.csv")
    # 使用 utf-8-sig 编码写出，自动带 BOM，便于 Excel 直接打开
    with open(result_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[step6] result saved to {result_path} ({len(rows)} rows)")


def main() -> None:
    """评估主流程：依次执行 step1 ~ step6。"""
    # 1. 读取黄金问题集
    qa_pairs = step1_load_qa_data()

    # 2. 初始化 RAG 查询流程入口
    rag_app = step2_init_rag_app()

    # 3. 初始化 ragas 评估组件（llm/embeddings 已注入各指标，此处仅需指标列表）
    _, _, metrics = step3_init_eval_clients()

    # 4. 逐条执行 RAG 流程并评估
    rows: List[Dict[str, Any]] = []
    for index, pair in enumerate(qa_pairs, start=1):
        question = pair["question"]
        ground_truth = pair["ground_truth"]
        print(f"\n[main] ===== {index}/{len(qa_pairs)} =====")

        # 4.1 执行 RAG 流程，获取答案与上下文
        answer, contexts = step4_run_rag_for_question(question)
        print(f"[step4] answer_len={len(answer)} contexts={len(contexts)}")

        # 4.2 使用 ragas 计算指标
        row = step5_evaluate_question(
            metrics=metrics,
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        rows.append(row)

        # 4.3 打印本条指标结果（便于实时观察）
        scores = {k: row[k] for k in row if k in (
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "answer_correctness")}
        print(f"[step5] scores={scores}")

    # 5. 保存评估结果
    step6_save_result(rows)


if __name__ == "__main__":
    main()
