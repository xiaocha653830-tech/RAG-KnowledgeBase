# 掌柜智库 · RAG KnowledgeBase

> 面向商品/设备知识库场景的 RAG（检索增强生成）智能问答系统。
> 支持 PDF / Markdown 文档导入 → 版面解析 → 语义切分 → 混合向量入库 → 多路并行召回 → 融合重排 → 流式答案生成。

基于 **LangGraph** 编排的节点化 RAG 全流程，内置 **混合向量检索（稠密 + 稀疏）**、**RRF 融合**、**HyDE 假设性文档增强**、**Rerank 断崖检测动态 TopK** 与 **SSE 流式输出**。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **非结构化文档语义重建** | 基于 MinerU 将 PDF 还原为结构化 Markdown，保留标题层级、表格与图文关系 |
| **图文联合解析** | 提取 Markdown 中的图片，调用多模态大模型（Qwen3-VL）生成图片摘要并回填正文 |
| **语义自治切片** | 递归字符切分 + 重叠窗口，保证每个 chunk 是最小可解释的知识单元 |
| **混合向量索引** | BGE-M3 单模型双输出：1024 维稠密向量（语义）+ 稀疏词权重（关键词），Milvus `hybrid_search` 加权融合 |
| **多路并行召回** | 向量检索 ∥ HyDE 检索 ∥ 网络搜索(MCP) ∥ 知识图谱，四路并发执行 |
| **RRF 融合 + 重排** | 倒数排名融合统一多路结果，BGE-Reranker 交叉编码器精细重排 |
| **断崖检测动态 TopK** | 依据候选得分曲线自动截断，避免机械取前 N 条，剔除长尾噪声 |
| **SSE 流式输出** | 节点进度 + LLM 增量双通道推送，前端实时反馈 |
| **可控后台任务** | 文件导入流程由后台任务执行，任务状态 `pending → processing → completed / failed` 可轮询 |

---

## 🏗️ 系统架构

系统分为 **导入链路** 与 **查询链路** 两条 LangGraph 工作流，由 FastAPI 分别暴露为独立服务。

### 导入链路（Import Pipeline）

```
上传文件 → node_entry（入口校验 / 路由）
              ├── (Markdown) ──────────────► node_md_img
              └── (PDF) → node_pdf_to_md ──► node_md_img
                                              ↓
                            node_md_img         图片提取 → 多模态摘要 → MinIO 上传 → 回填 URL
                                              ↓
                            node_document_split 递归语义切分 + 重叠
                                              ↓
                            node_item_name_recognition  LLM 提取商品/设备名 → 向量对齐入库
                                              ↓
                            node_bge_embedding  BGE-M3 稠密 + 稀疏向量
                                              ↓
                            node_import_milvus  批量写入 Milvus（kb_chunks / kb_item_names）
                                              ↓
                                             END
```

### 查询链路（Query Pipeline）

```
用户提问 → node_item_name_confirm（商品名确认 / 指代消解 / 查询改写）
              ├── 命中唯一商品 ──► node_multi_search ─┬─► node_search_embedding       （向量混合检索）
              │                                        ├─► node_search_embedding_hyde（HyDE 假设文档检索）
              │                                        ├─► node_web_search_mcp        （MCP 联网搜索）
              │                                        └─► node_query_kg              （知识图谱检索）
              │                                                  ↓  node_join
              │                                        node_rrf（加权 RRF 融合）
              │                                                  ↓
              │                                        node_rerank（重排 + 断崖检测动态 TopK）
              │                                                  ↓
              │                                        node_answer_output（提示词组装 → 流式生成）
              └── 无法唯一确认 ──► 直接生成反问 / 拒答 ──► node_answer_output
```

> 多路搜索通过虚拟节点 `node_multi_search`（分叉）与 `node_join`（汇聚）组织，实现并行召回。
> 当商品名无法唯一确认（多型号歧义或查无此商品）时，直接跳过检索生成反问句或拒答，避免无效检索。

---

## 🧰 技术栈

### 编排与框架

| 组件 | 用途 |
|------|------|
| [LangGraph](https://langchain-ai.github.io/langgraph) | 工作流编排（DAG、条件分支、并行） |
| [LangChain](https://python.langchain.com) | LLM 统一调用接口 |
| [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) | Web 服务与 ASGI 运行 |
| [Pydantic](https://docs.pydantic.dev) | 请求 / 响应模型校验 |

### 模型

| 模型 | 用途 |
|------|------|
| **BGE-M3**（`BAAI/bge-m3`） | 混合向量嵌入：稠密 1024 维 + 稀疏词权重 |
| **BGE-Reranker-Large** | 交叉编码器精排 |
| **Qwen-Flash**（OpenAI 兼容接口） | 商品名识别、查询改写、HyDE 生成、答案生成 |
| **Qwen3-VL-Plus** | 图片内容理解与摘要 |

### 中间件

| 中间件 | 用途 |
|--------|------|
| **Milvus** | 向量数据库：`kb_chunks`（切片向量）、`kb_item_names`（商品名向量） |
| **MongoDB** | 会话历史存储（多轮上下文） |
| **MinIO** | 对象存储：原始文件与文档图片 |
| **Neo4j** | 知识图谱（查询节点已预留，图谱构建待接入） |
| **MinerU**（在线 API） | PDF → Markdown 版面解析 |
| **百炼 MCP** | 联网搜索（`node_web_search_mcp`） |

---

## 📁 目录结构

```
knowledge_base/
├── app/
│   ├── clients/                 # 基础设施客户端
│   │   ├── milvus_utils.py      #   Milvus 混合检索封装
│   │   ├── mongo_history_utils.py  #   会话历史
│   │   ├── minio_utils.py       #   对象存储
│   │   └── neo4j_utils.py       #   知识图谱驱动
│   ├── conf/                    # 各中间件 / 模型配置（dataclass + .env）
│   ├── core/                    # 日志、提示词加载
│   ├── import_process/          # 导入链路
│   │   ├── agent/
│   │   │   ├── state.py         #   ImportGraphState 状态定义
│   │   │   ├── main_graph.py    #   导入图编排
│   │   │   └── nodes/           #   7 个导入节点
│   │   ├── api/file_import_service.py  # 导入 Web 服务（:8000）
│   │   └── page/import.html     #   导入前端页面
│   ├── query_process/           # 查询链路
│   │   ├── agent/
│   │   │   ├── state.py         #   QueryGraphState 状态定义
│   │   │   ├── main_graph.py    #   查询图编排
│   │   │   └── nodes/           #   9 个查询节点
│   │   ├── api/query_service.py # 查询 Web 服务（:8001）
│   │   └── page/chat.html       #   问答前端页面
│   ├── lm/                      # LLM / Embedding / Reranker 封装
│   ├── utils/                   # 通用工具（SSE、任务、路径、限流等）
│   └── tool/                    # 模型下载脚本
├── prompts/                     # 提示词模板（.prompt）
├── eval/                        # 基于 Ragas 的检索 / 生成效果评估
├── test/                        # 各节点单元测试脚本
├── sse/                         # SSE 分步学习示例
├── fastapi/                     # FastAPI 基础示例
├── doc/                         # 示例文档（PDF）
├── start_import.bat             # 一键启动导入服务
├── start_query.bat              # 一键启动问答服务
├── pyproject.toml               # 依赖声明（uv 管理）
└── cleanup_milvus.py            # Milvus 集合清理脚本
```

---

## 🚀 快速开始

### 1. 前置依赖

需先启动以下服务（推荐 Docker）：

```yaml
# docker-compose.yml 参考
services:
  milvus:
    image: milvusdb/milvus:v2.4.x
    ports: ["19530:19530"]
  mongodb:
    image: mongo:6.0
    ports: ["27017:27017"]
  minio:
    image: minio/minio:latest
    ports: ["9000:9000", "9001:9001"]
    command: server /data --console-address ":9001"
  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]
```

### 2. 安装依赖

项目使用 [uv](https://github.com/astral-sh/uv) 管理虚拟环境，已配置清华 PyPI 镜像：

```bash
uv sync                          # 创建 .venv 并安装依赖
```

模型权重需提前下载到本地（路径写入 `.env`），或使用内置脚本：

```bash
python app/tool/download_bgem3.py
python app/tool/download_reranker.py
```

### 3. 配置环境变量

在项目根目录创建 `.env`，按下方清单填写：

```ini
# ---- LLM / 多模态 ----
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxxxxxxx
LLM_DEFAULT_MODEL=qwen-flash
LLM_DEFAULT_TEMPERATURE=0.1
VL_MODEL=qwen3-vl-plus

# ---- 本地模型路径 ----
BGE_M3_PATH=/path/to/bge-m3
BGE_RERANKER_LARGE=/path/to/bge-reranker-large
BGE_FP16=0
BGE_RERANKER_FP16=0

# ---- 向量库 ----
MILVUS_URL=http://127.0.0.1:19530
CHUNKS_COLLECTION=kb_chunks
ITEM_NAME_COLLECTION=kb_item_names
EMBEDDING_DIM=1024

# ---- 会话历史 / 对象存储 / 图谱 ----
MONGO_URL=mongodb://127.0.0.1:27017
MONGO_DB_NAME=kb002
MINIO_ENDPOINT=127.0.0.1:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base-files
MINIO_IMG_DIR=/upload-images
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# ---- MinerU PDF 解析 ----
MINERU_BASE_URL=https://mineru.net/api/v4
MINERU_API_TOKEN=your-token
```

### 4. 启动服务

```bash
# 方式一：双击批处理脚本
start_import.bat    # 导入服务 → http://127.0.0.1:8000/import.html
start_query.bat     # 问答服务 → http://127.0.0.1:8001/chat.html

# 方式二：手动启动
.venv/Scripts/python.exe -m uvicorn app.import_process.api.file_import_service:app --host 127.0.0.1 --port 8000 --reload
.venv/Scripts/python.exe -m uvicorn app.query_process.api.query_service:app   --host 127.0.0.1 --port 8001 --reload
```

---

## 🔌 核心接口

### 导入服务（`:8000`）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/import.html` | 导入前端页面 |
| `POST` | `/upload` | 多文件批量上传，触发后台导入全流程 |
| `GET` | `/status/{task_id}` | 查询任务进度（含已完成节点列表） |
| `GET` | `/docs` | Swagger 接口文档 |

### 查询服务（`:8001`）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/chat.html` | 问答前端页面 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/query` | 提交问题（`query` / `session_id` / `is_stream`） |
| `GET` | `/stream/{session_id}` | SSE 流式通道：节点进度 + 答案增量 |
| `GET` | `/history/{session_id}` | 拉取会话历史 |
| `DELETE` | `/history/{session_id}` | 清空会话历史 |

**SSE 事件类型**：`ready`（连接建立）、`progress`（节点进度）、`delta`（LLM 增量）、`final`（完整答案）、`error`（异常）。

---

## 🔍 关键实现

### 混合向量检索

BGE-M3 一次编码即可输出两种向量，兼顾「懂语义」与「查字面」：

```python
dense_vector  = model.encode(texts)["dense_vecs"]        # 稠密：语义相似
sparse_vector = model.encode(texts)["lexical_weights"]   # 稀疏：关键词权重

results = collection.hybrid_search(
    [AnnSearchRequest(dense_vector,  "dense_vector",  ...),
     AnnSearchRequest(sparse_vector, "sparse_vector", ...)],
    rerank=WeightedRanker(0.7, 0.3),   # 稠密 : 稀疏 = 7 : 3
)
```

### RRF 加权融合

对多路召回结果按 `score = Σ wᵢ / (k + rankᵢ)`（k 默认 60）融合，某文档在多路中同时靠前即获得更高权重，避免单路结果「劫持」最终排名。

### 断崖检测动态 TopK

重排后按得分曲线寻找「断崖」拐点，截断后续长尾候选，而非机械取前 N 条，兼顾召回率与上下文质量。

---

## 📊 效果评估

`eval/` 内置基于 [Ragas](https://github.com/explodinggradients/ragas) 的评估脚本，从黄金问题集 `eval/qa.csv` 出发，调用查询链路获取检索上下文与生成答案，计算 5 项指标：

- **Faithfulness** — 答案对上下文的忠实度
- **Answer Relevancy** — 答案与问题的相关性
- **Context Precision / Recall** — 上下文精准度 / 召回率
- **Answer Correctness** — 答案正确性

```bash
.venv/Scripts/python.exe eval/eval.py   # 结果输出至 eval/qa_result.csv
```

---

## 🛠️ 可优化方向

| 方向 | 方案 |
|------|------|
| 查询缓存 | 对 Query Embedding 与检索结果引入 Redis 缓存 |
| 异步化 | 全链路异步改造，提升并发吞吐 |
| 多模态检索 | 图片向量化，支持「以图搜文」 |
| 增量更新 | 支持文档级增量导入与删除 |
| 反馈闭环 | 收集点赞/点踩数据，用于检索与提示词迭代 |

---

## 📚 参考文档

本项目配套笔记（`llm_wiki` 知识库「掌柜智库」系列，共 19 篇）覆盖了从环境搭建到各节点实现、再到 Web 层对接的完整过程，核心主题包括：

- 项目全景与模块流程设计
- 导入处理：入口 / PDF 转 Markdown / 图片处理 / 文档切分 / 商品名识别 / 向量化 / Milvus 入库
- 查询处理：商品名确认 / 向量检索 / HyDE / 网络搜索 / RRF / Rerank / 答案生成
- Web 层与前端对接、项目总结与面试高频题

---

## 📄 License

本项目仅用于学习与研究用途。
