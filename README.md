# 论文内容查询与解释智能体

这是一个运行在 Windows 本地的 FastAPI + Streamlit + LangChain RAG 应用。它针对目录中的中文论文 PDF，提取页级文字和嵌入图片，尝试解析表格，并建立可追溯的页码引用；问答阶段通过 OpenAI-compatible 服务生成中文解释。

## 1. 安装

建议使用 Python 3.11 或更高版本：

```powershell
cd D:\论文内容查询
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

编辑 `.env`（不要直接修改 `.env.example`，也不要把真实密钥提交到代码库）：

```dotenv
DASHSCOPE_API_KEY=你的密钥
OPENAI_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
CHAT_MODEL=qwen3.8-omni-flash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_CACHE_DIR=
```

- `OPENAI_API_KEY`：通用 OpenAI-compatible 服务密钥，可留空。
- `DASHSCOPE_API_KEY`：通义千问兼容接口密钥；当 `OPENAI_API_KEY` 留空时作为聊天模型密钥使用。
- `OPENAI_BASE_URL`：兼容服务地址，DashScope 示例为 `https://maas.qianwenaiapi.com/compatible-mode/v1`。
- `CHAT_MODEL`：聊天模型名称，DashScope 示例为 `qwen3.8-omni-flash`；未配置有效密钥时仍可以建立索引和查询，查询会返回带页码的原文证据。
- `EMBEDDING_MODEL`：可选本地 Hugging Face embedding 模型名称或本地目录；当前环境可使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（已缓存）。首次使用其他模型可能需要下载模型文件并占用本机内存；模型下载完成后向量生成在本地执行，论文文本不会发送到 embedding 服务。留空时使用 BM25。
- `EMBEDDING_DEVICE`：本地模型设备，默认 `cpu`；可按环境设为 `cuda`。
- `EMBEDDING_CACHE_DIR`：可选模型缓存目录，留空时使用 Hugging Face 默认缓存。
- `ENABLE_VECTOR`：是否允许 Chroma 分支，默认 `true`。
- `ENABLE_OCR`：是否启用 RapidOCR，默认 `false`。开启后需要 `rapidocr-onnxruntime`；缺少依赖时会写入 warning 而不会阻塞摄取。
- `VECTOR_COLLECTION`：Chroma collection 名称。

LangChain 链路为：`Document` → `BM25Retriever` + 本地 Hugging Face embeddings / Chroma → RRF 排名融合 → `ChatPromptTemplate` → `ChatOpenAI`。未配置本地 embedding 模型或模型加载失败时，系统自动保留 BM25 检索；本地向量索引写入 `app_data/chroma`，同步状态保存在本地 `app_data/chroma_sync.json`。

## 2. 启动

后端：

```powershell
.\.venv\Scripts\activate
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

首次建立索引（另一个窗口）：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/ingest
```

启动前端：

```powershell
streamlit run frontend\streamlit_app.py
```

浏览器打开 Streamlit 显示的地址。也可以点击侧边栏的“重新建立索引”。

## 3. API

- `GET /api/health`：健康检查。
- `GET /api/status`：论文、索引、处理统计和 retrieval mode。
- `POST /api/ingest?force=true`：解析并重建索引。
- `POST /api/query`：请求体 `{"question":"你的问题", "top_k":6, "page":null}`。
- `/resources/images/...`、`/resources/tables/...`：访问已生成的图片和表格资源。

查询回答只使用检索到的论文证据，并附带页码。没有索引或没有检索结果时不会伪造答案；没有配置在线模型时，接口会返回证据原文清单而不是生成解释。

## 4. 摄取和检索能力

- PyMuPDF 页级中文文本解析；重复出现的短文本会被识别为页眉/页脚噪声并从正文检索文本中剔除。
- 嵌入图片导出到 `app_data/images`，查询引用会转换为受限的资源 URL。
- `pdfplumber` 可将检测到的表格写为 `app_data/tables/*.md` 并建立表格 chunk；缺少该依赖或 PDF 表格无法解析时会保留页面文本并写入 warning。
- `ENABLE_OCR=true` 时尝试对页面渲染图执行 RapidOCR，OCR 失败会降级到已有文本。
- 重叠切块、LangChain BM25、可选 Chroma 持久化向量分支、页码筛选和 hash 幂等摄取。
- 资源 URL 只映射到 `app_data/images` 或 `app_data/tables` 中真实存在的文件，避免把 manifest 中的任意路径直接暴露给客户端。

## 5. 测试和限制

```powershell
python -m compileall backend frontend tests
python -m pytest -q
```

测试不需要在线大模型；真实回答需要有效的 OpenAI-compatible 配置。若只验证基础服务，可以检查 health、status、ingest 和 BM25 模式。向量、OCR、表格解析和真实问答必须在相应依赖、服务和凭据可用时单独验收。
