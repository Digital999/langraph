# AI 智能查询助手

基于 FastAPI + LangGraph 的多 Agent 协同查询系统，支持实时流式输出、会话管理和向量检索知识库问答。

## 特性

- 🤖 多 Agent 协同工作（意图理解、数据查询、知识问答、报告生成）
- 🔄 MCP 工具架构，大模型自主决策工具调用
- 🧠 向量检索（RAG）能力，支持通用客服问答
- 💬 实时流式输出，优秀的用户体验
- 📊 性能监控和统计
- 🔐 会话管理，支持多轮对话
- 📄 自动生成 Word 报告

## 快速开始

### 环境要求

- Python 3.11 或 3.12（推荐 3.11）
- uv 包管理器
- MySQL 5.7+（业务数据库）
- PostgreSQL 12+（LangGraph 长期记忆）
- Milvus 向量数据库（知识库检索）
- OpenAI API Key

### 1. 安装 uv

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装依赖

**重要**：确保使用 Python 3.11 或 3.12

```bash
# 检查 Python 版本
python --version

# 如果版本不对，请先安装正确的 Python 版本
# 然后删除旧的虚拟环境
rm -rf .venv  # Linux/macOS
# 或
Remove-Item -Recurse -Force .venv  # Windows PowerShell

# 安装依赖
uv sync
```

如果遇到 Python 版本问题，请参考 `FIX_PYTHON_VERSION.md`。

### 3. 配置环境

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置数据库和 API 密钥：

```env
# Milvus 向量数据库配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=knowledge_base

# PostgreSQL 配置（LangGraph 长期记忆）
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=langgraph_memory
```

### 4. 创建 PostgreSQL 数据库

```bash
# 连接到 PostgreSQL
psql -U postgres

# 创建数据库
CREATE DATABASE langgraph_memory;

# 退出
\q
```

### 5. 测试数据库连接

```bash
# 测试 Milvus 连接
make test-milvus

# 测试 PostgreSQL 连接
make test-postgres
```

### 6. 初始化数据库和向量库

```bash
# 初始化MySQL数据库
uv run python init_db.py

# 初始化向量数据库（知识库）
uv run python scripts/init_vector_store.py
```

或使用 Makefile：
```bash
make init-db
make init-vector
```

### 7. 启动服务

```bash
uv run python main.py
```

或使用 Makefile：
```bash
make run
```

访问 `http://localhost:8000` 使用 Web 聊天界面。

## 系统架构

### Agent 架构

```
用户输入
    ↓
意图理解 Agent (识别查询类型：结构化查询 or 知识问答)
    ↓
    ├─→ [结构化查询] → 查询 Agent (ReAct Agent 自主调用工具)
    │                      ↓
    │                  结果格式化
    │                      ↓
    │                  [用户确认] → 报告生成 Agent
    │
    └─→ [知识问答] → RAG Agent (向量检索 + LLM生成答案)
```

### 查询类型

1. **结构化查询**（查数据库）
   - 套餐查询：查询用户套餐信息
   - 实名查询：查询实名认证状态
   - 身份验证：验证身份三要素

2. **知识问答**（查知识库）
   - 产品介绍：5G套餐优势、功能特点等
   - 常见问题：实名认证、套餐变更等
   - 业务流程：如何办理业务
   - 资费规则：漫游费用、套餐资费等

### MCP 工具

系统使用 Model Context Protocol 管理工具，大模型自主决策调用：

- `query_user_by_phone`: 根据手机号查询用户
- `query_user_package`: 查询用户套餐
- `query_realname_info`: 查询实名信息
- `query_identity_info`: 验证身份三要素

## 使用示例

### Web 界面

访问 `http://localhost:8000`，在聊天界面输入：

**结构化查询示例：**
```
查询17775711190的套餐信息
```

**知识问答示例：**
```
你们的5G套餐有什么优势？
如何办理实名认证？
套餐可以变更吗？
```

系统会自动识别查询类型并返回相应结果。

### API 调用

**流式接口（推荐）:**

```bash
curl -X POST http://localhost:8000/api/chat-stream \
  -H "Content-Type: application/json" \
  -d '{"user_input": "查询17775711190的套餐信息", "session_id": "session_123"}'
```

**标准接口:**

```bash
curl -X POST http://localhost:8000/api/generate-report \
  -H "Content-Type: application/json" \
  -d '{"user_input": "查询17775711190的套餐信息"}'
```

## 测试账号

系统初始化后提供以下测试账号：

- 13800138000 (张三)
- 13900139000 (李四)
- 13700137000 (王五)

## 项目结构

```
langraph_1/
├── agents/              # Agent 模块
│   ├── intent_agent.py  # 意图识别（区分结构化查询和知识问答）
│   ├── query_agent.py   # 数据库查询
│   ├── rag_agent.py     # 知识库问答（新增）
│   ├── report_agent.py  # 报告生成
│   └── result_formatter.py
├── graph/               # LangGraph 工作流
├── models/              # 数据模型
├── tools/               # MCP 工具集
├── rag/                 # 向量检索模块（新增）
│   └── vector_store.py  # 向量数据库管理
├── knowledge_base/      # 知识库文档（新增）
│   ├── product_intro/   # 产品介绍
│   ├── faq/             # 常见问题
│   ├── business_process/# 业务流程
│   └── pricing_rules/   # 资费规则
├── scripts/             # 工具脚本
│   └── init_vector_store.py  # 初始化向量库
├── utils/               # 工具函数
├── static/              # Web 界面
├── main.py              # FastAPI 应用
└── config.py            # 配置文件
```

## 常用命令

使用 Makefile 简化操作：

```bash
make install    # 安装依赖
make run        # 运行应用
make init-db    # 初始化数据库
make test-milvus # 测试 Milvus 连接
make test-postgres # 测试 PostgreSQL 连接
make init-vector # 初始化向量库
make clean      # 清理临时文件
make help       # 查看所有命令
```

## 技术栈

- **Web 框架**: FastAPI
- **Agent 框架**: LangGraph
- **LLM**: OpenAI API (支持自定义端点)
- **向量数据库**: Milvus (高性能向量检索)
- **长期记忆**: PostgreSQL (LangGraph checkpointer)
- **业务数据库**: MySQL + SQLAlchemy
- **Embedding**: OpenAI text-embedding-3-small
- **包管理**: uv
- **文档生成**: python-docx

## 开发

### 添加知识文档

在 `knowledge_base/` 对应目录下添加 `.md` 或 `.txt` 文件：

```bash
# 添加文档后重新初始化向量库
make init-vector
```

### 添加新工具

在 `tools/mcp_tools.py` 中添加：

```python
@tool
def your_new_tool(param: Annotated[str, "参数说明"]) -> Dict[str, Any]:
    """工具描述"""
    # 实现逻辑
    return {"success": True, "data": {...}}
```

### 修改工作流

编辑 `graph/workflow.py` 添加新的节点和路由。

### 自定义 Agent

在 `agents/` 目录下创建新的 Agent 文件。

## 性能优化

系统自动收集性能指标，包括：

- 每个工具的调用耗时
- 每个 Agent 的执行耗时
- 端到端总耗时

查看性能数据：
- 在浏览器控制台（F12）查看详细统计
- 在服务器日志中查看执行日志

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
