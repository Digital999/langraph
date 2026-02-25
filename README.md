# AI 智能查询助手

基于 FastAPI + LangGraph 的多 Agent 协同查询系统，支持实时流式输出和会话管理。

## 特性

- 🤖 多 Agent 协同工作（意图理解、数据查询、报告生成）
- 🔄 MCP 工具架构，大模型自主决策工具调用
- 💬 实时流式输出，优秀的用户体验
- 📊 性能监控和统计
- 🔐 会话管理，支持多轮对话
- 📄 自动生成 Word 报告

## 快速开始

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

```bash
uv sync
```

### 3. 配置环境

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置数据库和 API 密钥。

### 4. 初始化数据库

```bash
uv run python init_db.py
```

### 5. 启动服务

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
意图理解 Agent (识别查询类型和参数)
    ↓
查询 Agent (ReAct Agent 自主调用工具)
    ↓
结果格式化 (返回关键信息)
    ↓
[用户确认] → 报告生成 Agent (生成 Word 报告)
```

### MCP 工具

系统使用 Model Context Protocol 管理工具，大模型自主决策调用：

- `query_user_by_phone`: 根据手机号查询用户
- `query_user_package`: 查询用户套餐
- `query_realname_info`: 查询实名信息
- `query_identity_info`: 验证身份三要素

## 使用示例

### Web 界面

访问 `http://localhost:8000`，在聊天界面输入：

```
查询17775711190的套餐信息
```

系统会：
1. 识别查询意图和手机号
2. 自动查询数据库
3. 返回格式化的结果
4. 询问是否需要生成报告

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
├── graph/               # LangGraph 工作流
├── models/              # 数据模型
├── tools/               # MCP 工具集
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
make clean      # 清理临时文件
make help       # 查看所有命令
```

## 技术栈

- **Web 框架**: FastAPI
- **Agent 框架**: LangGraph
- **LLM**: OpenAI API (支持自定义端点)
- **数据库**: MySQL + SQLAlchemy
- **包管理**: uv
- **文档生成**: python-docx

## 开发

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
