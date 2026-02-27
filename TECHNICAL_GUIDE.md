# 多轮对话 AI 应用技术指南

## 项目概述

这是一个基于 LangGraph + FastAPI 构建的多轮对话 AI 应用，实现了智能查询助手功能。系统能够理解用户意图、查询数据库、格式化结果并生成报告，支持流式输出和会话管理。

### 核心技术栈
- **LangGraph**: 状态机工作流编排
- **LangChain**: LLM 调用和工具集成
- **FastAPI**: Web 服务框架
- **SQLAlchemy**: 数据库 ORM
- **Server-Sent Events (SSE)**: 流式响应

---

## 一、架构设计

### 1.1 整体架构

```
用户请求 → FastAPI → LangGraph 工作流 → 多个 Agent 节点 → 流式响应
                          ↓
                    MemorySaver (会话管理)
```

### 1.2 核心组件

#### Agent 节点
- **Intent Agent**: 意图理解和信息收集
- **Query Agent**: 数据库查询（ReAct Agent）
- **Format Agent**: 结果格式化
- **Report Agent**: 报告生成

#### 状态管理
- 使用 `TypedDict` 定义状态模型
- LangGraph 的 `MemorySaver` 实现会话持久化
- 通过 `thread_id` 区分不同会话

---

## 二、多轮对话实现核心

### 2.1 状态机设计


**状态定义** (`models/schemas.py`):

```python
class AgentState(TypedDict, total=False):
    user_input: str                          # 当前用户输入
    is_complete: bool                        # 信息是否完整
    user_info: Optional[Dict[str, Any]]      # 提取的用户信息
    query_results: Optional[Dict[str, Any]]  # 查询结果
    conversation_history: List[str]          # 对话历史
    next_step: str                           # 下一步路由
    waiting_for_report_confirmation: bool    # 等待报告确认
    performance_metrics: Dict[str, float]    # 性能指标
```

**关键点**：
- `TypedDict` 提供类型提示，但保持字典的灵活性
- `total=False` 允许字段可选，适合动态状态
- 状态在节点间传递，每个节点可读写状态

### 2.2 工作流编排

**工作流定义** (`graph/workflow.py`):

```python
workflow = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

# 添加节点
workflow.add_node("intent", process_intent)
workflow.add_node("query", process_query)
workflow.add_node("format_result", process_format_result)
workflow.add_node("report", process_report)

# 条件路由
workflow.add_conditional_edges(
    "intent",
    route_after_intent,
    {"query": "query", "report": "report", "end": END}
)
```

**路由逻辑**：
- 根据 `state["next_step"]` 决定下一个节点
- 支持动态路由（如报告确认）
- `END` 表示工作流结束


### 2.3 会话管理（核心）

**MemorySaver 的使用**:

```python
# 创建内存检查点
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 使用 thread_id 管理会话
config = {"configurable": {"thread_id": session_id}}

# 调用工作流（自动保存状态）
result = app.invoke({"user_input": "..."}, config)

# 获取会话状态
current_state = app.get_state(config)
conversation_history = current_state.values.get("conversation_history", [])

# 更新会话状态
app.update_state(config, {"conversation_history": updated_history})
```

**工作原理**：
1. **自动持久化**: 每次 `invoke` 后，状态自动保存到 `MemorySaver`
2. **会话隔离**: 不同 `thread_id` 的状态完全独立
3. **状态恢复**: 下次调用时，自动加载上次的状态
4. **手动更新**: 可通过 `update_state` 修改状态（如对话历史）

**关键实现** (`main.py`):

```python
# 获取对话历史
current_state = workflow_app.get_state(config)
if current_state and current_state.values:
    conversation_history = current_state.values.get("conversation_history", [])

# 更新对话历史
conversation_history = update_conversation_history(
    conversation_history,
    user_input,
    assistant_response,
    max_turns=MAX_CONVERSATION_TURNS
)

await loop.run_in_executor(
    None,
    lambda: workflow_app.update_state(
        config,
        {"conversation_history": conversation_history}
    )
)
```


### 2.4 对话历史管理

**历史记录策略**:

```python
def update_conversation_history(
    history: list, 
    user_input: str, 
    assistant_response: str = "", 
    max_turns: int = 10
) -> list:
    """更新对话历史，保留最近 N 轮"""
    history.append(f"用户: {user_input}")
    if assistant_response:
        history.append(f"助手: {assistant_response}")
    
    # 只保留最近的对话（避免 token 过多）
    max_messages = max_turns * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    
    return history
```

**传递给 LLM**:

```python
def build_history_string(history: list, max_turns: int = 3) -> str:
    """构建对话历史字符串（只显示最近几轮）"""
    if not history:
        return "（无历史对话）"
    
    recent_history = history[-(max_turns * 2):]
    return "\n".join(recent_history)

# 在提示词中使用
prompt = f"""对话历史：
{build_history_string(conversation_history)}

当前用户输入：{user_input}"""
```

**设计考虑**：
- **存储**: 保留 10 轮对话（20 条消息）
- **显示**: 只给 LLM 看最近 3 轮（6 条消息）
- **原因**: 平衡上下文理解和 token 消耗

---

## 三、流式输出实现

### 3.1 Server-Sent Events (SSE)

**FastAPI 实现** (`main.py`):

```python
async def generate_stream() -> AsyncGenerator[str, None]:
    """生成流式响应"""
    # 发送开始事件
    yield f"data: {json.dumps({'type': 'start', 'session_id': thread_id})}\n\n"
    
    # 流式输出内容
    for chunk in content_chunks:
        yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
        await asyncio.sleep(0.02)  # 控制输出速度
    
    # 发送结束事件
    yield f"data: {json.dumps({'type': 'end', 'status': 'complete'})}\n\n"

return StreamingResponse(
    generate_stream(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
)
```


### 3.2 LLM 流式调用

**LangChain 流式** (`agents/intent_agent_stream.py`):

```python
# 创建支持流式的 LLM
llm_stream = ChatOpenAI(
    model=settings.MODEL_NAME,
    streaming=True,  # 启用流式
    max_tokens=2000  # 限制 token，加快首字响应
)

chain_stream = prompt | llm_stream

# 流式调用
full_content = ""
for chunk in chain_stream.stream({"user_input": user_input}):
    content = chunk.content
    if content:
        full_content += content
        if callback:
            callback(content)  # 实时回调
```

### 3.3 线程间通信

**问题**: FastAPI 异步环境 + LangChain 同步调用

**解决方案**: 使用 `queue.Queue` + `run_in_executor`

```python
import queue
import asyncio

# 创建队列
content_queue = queue.Queue()

def sync_callback(content: str):
    """同步回调函数（在线程中执行）"""
    content_queue.put(('content', content))

# 在线程池中执行同步任务
loop = asyncio.get_event_loop()
future = loop.run_in_executor(
    None, 
    lambda: process_intent_stream(state, sync_callback)
)

# 异步读取队列
while True:
    try:
        msg_type, content = content_queue.get_nowait()
        yield f"data: {json.dumps({'type': msg_type, 'content': content})}\n\n"
    except queue.Empty:
        if future.done():
            break
        await asyncio.sleep(0.01)
```

**关键点**：
- `run_in_executor`: 在线程池中运行同步代码
- `queue.Queue`: 线程安全的队列，用于线程间通信
- `get_nowait()`: 非阻塞获取，避免死锁
- `asyncio.sleep()`: 让出控制权，避免 CPU 空转


---

## 四、意图理解与信息收集

### 4.1 意图识别策略

**系统提示词设计** (`agents/intent_agent_stream.py`):

```python
INTENT_SYSTEM_PROMPT = """你是智能查询助手，帮助用户查询信息。

核心任务：
1. 理解用户意图（结合对话历史，用户可能分多次提供信息）
2. 收集必要信息：手机号（11位）+ 查询类型
3. 信息不完整时，自然地追问
4. 信息完整时，输出JSON格式

可查询的信息类型：
- package: 套餐信息
- realname: 实名认证状态
- identity: 身份三要素验证

输出规则：
1. 信息完整时，输出JSON：{"complete": true, "query_type": "类型", "params": {"phone": "手机号"}}
2. 其他情况（问候、追问、闲聊），输出自然语言文本

重要：
- query_type 只能是 package、realname、identity 三者之一
- 用户一次只能查询一种信息
- 从对话历史中提取已有的信息
- 理解用户的自然表达，不要过于死板
"""
```

**设计原则**：
1. **简洁明确**: 避免过多示例，让模型自然理解
2. **结构化输出**: 完整信息时输出 JSON，否则输出文本
3. **上下文感知**: 强调结合对话历史
4. **灵活性**: 不限制具体表达方式

### 4.2 信息完整性判断

**流程**:

```python
# 1. 调用 LLM 理解意图
response = llm.invoke({"user_input": user_input, "history": history})

# 2. 尝试解析 JSON
try:
    result = json.loads(response.content)
    
    if result.get("complete"):
        # 信息完整，进入查询流程
        state["is_complete"] = True
        state["user_info"] = result
        state["next_step"] = "query"
    else:
        # 信息不完整，返回提示
        state["is_complete"] = False
        state["error"] = result.get("response")
        state["next_step"] = "end"
        
except json.JSONDecodeError:
    # 不是 JSON，说明是自然语言回复
    state["is_complete"] = False
    state["error"] = response.content
    state["next_step"] = "end"
```

**关键点**：
- JSON 解析成功 + `complete: true` → 进入查询
- JSON 解析失败 → 自然语言回复（追问或闲聊）
- 不输出 JSON 到前端，只用于内部判断


---

## 五、ReAct Agent 实现

### 5.1 什么是 ReAct Agent

**ReAct** = Reasoning + Acting

- **Reasoning**: 思考下一步该做什么
- **Acting**: 执行工具调用
- **循环**: 根据工具结果继续思考和行动

### 5.2 工具定义

**MCP 工具** (`tools/mcp_tools.py`):

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def query_user_by_phone(phone: Annotated[str, "11位手机号码"]) -> Dict[str, Any]:
    """
    根据电话号码查询用户基本信息。
    这是第一步必须调用的工具，用于获取用户的user_id等基本信息。
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT id, phone, username FROM users WHERE phone = :phone"),
            {"phone": phone}
        )
        row = result.fetchone()
        
        if row:
            return {
                "success": True, 
                "data": serialize_row(dict(row._mapping)),
                "_perf": {"query_user_by_phone": elapsed_ms}
            }
        return {"success": False, "error": "未找到该手机号对应的用户信息"}
```

**关键点**：
- `@tool` 装饰器：将函数转换为 LangChain 工具
- `Annotated`: 提供参数描述，帮助 LLM 理解
- 文档字符串：详细说明工具用途和使用方法
- 返回格式统一：`{"success": bool, "data": dict, "_perf": dict}`

### 5.3 ReAct Agent 创建

**创建 Agent** (`agents/query_agent.py`):

```python
from langgraph.prebuilt import create_react_agent

# 创建 LLM
llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0)

# 创建 ReAct Agent（自动处理工具调用循环）
query_agent = create_react_agent(llm, ALL_TOOLS)

# 调用 Agent
result = query_agent.invoke({
    "messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query_message)
    ]
})
```

**工作流程**：
1. LLM 分析任务，决定调用哪个工具
2. 执行工具，获取结果
3. LLM 分析结果，决定下一步
4. 重复 1-3，直到任务完成


### 5.4 工具结果提取

**解析 Agent 返回** (`agents/query_agent.py`):

```python
tool_results = {}

if "messages" in result:
    for msg in result["messages"]:
        # 只处理工具消息（ToolMessage）
        if hasattr(msg, 'name') and hasattr(msg, 'content') and msg.name:
            tool_name = msg.name
            
            # 解析工具返回
            if isinstance(msg.content, dict):
                tool_content = msg.content
            elif isinstance(msg.content, str):
                tool_content = json.loads(msg.content)
            
            # 保存结果
            tool_results[tool_name] = tool_content
            
            # 提取性能数据
            if "_perf" in tool_content:
                perf_data = tool_content.pop("_perf")
                state["performance_metrics"].update(perf_data)
```

**关键点**：
- 过滤 `ToolMessage`：只处理工具调用结果
- 统一格式：处理字典和 JSON 字符串
- 性能追踪：提取 `_perf` 字段
- 错误处理：检查 `success` 字段

---

## 六、提示词工程

### 6.1 系统提示词设计原则

**好的提示词**：
```python
QUERY_SYSTEM_PROMPT = """你是数据查询助手，负责调用工具查询数据库。

可用工具：
- query_user_by_phone(phone): 查询用户基本信息，返回 user_id 等
- query_user_package(user_id): 查询套餐信息
- query_realname_info(user_id): 查询实名信息

工作流程：
1. 先调用 query_user_by_phone 获取用户信息
2. 检查返回的 success 字段，false 表示用户不存在
3. 根据查询类型调用对应工具

注意：
- 必须完成两步：先查用户，再查详细信息
- user_id 从第一步的 data.id 中获取
"""
```

**设计原则**：
1. **简洁明确**: 避免冗长的描述
2. **结构清晰**: 使用列表和分段
3. **关键信息**: 突出必须遵守的规则
4. **避免过度约束**: 不要列举所有可能的场景


### 6.2 动态提示词构建

**根据查询类型构建提示** (`agents/query_agent.py`):

```python
query_message = f"""用户输入：{state["user_input"]}
查询类型：{query_type}
手机号：{params.get('phone')}

请执行查询：
1. 调用 query_user_by_phone({params.get('phone')})
2. 如果成功，根据查询类型调用对应工具"""

if query_type == "package":
    query_message += "\n   - 调用 query_user_package(user_id)"
elif query_type == "realname":
    query_message += "\n   - 调用 query_realname_info(user_id)"
elif query_type == "identity":
    query_message += "\n   - 调用 query_identity_info(name, id_card, phone)"
```

**优势**：
- 根据实际情况动态生成
- 避免无关信息干扰
- 提高 LLM 理解准确性

---

## 七、性能优化

### 7.1 性能计时器

**上下文管理器实现** (`utils/decorators.py`):

```python
class PerformanceTimer:
    """性能计时器上下文管理器"""
    
    def __init__(self, name: str, metrics_dict: dict = None):
        self.name = name
        self.metrics_dict = metrics_dict
        self.start_time = None
        self.elapsed_ms = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        logger.debug(f"{self.name} 耗时: {self.elapsed_ms:.0f}ms")
        
        if self.metrics_dict is not None:
            self.metrics_dict[self.name] = self.elapsed_ms
        
        return False

# 使用
with PerformanceTimer("query_total", state["performance_metrics"]):
    # 执行查询
    result = query_agent.invoke(messages)
```

**优势**：
- 自动计时，无需手动记录
- 统一管理性能指标
- 支持嵌套计时

### 7.2 快速响应缓存

**常见问候缓存** (`main.py`):

```python
QUICK_RESPONSES = {
    "你好": "您好！我是智能查询助手...",
    "hi": "Hello! I'm an intelligent query assistant...",
}

# 检查缓存
user_input_lower = user_input.strip().lower()
if user_input_lower in QUICK_RESPONSES:
    response_text = QUICK_RESPONSES[user_input_lower]
    # 直接返回，跳过 LLM 调用
    yield response_text
    return
```

**效果**：
- 常见问候响应时间从 2-3 秒降至 < 100ms
- 减少 LLM API 调用成本


### 7.3 数据序列化优化

**处理 Decimal 类型** (`tools/mcp_tools.py`):

```python
from decimal import Decimal
from datetime import datetime, date

def serialize_value(value):
    """将数据库值转换为 JSON 兼容的类型"""
    if isinstance(value, Decimal):
        return float(value)
    elif isinstance(value, (datetime, date)):
        return value.isoformat()
    elif value is None:
        return None
    else:
        return value

def serialize_row(row_mapping: dict) -> dict:
    """将数据库行转换为 JSON 兼容的字典"""
    return {key: serialize_value(value) for key, value in row_mapping.items()}
```

**问题**：
- SQLAlchemy 返回的 `Decimal` 类型无法直接 JSON 序列化
- `datetime` 对象也需要转换

**解决**：
- 统一序列化函数
- 在工具返回前处理

---

## 八、错误处理与日志

### 8.1 统一错误处理

**装饰器模式** (`utils/decorators.py`):

```python
def handle_agent_error(error_message: str = "处理失败"):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            try:
                return func(state, *args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} 错误:\n{traceback.format_exc()}")
                
                # 根据错误类型返回友好提示
                if "500" in str(e) or "InternalServerError" in str(e):
                    state["error"] = "抱歉，服务暂时不可用，请稍后重试。"
                elif "timeout" in str(e).lower():
                    state["error"] = "请求超时，请稍后重试。"
                else:
                    state["error"] = f"{error_message}: {str(e)}"
                
                state["next_step"] = "end"
                return state
        return wrapper
    return decorator

# 使用
@handle_agent_error("意图理解失败")
def process_intent(state: AgentState) -> AgentState:
    # 处理逻辑
    pass
```

### 8.2 结构化日志

**自定义 Logger** (`utils/logger.py`):

```python
class CustomLogger:
    def separator(self, char="=", length=80):
        """打印分隔线"""
        print(char * length)
    
    def info(self, msg):
        print(f"[{datetime.now()}] [INFO] {msg}")
    
    def debug(self, msg):
        print(f"[{datetime.now()}] [DEBUG] {msg}")
    
    def error(self, msg):
        print(f"[{datetime.now()}] [ERROR] {msg}")

logger = CustomLogger()
```

**使用**：
```python
logger.separator()
logger.info("开始数据查询")
logger.debug(f"查询参数: {params}")
logger.error(f"查询失败: {error}")
```


---

## 九、前端集成

### 9.1 EventSource 接收 SSE

**前端代码** (`static/chat.html`):

```javascript
const eventSource = new EventSource('/api/chat-stream', {
    method: 'POST',
    body: JSON.stringify({
        user_input: userInput,
        session_id: sessionId
    })
});

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'start':
            sessionId = data.session_id;
            break;
        case 'content':
            appendToMessage(data.content);
            break;
        case 'performance':
            console.log('性能指标:', data.metrics);
            break;
        case 'end':
            eventSource.close();
            break;
    }
};

eventSource.onerror = (error) => {
    console.error('SSE 错误:', error);
    eventSource.close();
};
```

**事件类型**：
- `start`: 会话开始，返回 session_id
- `content`: 流式内容
- `performance`: 性能指标
- `report`: 报告下载链接
- `end`: 流式结束

### 9.2 会话 ID 管理

```javascript
let sessionId = localStorage.getItem('session_id');

if (!sessionId) {
    sessionId = generateUUID();
    localStorage.setItem('session_id', sessionId);
}

// 每次请求携带 session_id
fetch('/api/chat-stream', {
    method: 'POST',
    body: JSON.stringify({
        user_input: userInput,
        session_id: sessionId
    })
});
```

**关键点**：
- 使用 `localStorage` 持久化 session_id
- 同一会话的所有请求使用相同 session_id
- 服务端通过 session_id 恢复对话历史

---

## 十、最佳实践总结

### 10.1 多轮对话设计

1. **状态管理**
   - 使用 LangGraph 的 `MemorySaver` 自动持久化
   - 通过 `thread_id` 隔离不同会话
   - 定期清理过期会话（可选）

2. **对话历史**
   - 存储完整历史（10 轮）
   - 只给 LLM 看最近几轮（3 轮）
   - 平衡上下文理解和 token 消耗

3. **信息收集**
   - 允许用户分多次提供信息
   - 从历史中提取已有信息
   - 自然追问缺失信息


### 10.2 提示词工程

1. **简洁明确**
   - 避免冗长描述
   - 突出关键规则
   - 使用结构化格式

2. **避免过度约束**
   - 不要列举所有场景
   - 让模型自然理解
   - 相信模型的泛化能力

3. **动态构建**
   - 根据实际情况生成提示
   - 避免无关信息干扰
   - 提高理解准确性

### 10.3 流式输出

1. **用户体验**
   - 立即响应，避免等待
   - 逐字输出，模拟打字效果
   - 控制输出速度（20ms/chunk）

2. **技术实现**
   - 使用 SSE（Server-Sent Events）
   - 线程池 + 队列处理同步/异步
   - 非阻塞读取，避免死锁

3. **错误处理**
   - 捕获流式过程中的异常
   - 发送错误事件到前端
   - 确保连接正确关闭

### 10.4 性能优化

1. **缓存策略**
   - 常见问候快速响应
   - 减少不必要的 LLM 调用
   - 降低 API 成本

2. **性能监控**
   - 记录每个环节耗时
   - 识别性能瓶颈
   - 持续优化

3. **数据库优化**
   - 使用连接池（可选）
   - 优化 SQL 查询
   - 添加索引

### 10.5 错误处理

1. **分层处理**
   - Agent 层：捕获业务逻辑错误
   - API 层：捕获网络和系统错误
   - 前端：展示友好错误提示

2. **友好提示**
   - 根据错误类型返回不同提示
   - 避免暴露技术细节
   - 引导用户正确操作

3. **日志记录**
   - 记录完整错误堆栈
   - 便于问题排查
   - 支持性能分析

---

## 十一、扩展方向

### 11.1 功能扩展

1. **更多查询类型**
   - 添加新的工具函数
   - 更新意图识别提示词
   - 扩展路由逻辑

2. **多模态支持**
   - 图片输入（OCR 识别）
   - 语音输入（ASR 转文字）
   - 文件上传（解析内容）

3. **个性化**
   - 用户偏好记忆
   - 自定义回复风格
   - 智能推荐


### 11.2 架构优化

1. **持久化存储**
   - 使用 Redis 替代 MemorySaver
   - 支持分布式部署
   - 会话数据持久化

2. **消息队列**
   - 异步处理长时间任务
   - 解耦服务依赖
   - 提高系统吞吐量

3. **微服务化**
   - 拆分 Agent 为独立服务
   - 独立扩展和部署
   - 提高系统可维护性

### 11.3 监控与运维

1. **监控指标**
   - 请求量、响应时间
   - 错误率、成功率
   - LLM token 消耗

2. **日志分析**
   - 集中式日志收集
   - 日志查询和分析
   - 告警机制

3. **A/B 测试**
   - 不同提示词效果对比
   - 模型版本对比
   - 用户体验优化

---

## 十二、常见问题

### Q1: 为什么使用 LangGraph 而不是直接调用 LLM？

**A**: LangGraph 提供了：
- **状态管理**: 自动持久化会话状态
- **工作流编排**: 清晰的节点和路由逻辑
- **可观测性**: 内置的状态追踪和调试
- **可扩展性**: 易于添加新节点和路由

### Q2: 如何处理 LLM 返回格式不一致的问题？

**A**: 
1. **提示词约束**: 明确要求输出格式
2. **多次尝试**: 解析失败时重试
3. **兜底策略**: 解析失败时使用默认值
4. **结构化输出**: 使用 LangChain 的 `with_structured_output`

### Q3: 如何优化 LLM 响应速度？

**A**:
1. **流式输出**: 立即返回首字
2. **缓存**: 常见问题快速响应
3. **并行调用**: 多个独立任务并行
4. **模型选择**: 使用更快的模型（如 GPT-3.5）
5. **限制 token**: 减少 max_tokens 参数

### Q4: 如何保证多轮对话的上下文一致性？

**A**:
1. **会话隔离**: 使用 thread_id 区分会话
2. **状态持久化**: 使用 MemorySaver 或 Redis
3. **历史管理**: 保留足够的对话历史
4. **状态验证**: 检查状态完整性

### Q5: 如何处理并发请求？

**A**:
1. **异步处理**: FastAPI 原生支持异步
2. **线程池**: 使用 `run_in_executor` 处理同步任务
3. **连接池**: 数据库连接池
4. **限流**: 使用 rate limiting 中间件

---

## 十三、总结

本项目展示了如何构建一个完整的多轮对话 AI 应用，核心技术点包括：

1. **LangGraph 状态机**: 清晰的工作流编排和状态管理
2. **会话管理**: 基于 MemorySaver 的自动持久化
3. **流式输出**: SSE + 线程池实现实时响应
4. **ReAct Agent**: 自主决策和工具调用
5. **提示词工程**: 简洁明确的系统提示词
6. **性能优化**: 缓存、计时器、序列化优化
7. **错误处理**: 分层处理和友好提示

通过学习本项目，你可以掌握：
- 如何设计多轮对话的状态机
- 如何管理会话和对话历史
- 如何实现流式输出
- 如何使用 ReAct Agent 进行工具调用
- 如何优化 LLM 应用的性能和用户体验

希望这份技术指南能帮助你深入理解多轮对话 AI 应用的构建技术！

---

## 附录：项目结构

```
langraph_1/
├── agents/                    # Agent 节点
│   ├── intent_agent.py       # 意图理解（非流式）
│   ├── intent_agent_stream.py # 意图理解（流式）
│   ├── query_agent.py        # 数据查询（ReAct）
│   ├── result_formatter.py   # 结果格式化
│   └── report_agent.py       # 报告生成
├── graph/                     # LangGraph 工作流
│   └── workflow.py           # 工作流定义
├── models/                    # 数据模型
│   └── schemas.py            # Pydantic 模型
├── tools/                     # 工具集
│   └── mcp_tools.py          # MCP 工具（数据库查询）
├── utils/                     # 工具函数
│   ├── constants.py          # 常量定义
│   ├── decorators.py         # 装饰器和工具函数
│   ├── logger.py             # 日志工具
│   └── validators.py         # 输入验证
├── static/                    # 静态文件
│   └── chat.html             # 前端页面
├── reports/                   # 生成的报告
├── config.py                  # 配置文件
├── main.py                    # FastAPI 入口
└── .env                       # 环境变量
```

---

**作者**: AI Assistant  
**日期**: 2026-02-26  
**版本**: 1.0
