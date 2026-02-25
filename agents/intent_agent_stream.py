"""意图理解 Agent - 支持流式输出"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from models.schemas import AgentState
from utils.constants import LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_MAX_RETRIES, MAX_HISTORY_DISPLAY_TURNS
from utils.decorators import build_history_string
from utils.logger import logger
import traceback
import json

INTENT_SYSTEM_PROMPT = """你是一个智能查询助手，可以帮助用户查询套餐信息、实名信息、身份验证等。

你的任务：
1. 如果用户只是打招呼或闲聊，自然地回应并简单介绍你的功能
2. 结合对话历史理解用户意图，用户可能分多次提供信息
3. 如果信息不完整，追问缺少的部分（手机号或查询类型）
4. 如果信息完整，提取并确认

可用的查询功能（用户一次只能选择一个）：
- 套餐信息查询：查询用户的电池套餐详情
- 实名信息查询：查询用户的实名认证状态
- 身份验证：验证姓名、身份证号、手机号三要素是否匹配

当用户提供了完整的查询信息（包含11位电话号码和明确的查询类型）时，请严格按照以下JSON格式输出：
{{"complete": true, "query_type": "类型", "params": {{"phone": "电话号码"}}}}

查询类型：
- package: 查询套餐信息
- identity: 验证身份（需要额外的 name 和 id_card 参数）
- realname: 查询实名信息

其他情况（问候、闲聊、信息不完整等），请自然地用文字回复，不需要JSON格式。

重要规则：
- 结合对话历史理解用户意图，用户可能分多次提供信息
- 如果对话历史中已经有手机号，用户只需要说查询类型即可
- 如果对话历史中已经有查询类型，用户只需要提供手机号即可
- 用户一次只能查询一种信息

示例对话：
对话1：
用户："你好" → 回复："您好！我可以帮您查询：1) 套餐信息 2) 实名信息 3) 身份验证。请告诉我您的手机号码和想查询的内容。"

对话2：
用户："17775711190" → 回复："您想查询什么信息？套餐信息、实名信息，还是身份验证？"
用户："套餐信息" → 回复：{{"complete": true, "query_type": "package", "params": {{"phone": "17775711190"}}}}

对话3：
用户："查询套餐" → 回复："好的，请提供您的11位手机号码。"
用户："17775711190" → 回复：{{"complete": true, "query_type": "package", "params": {{"phone": "17775711190"}}}}

对话4：
用户："查询17775711190的套餐信息" → 回复：{{"complete": true, "query_type": "package", "params": {{"phone": "17775711190"}}}}
"""

# 创建支持流式的 LLM - 优化配置减少延迟
llm_stream = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=LLM_TEMPERATURE,
    streaming=True,  # 启用流式输出
    max_tokens=LLM_MAX_TOKENS,  # 限制最大 token 数，加快首字响应
)

llm_normal = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=LLM_TEMPERATURE,
    max_retries=LLM_MAX_RETRIES,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_SYSTEM_PROMPT),
    ("human", "对话历史：\n{history}\n\n当前用户输入：{user_input}")
])

chain_stream = prompt | llm_stream
chain_normal = prompt | llm_normal

def process_intent_stream(state: AgentState, callback=None):
    """处理用户意图 - 流式版本"""
    try:
        full_content = ""
        
        # 构建对话历史字符串
        history = build_history_string(
            state.get("conversation_history", []),
            max_turns=MAX_HISTORY_DISPLAY_TURNS
        )
        
        # 流式调用
        for chunk in chain_stream.stream({
            "user_input": state["user_input"],
            "history": history
        }):
            content = chunk.content
            if content:
                full_content += content
                # 暂不回调，等解析完成后再决定是否输出
        
        # 更新对话历史
        if "conversation_history" not in state:
            state["conversation_history"] = []
        state["conversation_history"].append(f"用户: {state['user_input']}")
        state["conversation_history"].append(f"助手: {full_content}")
        
        # 尝试解析为 JSON
        try:
            result = json.loads(full_content.strip())
            
            # 如果是完整的查询请求
            if result.get("complete"):
                state["is_complete"] = True
                state["user_info"] = result
                state["next_step"] = "query"
                # JSON 格式不输出到前端，只用于内部处理
            else:
                state["is_complete"] = False
                state["error"] = result.get("response") or result.get("suggestion") or full_content
                state["next_step"] = "end"
                # 非完整请求，输出提示信息
                if callback and state["error"]:
                    callback(state["error"])
        
        except json.JSONDecodeError:
            # 不是 JSON，说明是自然语言回复，输出到前端
            state["is_complete"] = False
            state["error"] = full_content
            state["next_step"] = "end"
            if callback:
                callback(full_content)
        
        return state
        
    except Exception as e:
        logger.error(f"意图理解错误:\n{traceback.format_exc()}")
        error_msg = str(e)
        
        if "500" in error_msg or "InternalServerError" in error_msg:
            state["error"] = "抱歉，服务暂时不可用，请稍后重试。"
        else:
            state["error"] = "您好！我可以帮您查询用户信息。请提供您的11位手机号码。"
        
        state["is_complete"] = False
        state["next_step"] = "end"
        if callback:
            callback(state["error"])
        return state

def process_intent(state: AgentState) -> AgentState:
    """处理用户意图 - 非流式版本（用于工作流）"""
    try:
        # 构建对话历史字符串
        history = build_history_string(
            state.get("conversation_history", []),
            max_turns=MAX_HISTORY_DISPLAY_TURNS
        )
        
        response = chain_normal.invoke({
            "user_input": state["user_input"],
            "history": history
        })
        
        content = response.content
        if isinstance(content, list):
            content = content[0] if content else ""
            if hasattr(content, 'text'):
                content = content.text
        
        content_str = str(content).strip()
        
        try:
            result = json.loads(content_str)
            
            if result.get("complete"):
                state["is_complete"] = True
                state["user_info"] = result
                state["next_step"] = "query"
            else:
                state["is_complete"] = False
                state["error"] = result.get("response") or result.get("suggestion") or content_str
                state["next_step"] = "end"
        
        except json.JSONDecodeError:
            state["is_complete"] = False
            state["error"] = content_str
            state["next_step"] = "end"
        
        return state
        
    except Exception as e:
        logger.error(f"意图理解错误:\n{traceback.format_exc()}")
        error_msg = str(e)
        
        if "500" in error_msg or "InternalServerError" in error_msg:
            state["error"] = "抱歉，服务暂时不可用，请稍后重试。"
        else:
            state["error"] = "您好！我可以帮您查询用户信息。请提供您的11位手机号码。"
        
        state["is_complete"] = False
        state["next_step"] = "end"
        return state
