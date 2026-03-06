"""RAG Agent - 基于知识库的问答"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.config import get_stream_writer

from config import settings
from models.schemas import AgentState
from rag import get_vector_store
from utils.constants import LLM_TEMPERATURE
from utils.logger import logger
import traceback


RAG_SYSTEM_PROMPT = """你是一个专业的客服助手，负责根据提供的知识库内容回答用户问题。

回答要求：
1. 基于检索到的知识库内容回答，不要编造信息
2. 如果知识库中没有相关信息，诚实告知用户
3. 回答要简洁、准确、友好
4. 可以适当引用知识库来源，增强可信度
5. 如果用户问题不清楚，可以询问更多细节

知识库内容：
{context}

用户问题：{question}

请基于以上知识库内容回答用户问题。
"""

# 创建支持流式的 LLM
llm_stream = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=LLM_TEMPERATURE,
    streaming=True,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
])

chain = prompt | llm_stream | StrOutputParser()


def process_rag(state: AgentState) -> AgentState:
    """
    处理知识库问答
    
    工作流程：
    1. 从向量数据库检索相关文档
    2. 使用 LLM 基于检索内容生成答案
    3. 流式输出答案
    """
    try:
        logger.info("开始 RAG 问答")
        
        # 获取流式写入器
        try:
            writer = get_stream_writer()
            writer({"type": "status", "message": "正在检索知识库..."})
        except Exception:
            writer = None
        
        user_input = state.get("user_input", "")
        if not user_input:
            state["error"] = "缺少用户输入"
            state["next_step"] = "end"
            return state
        
        # 从向量数据库检索相关文档
        vector_store = get_vector_store()
        relevant_docs = vector_store.similarity_search_with_score(
            query=user_input,
            k=3  # 检索 Top-3 相关文档
        )
        
        if not relevant_docs:
            logger.warning("未检索到相关文档")
            error_msg = "抱歉，我在知识库中没有找到相关信息。您可以换个方式提问，或者咨询其他问题。"
            state["error"] = error_msg
            state["next_step"] = "end"
            if writer:
                writer({"type": "message", "content": error_msg})
            return state
        
        # 构建上下文
        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            category = doc.metadata.get("category", "未分类")
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(
                f"[文档{i} - {category}]\n{doc.page_content}\n"
            )
        
        context = "\n".join(context_parts)
        logger.debug(f"检索到 {len(relevant_docs)} 个相关文档，相似度分数: {[f'{s:.3f}' for _, s in relevant_docs]}")
        
        if writer:
            writer({"type": "status", "message": "正在生成答案..."})
        
        # 使用 LLM 生成答案（流式输出）
        response = chain.invoke({
            "context": context,
            "question": user_input
        })
        
        # 保存答案到状态
        state["rag_answer"] = response
        state["rag_sources"] = [
            {
                "category": doc.metadata.get("category", "未分类"),
                "source": doc.metadata.get("source", "未知"),
                "score": float(score)
            }
            for doc, score in relevant_docs
        ]
        state["next_step"] = "end"
        
        # 更新对话历史
        if "conversation_history" not in state:
            state["conversation_history"] = []
        state["conversation_history"].append(f"用户: {user_input}")
        state["conversation_history"].append(f"助手: {response}")
        
        logger.info("RAG 问答完成")
        return state
        
    except Exception:
        logger.error(f"RAG 问答失败:\n{traceback.format_exc()}")
        error_msg = "抱歉，处理您的问题时出现错误，请稍后重试。"
        state["error"] = error_msg
        state["next_step"] = "end"
        
        try:
            writer = get_stream_writer()
            writer({"type": "content", "content": "\n\n" + error_msg})
        except Exception:
            pass
        
        return state
