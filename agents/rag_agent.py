"""RAG Agent - 基于知识库的问答（自适应检索 + 流式输出）"""
import traceback
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.config import get_stream_writer

from config import settings
from models.schemas import AgentState
from rag import get_vector_store
from utils.constants import (
    RAG_TOP_K,
    RAG_SCORE_THRESHOLD,
    RAG_LLM_TEMPERATURE,
    RAG_MAX_RETRIEVAL_RETRIES,
    MAX_HISTORY_DISPLAY_TURNS,
)
from utils.decorators import rewrite_query_with_history, reformulate_failed_query
from utils.logger import logger


RAG_SYSTEM_PROMPT = """\
你是一个专业的客服助手，负责根据提供的知识库内容回答用户问题。

回答要求：
1. 基于检索到的知识库内容回答，不要编造信息
2. 如果知识库中没有相关信息，诚实告知用户
3. 回答要简洁、准确、友好
4. 使用清晰的格式（适当分段、列表等）提升可读性
5. 如果用户问题涉及多个知识点，请分条说明

以下是从知识库检索到的相关内容：
{context}"""

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=RAG_LLM_TEMPERATURE,
    streaming=True,
)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])

answer_chain = answer_prompt | llm | StrOutputParser()


def _retrieve_with_retry(
    user_input: str,
    history: List[str],
    writer,
) -> Tuple[List[Tuple[Document, float]], str]:
    """自适应检索：失败后由 LLM 改写查询重试，最多 RAG_MAX_RETRIEVAL_RETRIES 轮。

    Returns:
        (检索结果列表, 最终命中的查询语句)
    """
    vector_store = get_vector_store()
    attempted_queries: List[str] = []

    # 第 1 轮：结合历史改写
    search_query = rewrite_query_with_history(
        query=user_input,
        history=history,
        max_turns=MAX_HISTORY_DISPLAY_TURNS,
    )

    for attempt in range(1, RAG_MAX_RETRIEVAL_RETRIES + 1):
        attempted_queries.append(search_query)
        logger.info(f"检索第{attempt}轮, query='{search_query}'")

        results = vector_store.similarity_search_with_score(
            query=search_query,
            k=RAG_TOP_K,
            score_threshold=RAG_SCORE_THRESHOLD,
        )

        if results:
            logger.info(
                f"第{attempt}轮检索命中 {len(results)} 条, "
                f"L2 距离: [{', '.join(f'{s:.3f}' for _, s in results)}]"
            )
            return results, search_query

        logger.warning(f"第{attempt}轮检索无结果")

        if attempt < RAG_MAX_RETRIEVAL_RETRIES:
            if writer:
                writer({
                    "type": "status",
                    "message": f"未找到相关内容，正在换一种方式检索（第{attempt+1}轮）...",
                })
            search_query = reformulate_failed_query(
                original_question=user_input,
                history=history,
                attempted_queries=attempted_queries,
                max_turns=MAX_HISTORY_DISPLAY_TURNS,
            )

    return [], search_query


def process_rag(state: AgentState) -> AgentState:
    """处理知识库问答（自适应检索 + 流式输出）"""
    try:
        logger.info("开始 RAG 问答")

        try:
            writer = get_stream_writer()
        except Exception:
            writer = None

        user_input = state.get("user_input", "")
        if not user_input:
            state["error"] = "缺少用户输入"
            state["next_step"] = "end"
            return state

        if writer:
            writer({"type": "status", "message": "正在检索知识库..."})

        # 自适应检索（最多重试 RAG_MAX_RETRIEVAL_RETRIES 轮）
        history = state.get("conversation_history", [])
        try:
            relevant_docs, final_query = _retrieve_with_retry(
                user_input=user_input,
                history=history,
                writer=writer,
            )
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            error_msg = "抱歉，知识库服务暂时不可用，请稍后重试。"
            state["error"] = error_msg
            state["next_step"] = "end"
            if writer:
                writer({"type": "message", "content": error_msg})
            return state

        if not relevant_docs:
            logger.warning(f"经过 {RAG_MAX_RETRIEVAL_RETRIES} 轮检索仍无结果")
            fallback_msg = (
                "抱歉，经过多轮检索仍未在知识库中找到与您问题相关的信息。\n"
                "您可以尝试换个方式提问，或者咨询以下类型的问题：\n"
                "• 产品介绍与套餐信息\n"
                "• 常见问题解答\n"
                "• 业务办理流程\n"
                "• 资费规则说明"
            )
            state["rag_answer"] = fallback_msg
            state["next_step"] = "end"
            if writer:
                writer({"type": "message", "content": fallback_msg})
            return state

        # 构建上下文
        context_parts = []
        for i, (doc, score) in enumerate(relevant_docs, 1):
            category = doc.metadata.get("category", "未分类")
            context_parts.append(f"[文档{i} - {category}]\n{doc.page_content}\n")
        context = "\n".join(context_parts)

        if writer:
            writer({"type": "status", "message": "正在生成答案..."})

        # 流式生成答案
        full_response = ""
        for chunk in answer_chain.stream({"context": context, "question": user_input}):
            full_response += chunk
            if writer:
                writer({"type": "message", "content": chunk})

        if not full_response.strip():
            logger.warning("流式输出为空，使用 invoke 重试")
            full_response = answer_chain.invoke({"context": context, "question": user_input})
            if writer and full_response:
                writer({"type": "message", "content": full_response})

        # 保存到状态
        state["rag_answer"] = full_response
        state["rag_sources"] = [
            {
                "category": doc.metadata.get("category", "未分类"),
                "source": doc.metadata.get("source", "未知"),
                "score": round(float(score), 4),
            }
            for doc, score in relevant_docs
        ]
        state["next_step"] = "end"

        if "conversation_history" not in state:
            state["conversation_history"] = []
        state["conversation_history"].append(f"用户: {user_input}")
        state["conversation_history"].append(f"助手: {full_response}")

        logger.info(f"RAG 问答完成，回答长度: {len(full_response)}")
        return state

    except Exception:
        logger.error(f"RAG 问答失败:\n{traceback.format_exc()}")
        error_msg = "抱歉，处理您的问题时出现错误，请稍后重试。"
        state["error"] = error_msg
        state["next_step"] = "end"

        try:
            writer = get_stream_writer()
            writer({"type": "message", "content": error_msg})
        except Exception:
            pass

        return state
