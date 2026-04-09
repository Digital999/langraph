"""FastAPI 主应用入口"""
import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from langgraph.types import Command

from graph.workflow import init_checkpointer, cleanup_checkpointer, get_app
from models.schemas import AgentResponse, AgentState, UserRequest
from rag import get_vector_store
from utils.constants import QUICK_RESPONSES
from utils.logger import logger
from utils.validators import sanitize_input


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期管理"""
    logger.info("应用启动中...")
    await init_checkpointer()
    logger.info("PostgreSQL Checkpointer 初始化完成")
    
    # 在 async 上下文中预初始化 Milvus，避免同步线程里缺少 event loop
    try:
        store = get_vector_store()
        _ = store.vector_store
        logger.info("Milvus 向量数据库连接初始化完成")
    except Exception as e:
        logger.warning(f"Milvus 预初始化失败（RAG 功能可能不可用）: {e}")
    
    yield
    
    # 关闭时清理资源
    logger.info("应用关闭中...")
    await cleanup_checkpointer()
    logger.info("资源清理完成")


app = FastAPI(
    title="AI 报告生成系统",
    description="基于 FastAPI + LangGraph 的多 Agent 协同报告生成系统",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """根路径 - 重定向到聊天界面"""
    return FileResponse("static/chat.html")

@app.get("/api")
async def api_info():
    """API 信息"""
    return {
        "message": "AI 报告生成系统",
        "version": "1.0.0",
        "endpoints": {
            "chat_ui": "/",
            "chat_stream": "/api/chat-stream",
            "generate_report": "/api/generate-report",
            "download_report": "/api/download/{filename}"
        }
    }

@app.post("/api/generate-report", response_model=AgentResponse)
async def generate_report(request: UserRequest):
    """
    生成报告接口
    
    Args:
        request: 用户请求，包含用户输入
        
    Returns:
        AgentResponse: 包含状态、消息和报告路径
    """
    logger.separator("=")
    logger.info(f"收到报告生成请求: {request.user_input}")
    
    try:
        # 创建初始状态（只传入必要字段）
        initial_state: AgentState = {
            "user_input": request.user_input,
            "is_complete": False
        }
        
        # 执行工作流
        try:
            result = get_app().invoke(initial_state)
        except Exception as workflow_error:
            logger.error(f"工作流执行错误:\n{traceback.format_exc()}")
            error_msg = str(workflow_error)
            
            # 检查是否是 API 错误
            if "500" in error_msg or "InternalServerError" in error_msg:
                return AgentResponse(
                    status="error",
                    message="AI 服务暂时不可用，请稍后重试",
                    data={"error_detail": "API 内部错误"}
                )
            elif "timeout" in error_msg.lower():
                return AgentResponse(
                    status="error",
                    message="请求超时，请稍后重试",
                    data={"error_detail": "请求超时"}
                )
            else:
                return AgentResponse(
                    status="error",
                    message=f"工作流执行失败: {error_msg}",
                    data=None
                )
        
        # 检查是否有错误
        if result.get("error"):
            logger.warning(f"请求处理失败: {result['error']}")
            return AgentResponse(
                status="error",
                message=result["error"],
                data=None
            )
        
        # 检查是否生成了报告
        if result.get("report_path"):
            filename = os.path.basename(result["report_path"])
            logger.info(f"✓ 报告生成成功: {filename}")
            return AgentResponse(
                status="success",
                message="报告生成成功",
                data={
                    "query_results": result.get("query_results"),
                    "report_filename": filename
                },
                report_url=f"/api/download/{filename}"
            )
        else:
            return AgentResponse(
                status="incomplete",
                message=result.get("error") or "请提供更多信息",
                data=result.get("user_info")
            )
    
    except Exception as e:
        logger.error(f"对话时发生错误:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.get("/api/download/{filename}")
async def download_report(filename: str):
    """
    下载报告文件
    
    Args:
        filename: 报告文件名
        
    Returns:
        FileResponse: Word 文档文件
    """
    filepath = f"reports/{filename}"
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="报告文件不存在")
    
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename
    )

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

@app.post("/api/chat-stream")
async def chat_stream(request: UserRequest):
    """
    流式对话接口 - 使用LangGraph原生流式支持
    
    Args:
        request: 用户请求,包含用户输入和会话ID
        
    Returns:
        StreamingResponse: Server-Sent Events (SSE) 流式响应
    """
    import uuid
    import json
    
    # 生成或使用会话ID
    thread_id = request.session_id or str(uuid.uuid4())
    logger.info(f"会话ID: {thread_id}")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """生成流式响应"""
        logger.separator("=")
        
        # 清理并验证用户输入
        user_input = sanitize_input(request.user_input)
        if not user_input:
            yield f"data: {json.dumps({'type': 'content', 'content': '请输入有效的查询内容'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'status': 'error'}, ensure_ascii=False)}\n\n"
            return
        
        logger.info(f"收到流式请求 [会话:{thread_id}]: {user_input}")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'session_id': thread_id}, ensure_ascii=False)}\n\n"
            
            # 检查是否是快速响应(缓存的问候语)
            user_input_lower = user_input.strip().lower()
            if user_input_lower in QUICK_RESPONSES:
                logger.info(f"✓ 使用缓存响应: {user_input_lower}")
                response_text = QUICK_RESPONSES[user_input_lower]
                for i in range(0, len(response_text), 5):
                    chunk = response_text[i:i+5]
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)
                
                elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
                yield f"data: {json.dumps({'type': 'end', 'status': 'complete', 'performance': {'total_ms': round(elapsed, 2)}}, ensure_ascii=False)}\n\n"
                logger.info(f"✓ 缓存响应完成,耗时: {elapsed:.0f}ms")
                return
            
            config = {"configurable": {"thread_id": thread_id}}
            graph_app = get_app()

            # 检测图是否处于 interrupt 中断状态（如等待报告确认）
            graph_state = await graph_app.aget_state(config)
            if graph_state.next:
                logger.info(f"检测到中断状态，恢复执行: pending={graph_state.next}")
                stream_input: AgentState | Command = Command(resume=user_input)
            else:
                stream_input = {
                    "user_input": user_input,
                    "is_complete": False,
                }

            async for event in graph_app.astream(
                stream_input,
                config=config,  # type: ignore
                stream_mode="custom",
            ):
                # 单一stream_mode时，event直接是数据字典
                if isinstance(event, dict):
                    chunk_type = event.get("type", "content")
                    
                    if chunk_type == "message":
                        # 完整消息（如：询问手机号）
                        yield f"data: {json.dumps({'type': 'content', 'content': event.get('content', '')}, ensure_ascii=False)}\n\n"
                    
                    elif chunk_type == "result":
                        # 查询结果
                        yield f"data: {json.dumps({'type': 'content', 'content': event.get('content', '')}, ensure_ascii=False)}\n\n"
                    
                    elif chunk_type == "intent_complete":
                        # 意图理解完成，开始查询（\n 不能写在 f-string 的 {} 表达式里的字面量中）
                        _thinking_msg = "正在思考，请稍后...\n"
                        yield f"data: {json.dumps({'type': 'content', 'content': _thinking_msg}, ensure_ascii=False)}\n\n"
                    
                    elif chunk_type == "status":
                        # 状态消息
                        yield f"data: {json.dumps({'type': 'status', 'message': event.get('message', '')}, ensure_ascii=False)}\n\n"
                    
                    elif chunk_type == "error":
                        # 错误消息
                        yield f"data: {json.dumps({'type': 'error', 'content': event.get('content', '')}, ensure_ascii=False)}\n\n"
                    
                    elif chunk_type == "report":
                        # 报告生成完成，推送报告信息（保持原始事件结构）
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    
                    else:
                        # 其他自定义事件
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            
            # 流式完成
            elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
            yield f"data: {json.dumps({'type': 'end', 'status': 'complete', 'performance': {'total_ms': round(elapsed, 2)}}, ensure_ascii=False)}\n\n"
            logger.info(f"✓ 对话完成,耗时: {elapsed:.0f}ms")
        
        except Exception:
            error_msg = "抱歉,处理请求时出错，请稍后重试。"
            logger.error(f"流式处理错误:\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'status': 'error'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )







