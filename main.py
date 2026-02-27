"""FastAPI 主应用入口"""
import os
import traceback
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models.schemas import UserRequest, AgentResponse, AgentState
from graph import app as workflow_app
from utils.logger import logger
from utils.constants import (
    QUICK_RESPONSES, CONFIRM_KEYWORDS, REJECT_KEYWORDS,
    STREAM_CHUNK_SIZE, STREAM_DELAY, QUEUE_CHECK_INTERVAL,
    MAX_CONVERSATION_TURNS
)
from utils.decorators import update_conversation_history
from utils.validators import sanitize_input

app = FastAPI(
    title="AI 报告生成系统",
    description="基于 FastAPI + LangGraph 的多 Agent 协同报告生成系统",
    version="1.0.0"
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
        # 创建初始状态
        initial_state: AgentState = {
            "user_input": request.user_input,
            "is_complete": False,
            "user_info": None,
            "query_results": None,
            "report_path": None,
            "error": None,
            "conversation_history": [],
            "next_step": "intent"
        }
        
        # 执行工作流
        try:
            result = workflow_app.invoke(initial_state)
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
    真正的流式对话接口 - 实时输出 LLM 响应，支持会话状态
    
    Args:
        request: 用户请求，包含用户输入和会话ID
        
    Returns:
        StreamingResponse: Server-Sent Events (SSE) 流式响应
    """
    from agents.intent_agent_stream import process_intent_stream
    import uuid
    
    # 生成或使用会话ID（作为 thread_id）
    thread_id = request.session_id or str(uuid.uuid4())
    
    logger.info(f"会话ID (thread_id): {thread_id}")
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """生成流式响应"""
        logger.separator("=")
        
        # 清理用户输入
        user_input = sanitize_input(request.user_input)
        if not user_input:
            yield f"data: {json.dumps({'type': 'content', 'content': '请输入有效的查询内容'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'status': 'error'}, ensure_ascii=False)}\n\n"
            return
        
        logger.info(f"收到流式请求 [会话:{thread_id}]: {user_input}")
        
        total_start_time = asyncio.get_event_loop().time()
        
        try:
            # 立即发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'session_id': thread_id}, ensure_ascii=False)}\n\n"
            
            # 检查是否是常见问候（快速响应）
            user_input_lower = user_input.strip().lower()
            if user_input_lower in QUICK_RESPONSES:
                logger.info(f"✓ 使用缓存响应: {user_input_lower}")
                response_text = QUICK_RESPONSES[user_input_lower]
                # 模拟流式输出
                for i in range(0, len(response_text), STREAM_CHUNK_SIZE):
                    chunk = response_text[i:i+STREAM_CHUNK_SIZE]
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(STREAM_DELAY)
                
                total_elapsed = (asyncio.get_event_loop().time() - total_start_time) * 1000
                yield f"data: {json.dumps({'type': 'end', 'status': 'complete', 'performance': {'total_ms': total_elapsed}}, ensure_ascii=False)}\n\n"
                logger.info(f"✓ 缓存响应完成，总耗时: {total_elapsed:.0f}ms")
                return
            
            # 创建 LangGraph 配置
            config = {"configurable": {"thread_id": thread_id}}
            
            # 创建初始输入
            initial_input = {"user_input": user_input}
            
            # 获取事件循环
            loop = asyncio.get_event_loop()
            
            # 检查是否在等待报告确认
            # 先获取当前会话状态
            try:
                current_state = workflow_app.get_state(config)
                if current_state and current_state.values.get("waiting_for_report_confirmation"):
                    logger.info("检测到报告确认请求")
                    # 用户正在回复是否生成报告
                    user_input_lower = user_input.lower()
                    
                    # 检查是否确认生成报告
                    is_confirm = any(keyword in user_input_lower for keyword in CONFIRM_KEYWORDS)
                    is_reject = any(keyword in user_input_lower for keyword in REJECT_KEYWORDS)
                    
                    logger.debug(f"用户输入: {user_input_lower}, 确认={is_confirm}, 拒绝={is_reject}")
                    
                    if is_reject:
                        # 用户拒绝生成报告，清除等待标志
                        yield f"data: {json.dumps({'type': 'content', 'content': '好的，如果以后需要生成报告，请告诉我。'}, ensure_ascii=False)}\n\n"
                        
                        # 更新状态，清除 waiting_for_report_confirmation 标志
                        await loop.run_in_executor(
                            None,
                            lambda: workflow_app.update_state(
                                config,
                                {"waiting_for_report_confirmation": False}
                            )
                        )
                        
                        total_elapsed = (asyncio.get_event_loop().time() - total_start_time) * 1000
                        yield f"data: {json.dumps({'type': 'end', 'status': 'complete', 'performance': {'total_ms': total_elapsed}}, ensure_ascii=False)}\n\n"
                        logger.info(f"✓ 用户拒绝生成报告，总耗时: {total_elapsed:.0f}ms")
                        return
                    elif is_confirm:
                        # 用户确认生成报告
                        yield f"data: {json.dumps({'type': 'content', 'content': '正在生成报告...'}, ensure_ascii=False)}\n\n"
                        
                        # 调用工作流生成报告
                        final_result = await loop.run_in_executor(
                            None, 
                            lambda: workflow_app.invoke(initial_input, config)
                        )
                        
                        # 检查报告是否生成成功
                        if final_result.get("report_path"):
                            filename = os.path.basename(final_result["report_path"])
                            report_url = f"/api/download/{filename}"
                            success_msg = f"\n\n✓ 报告生成成功！\n\n下载链接：{report_url}"
                            yield f"data: {json.dumps({'type': 'content', 'content': success_msg}, ensure_ascii=False)}\n\n"
                            yield f"data: {json.dumps({'type': 'report', 'filename': filename, 'url': report_url}, ensure_ascii=False)}\n\n"
                        else:
                            error_msg = final_result.get("error", "报告生成失败")
                            yield f"data: {json.dumps({'type': 'content', 'content': f'\n\n{error_msg}'}, ensure_ascii=False)}\n\n"
                        
                        total_elapsed = (asyncio.get_event_loop().time() - total_start_time) * 1000
                        yield f"data: {json.dumps({'type': 'end', 'status': 'complete', 'performance': {'total_ms': total_elapsed}}, ensure_ascii=False)}\n\n"
                        logger.info(f"✓ 报告生成完成，总耗时: {total_elapsed:.0f}ms")
                        return
                    else:
                        # 用户输入不明确，继续询问
                        yield f"data: {json.dumps({'type': 'content', 'content': '请明确回复"是"或"不需要"确认是否需要生成报告。'}, ensure_ascii=False)}\n\n"
                        total_elapsed = (asyncio.get_event_loop().time() - total_start_time) * 1000
                        yield f"data: {json.dumps({'type': 'end', 'status': 'complete', 'performance': {'total_ms': total_elapsed}}, ensure_ascii=False)}\n\n"
                        return
            except Exception as e:
                logger.warning(f"获取会话状态失败: {e}")
            
            # 使用队列在线程间传递流式内容
            import queue
            content_queue = queue.Queue()
            
            def sync_callback(content: str):
                content_queue.put(('content', content))
            
            # 获取对话历史
            conversation_history = []
            try:
                current_state = workflow_app.get_state(config)
                if current_state and current_state.values:
                    conversation_history = current_state.values.get("conversation_history", [])
                    logger.debug(f"加载对话历史: {len(conversation_history)} 条")
            except Exception as e:
                logger.warning(f"获取对话历史失败: {e}")
            
            # 创建临时状态用于流式输出意图理解
            temp_state: AgentState = {
                "user_input": user_input,
                "is_complete": False,
                "conversation_history": conversation_history,
            }
            
            # 启动后台任务
            future = loop.run_in_executor(
                None, 
                lambda: process_intent_stream(temp_state, sync_callback)
            )
            
            # 实时从队列中读取并输出
            result_state = None
            while True:
                try:
                    # 非阻塞获取，快速检查
                    msg_type, content = content_queue.get_nowait()
                    if msg_type == 'content':
                        yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    # 队列为空，检查任务是否完成
                    if future.done():
                        result_state = future.result()
                        break
                    # 短暂等待，避免CPU空转
                    await asyncio.sleep(QUEUE_CHECK_INTERVAL)
            
            # 输出队列中剩余的内容
            while not content_queue.empty():
                msg_type, content = content_queue.get_nowait()
                if msg_type == 'content':
                    yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
            
            # 更新对话历史到 LangGraph 状态
            if result_state:
                try:
                    # 获取当前状态
                    current_state = workflow_app.get_state(config)
                    if current_state:
                        # 更新对话历史
                        conversation_history = current_state.values.get("conversation_history", [])
                        assistant_response = result_state.get("error", "")
                        conversation_history = update_conversation_history(
                            conversation_history,
                            user_input,
                            assistant_response,
                            max_turns=MAX_CONVERSATION_TURNS
                        )
                        
                        # 更新状态
                        await loop.run_in_executor(
                            None,
                            lambda: workflow_app.update_state(
                                config,
                                {"conversation_history": conversation_history}
                            )
                        )
                        logger.debug(f"对话历史已更新: {len(conversation_history)} 条")
                except Exception as e:
                    logger.warning(f"更新对话历史失败: {e}")
            
            # 检查是否需要继续查询
            if result_state and result_state["is_complete"]:
                # 需要查询数据库
                yield f"data: {json.dumps({'type': 'content', 'content': '正在查询数据...'}, ensure_ascii=False)}\n\n"
                
                # 使用 LangGraph 的 invoke 方法，传入 config 实现会话管理
                final_result = await loop.run_in_executor(
                    None, 
                    lambda: workflow_app.invoke(initial_input, config)
                )
                
                # 计算总耗时
                total_elapsed = (asyncio.get_event_loop().time() - total_start_time) * 1000
                perf_data = final_result.get("performance_metrics", {})
                perf_data["total_ms"] = total_elapsed
                
                # 输出性能统计
                yield f"data: {json.dumps({'type': 'performance', 'metrics': perf_data}, ensure_ascii=False)}\n\n"
                logger.info(f"✓ 性能统计: {json.dumps(perf_data, ensure_ascii=False)}")
                
                # 输出查询结果（存储在 error 字段中）
                if final_result.get("error"):
                    result_content = final_result.get("error", "")
                    # 以 content 类型输出查询结果
                    yield f"data: {json.dumps({'type': 'content', 'content': '\n\n' + result_content}, ensure_ascii=False)}\n\n"
                    
                    # 更新对话历史，添加查询结果
                    try:
                        current_state = workflow_app.get_state(config)
                        if current_state:
                            conversation_history = current_state.values.get("conversation_history", [])
                            # 添加查询结果到历史（简化版本，避免过长）
                            conversation_history = update_conversation_history(
                                conversation_history,
                                user_input,
                                "查询完成",
                                max_turns=MAX_CONVERSATION_TURNS
                            )
                            
                            await loop.run_in_executor(
                                None,
                                lambda: workflow_app.update_state(
                                    config,
                                    {"conversation_history": conversation_history}
                                )
                            )
                    except Exception as e:
                        logger.warning(f"更新对话历史失败: {e}")
                    
                    yield f"data: {json.dumps({'type': 'end', 'status': 'complete'}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'content', 'content': '\\n\\n查询完成，但未返回结果。'}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'end', 'status': 'error'}, ensure_ascii=False)}\n\n"
            else:
                # 对话或提示，不需要查询
                total_elapsed = (asyncio.get_event_loop().time() - total_start_time) * 1000
                yield f"data: {json.dumps({'type': 'performance', 'metrics': {'total_ms': total_elapsed}}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'end', 'status': 'complete'}, ensure_ascii=False)}\n\n"
                logger.info(f"✓ 对话完成，总耗时: {total_elapsed:.0f}ms")
        
        except Exception as e:
            error_msg = f"\\n\\n抱歉，处理请求时出错: {str(e)}"
            logger.error(f"流式处理错误:\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'content', 'content': error_msg}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'status': 'error'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"  # 确保分块传输
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)






