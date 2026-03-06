from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app.core.settings import settings
from app.models.schemas import (
    ApiResponse,
    ChatRequest,
    ChatResponse,
    ClearRequest,
    FileUploadRes,
    SessionInfoResponse,
)
from app.services.aiops_service import AiOpsService
from app.services.dashscope_client import DashScopeClient
from app.services.session_store import SessionStore
from app.services.tools import AgentTools
from app.services.vector_store import VectorStore

app = FastAPI(title=settings.app_name)

session_store = SessionStore()
dashscope = DashScopeClient()
vector_store = VectorStore(dashscope)
aiops_service = AiOpsService()
agent_tools = AgentTools(vector_store)

def _api_success(data):
    return ApiResponse(code=200, message='success', data=data).model_dump()


def _sse_payload(payload: dict) -> dict:
    return {'event': 'message', 'data': json.dumps(payload, ensure_ascii=False)}


def _sse_message(msg_type: str, data, event_key: str | None = None) -> dict:
    payload = {'type': msg_type, 'data': data}
    if event_key:
        payload['eventKey'] = event_key
    return _sse_payload(payload)


@app.post('/api/chat')
async def chat(req: ChatRequest):
    question = (req.question or '').strip()
    if not question:
        return JSONResponse(_api_success(ChatResponse(success=False, errorMessage='问题内容不能为空').model_dump()))

    await agent_tools.ensure_runtime_tools()

    session = session_store.get_or_create(req.id)
    history = session.history()

    system_prompt = (
        '你是一个专业的智能助手。可调用工具：getCurrentDateTime、queryInternalDocs、'
        'queryPrometheusAlerts、getAvailableLogTopics、queryLogs。'
        '如果问题涉及时间、内部文档、告警或日志，请优先调用工具再回答。'
    )

    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(history)

    refs = await vector_store.search(question, settings.rag_top_k)
    if refs:
        context = '\n\n'.join([f"参考{i+1}: {x.content}" for i, x in enumerate(refs)])
        messages.append({'role': 'system', 'content': f'已检索到内部文档片段:\n{context}'})

    messages.append({'role': 'user', 'content': question})

    try:
        if dashscope.enabled:
            answer = await dashscope.chat_with_tools(messages, agent_tools)
        else:
            answer = await dashscope.chat(messages, stream=False)
        session.add_message(question, answer)
        return JSONResponse(_api_success(ChatResponse(success=True, answer=answer).model_dump()))
    except Exception as e:
        return JSONResponse(_api_success(ChatResponse(success=False, errorMessage=str(e)).model_dump()))


@app.post('/api/chat_stream')
async def chat_stream(req: ChatRequest):
    question = (req.question or '').strip()

    async def event_gen():
        if not question:
            yield _sse_message('error', '问题内容不能为空', 'assistant.error')
            return

        session = session_store.get_or_create(req.id)
        yield _sse_message(
            'tool_started',
            {'tool': 'prepareToolchain', 'arguments': {}, 'resultPreview': ''},
            'assistant.tool.started',
        )
        await agent_tools.ensure_runtime_tools()
        yield _sse_message(
            'tool_finished',
            {'tool': 'prepareToolchain', 'arguments': {}, 'resultPreview': '工具链初始化完成'},
            'assistant.tool.finished',
        )

        history = session.history()
        system_prompt = (
            '你是一个专业的智能助手。可调用工具：getCurrentDateTime、queryInternalDocs、'
            'queryPrometheusAlerts、getAvailableLogTopics、queryLogs。'
            '如果问题涉及时间、内部文档、告警或日志，请优先调用工具再回答。'
        )

        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(history)

        yield _sse_message(
            'tool_started',
            {'tool': 'prepareEmbeddingModel', 'arguments': {'topK': settings.rag_top_k}, 'resultPreview': ''},
            'assistant.tool.started',
        )
        yield _sse_message(
            'tool_started',
            {'tool': 'retrieveInternalDocs', 'arguments': {'query': question}, 'resultPreview': ''},
            'assistant.tool.started',
        )
        refs = await vector_store.search(question, settings.rag_top_k)
        yield _sse_message(
            'tool_finished',
            {
                'tool': 'retrieveInternalDocs',
                'arguments': {'query': question},
                'resultPreview': f'命中 {len(refs)} 条文档片段',
            },
            'assistant.tool.finished',
        )
        yield _sse_message(
            'tool_finished',
            {
                'tool': 'prepareEmbeddingModel',
                'arguments': {'topK': settings.rag_top_k},
                'resultPreview': 'embedding 检索准备完成',
            },
            'assistant.tool.finished',
        )

        if refs:
            context = '\n\n'.join([f"参考{i+1}: {x.content}" for i, x in enumerate(refs)])
            messages.append({'role': 'system', 'content': f'内部文档检索结果:\n{context}'})

        messages.append({'role': 'user', 'content': question})

        try:
            yield _sse_message(
                'tool_started',
                {'tool': 'invokeLLM', 'arguments': {'mode': 'chat_with_tools'}, 'resultPreview': ''},
                'assistant.tool.started',
            )
            if dashscope.enabled:
                full, tool_trace = await dashscope.chat_with_tools_trace(messages, agent_tools)
                for item in tool_trace:
                    evt = item.get('event') or 'tool_finished'
                    name = item.get('tool') or 'unknown_tool'
                    args_obj = item.get('arguments') or {}
                    args = json.dumps(args_obj, ensure_ascii=False)
                    preview = item.get('resultPreview') or ''
                    if evt == 'tool_started':
                        tool_msg = f"[TOOL-START] {name} args={args}"
                    else:
                        tool_msg = f"[TOOL-END] {name} args={args} result={preview}"
                    tool_event_key = 'assistant.tool.started' if evt == 'tool_started' else 'assistant.tool.finished'
                    yield _sse_message(evt, {'tool': name, 'arguments': args_obj, 'resultPreview': preview}, tool_event_key)
                    yield _sse_message('content', tool_msg, 'assistant.content.delta')
                for i in range(0, len(full), 40):
                    yield _sse_message('content', full[i:i + 40], 'assistant.content.delta')
            else:
                chunks = []
                stream = await dashscope.chat(messages, stream=True)
                async for delta in stream:
                    chunks.append(delta)
                    yield _sse_message('content', delta, 'assistant.content.delta')
                full = ''.join(chunks)

            yield _sse_message(
                'tool_finished',
                {'tool': 'invokeLLM', 'arguments': {'mode': 'chat_with_tools'}, 'resultPreview': f'输出长度 {len(full)}'},
                'assistant.tool.finished',
            )

            session.add_message(question, full)
            yield _sse_message('done', None, 'assistant.done')
        except Exception as e:
            yield _sse_message('error', str(e), 'assistant.error')

    return EventSourceResponse(event_gen())


@app.post('/api/ai_ops')
async def ai_ops():
    async def event_gen():
        try:
            await agent_tools.ensure_runtime_tools()
            yield _sse_message('content', '正在读取告警并拆解任务...\n', 'assistant.content.delta')
            alerts_raw = await agent_tools.query_prometheus_alerts()
            yield _sse_message('content', '已完成告警拉取，正在关联日志与知识库...\n', 'assistant.content.delta')
            logs_raw = agent_tools.query_logs('ap-guangzhou', 'application-logs', 'level:ERROR OR slow', 5)
            docs_raw = await agent_tools.query_internal_docs('cpu high usage memory high usage slow response')

            preface = (
                '以下是分析输入（真实查询或Mock）：\n'
                f'- alerts: {alerts_raw[:500]}\n'
                f'- logs: {logs_raw[:500]}\n'
                f'- docs: {docs_raw[:500]}\n\n'
            )
            yield _sse_message('content', preface, 'assistant.content.delta')

            yield _sse_message('content', '正在启动 Planner-Executor-Replanner 闭环...\n', 'assistant.content.delta')
            report, trace = await aiops_service.run_planner_executor_loop(
                dashscope_client=dashscope,
                tools=agent_tools,
                alerts_raw=alerts_raw,
                logs_raw=logs_raw,
                docs_raw=docs_raw,
            )

            for item in trace:
                decision = item.get('decision', 'UNKNOWN')
                step = item.get('step') or '无'
                tool_name = ((item.get('tool') or {}).get('name') or 'N/A')
                progress = f"[Round {item.get('round', '?')}] decision={decision}, step={step}, tool={tool_name}\n"
                yield _sse_message('planner_step', item, 'assistant.planner.step')
                yield _sse_message('content', progress, 'assistant.content.delta')

            yield _sse_message('content', '闭环执行完成，正在输出最终报告...\n', 'assistant.content.delta')

            for item in aiops_service.stream_text_chunks(report):
                yield {'event': 'message', 'data': item}
        except Exception as e:
            yield _sse_message('error', f'AI Ops 流程失败: {e}', 'assistant.error')

    return EventSourceResponse(event_gen())


@app.post('/api/chat/clear')
async def clear_chat(req: ClearRequest):
    if not req.id:
        return JSONResponse(ApiResponse(code=500, message='会话ID不能为空', data=None).model_dump())
    sess = session_store.get(req.id)
    if not sess:
        return JSONResponse(ApiResponse(code=500, message='会话不存在', data=None).model_dump())
    sess.clear()
    return JSONResponse(_api_success('会话历史已清空'))


@app.get('/api/chat/session/{session_id}')
async def get_session(session_id: str):
    sess = session_store.get(session_id)
    if not sess:
        return JSONResponse(ApiResponse(code=500, message='会话不存在', data=None).model_dump())
    data = SessionInfoResponse(
        sessionId=session_id,
        messagePairCount=sess.pair_count(),
        createTime=sess.create_time,
    )
    return JSONResponse(_api_success(data.model_dump()))


@app.post('/api/upload')
async def upload(file: Optional[UploadFile] = File(default=None)):
    if file is None:
        return JSONResponse(ApiResponse(code=400, message='文件不能为空', data=None).model_dump(), status_code=400)

    if not file.filename:
        return JSONResponse(ApiResponse(code=400, message='文件名不能为空', data=None).model_dump(), status_code=400)

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    allow = [x.strip().lower() for x in settings.allowed_extensions.split(',') if x.strip()]
    if ext not in allow:
        return JSONResponse(
            ApiResponse(code=400, message=f'不支持的文件格式，仅支持: {settings.allowed_extensions}', data=None).model_dump(),
            status_code=400,
        )

    try:
        upload_dir = Path(settings.upload_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / file.filename

        data = await file.read()
        target.write_bytes(data)

        try:
            await vector_store.index_file(str(target))
        except Exception:
            pass

        res = FileUploadRes(fileName=file.filename, filePath=str(target), fileSize=len(data))
        return JSONResponse(_api_success(res.model_dump()))
    except Exception as e:
        return JSONResponse(ApiResponse(code=500, message=f'文件上传失败: {e}', data=None).model_dump(), status_code=500)


@app.get('/milvus/health')
async def milvus_health():
    health = vector_store.health()
    if health.get('ok'):
        return JSONResponse({'message': 'ok', 'collections': health.get('collections', [])})
    return JSONResponse({'message': health.get('error', 'unavailable')}, status_code=503)


static_dir = settings.static_dir
if static_dir.exists():
    app.mount('/', StaticFiles(directory=str(static_dir), html=True), name='static')
