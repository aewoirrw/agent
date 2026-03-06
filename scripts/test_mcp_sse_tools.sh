#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$APP_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

python - <<'PY'
import asyncio
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.core.settings import settings
from app.services.dashscope_client import DashScopeClient
from app.services.tools import AgentTools
from app.services.vector_store import VectorStore

HOST = '127.0.0.1'
PORT = 18092


class SseHub:
    def __init__(self):
        self._lock = threading.Lock()
        self._subs = []

    def subscribe(self):
        q = queue.Queue()
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs = [x for x in self._subs if x is not q]

    def publish(self, payload: dict):
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            q.put(payload)


HUB = SseHub()


class MockMcpSseHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, payload: dict, status: int = 202):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/sse':
            self._send_json({'ok': False, 'message': 'not found'}, status=404)
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        q = HUB.subscribe()
        try:
            endpoint_data = f'http://{HOST}:{PORT}/message'
            self.wfile.write(f'event: endpoint\ndata: {endpoint_data}\n\n'.encode('utf-8'))
            self.wfile.flush()

            while True:
                try:
                    payload = q.get(timeout=12)
                    line = json.dumps(payload, ensure_ascii=False)
                    self.wfile.write(f'event: message\ndata: {line}\n\n'.encode('utf-8'))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b': ping\n\n')
                    self.wfile.flush()
        except Exception:
            pass
        finally:
            HUB.unsubscribe(q)

    def do_POST(self):
        if self.path != '/message':
            self._send_json({'ok': False, 'message': 'not found'}, status=404)
            return

        content_len = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(content_len).decode('utf-8') if content_len > 0 else '{}'
        req = json.loads(raw)
        req_id = req.get('id')
        method = req.get('method')

        if req_id is None:
            self._send_json({'ok': True}, status=202)
            return

        if method == 'initialize':
            resp = {
                'jsonrpc': '2.0',
                'id': req_id,
                'result': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {'tools': {}},
                    'serverInfo': {'name': 'mock-mcp-sse', 'version': '0.1.0'},
                },
            }
        elif method == 'tools/list':
            resp = {
                'jsonrpc': '2.0',
                'id': req_id,
                'result': {
                    'tools': [
                        {
                            'name': 'sseEcho',
                            'description': 'Echo text from MCP SSE server',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {'text': {'type': 'string'}},
                                'required': ['text'],
                            },
                        }
                    ]
                },
            }
        elif method == 'tools/call':
            args = ((req.get('params') or {}).get('arguments') or {})
            text = str(args.get('text') or '')
            resp = {
                'jsonrpc': '2.0',
                'id': req_id,
                'result': {
                    'content': [{'type': 'text', 'text': f'sse-echo:{text}'}],
                    'isError': False,
                },
            }
        else:
            resp = {
                'jsonrpc': '2.0',
                'id': req_id,
                'error': {'code': -32601, 'message': f'Method not found: {method}'},
            }

        HUB.publish(resp)
        self._send_json({'accepted': True}, status=202)


def start_server():
    server = ThreadingHTTPServer((HOST, PORT), MockMcpSseHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


async def run_checks():
    app_dir = Path.cwd()
    cfg = app_dir / 'mcp_servers.sse.test.json'

    cfg.write_text(
        json.dumps(
            [
                {
                    'name': 'mock-sse',
                    'transport': 'sse',
                    'url': f'http://{HOST}:{PORT}',
                    'sse_endpoint': '/sse',
                    'timeout': 15,
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    settings.mcp_servers_config_path = str(cfg)

    dashscope = DashScopeClient()
    vector_store = VectorStore(dashscope)
    tools = AgentTools(vector_store)

    await tools.ensure_runtime_tools()

    names = [x['function']['name'] for x in tools.definitions]
    assert 'sseEcho' in names, f'mcp sse tool not discovered: {names}'

    out1_raw = await tools.run('sseEcho', {'text': 'hello'})
    out1 = json.loads(out1_raw)
    assert out1.get('success') is True, f'mcp sse call failed: {out1_raw}'
    assert 'sse-echo:hello' in (out1.get('content') or ''), f'unexpected mcp sse output: {out1_raw}'

    out2_raw = await tools.run('mock-sse.sseEcho', {'text': 'world'})
    out2 = json.loads(out2_raw)
    assert out2.get('success') is True, f'mcp sse alias call failed: {out2_raw}'
    assert 'sse-echo:world' in (out2.get('content') or ''), f'unexpected mcp sse alias output: {out2_raw}'

    await tools.close_runtime_tools()
    cfg.unlink(missing_ok=True)


if __name__ == '__main__':
    srv = start_server()
    try:
        asyncio.run(run_checks())
        print('mcp_sse_tools_test: PASS')
    finally:
        srv.shutdown()
PY
