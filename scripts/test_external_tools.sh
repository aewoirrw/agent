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
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from app.core.settings import settings
from app.services.dashscope_client import DashScopeClient
from app.services.tools import AgentTools
from app.services.vector_store import VectorStore

PORT = 18081
HOST = '127.0.0.1'


class MockHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path.startswith('/api/runbooks/'):
            runbook_id = self.path.split('/')[-1]
            self._send_json(
                {
                    'ok': True,
                    'id': runbook_id,
                    'auth': self.headers.get('Authorization', ''),
                }
            )
            return
        self._send_json({'ok': False, 'message': 'not found'}, status=404)

    def do_POST(self):
        if self.path == '/api/form_submit':
            content_len = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(content_len).decode('utf-8') if content_len > 0 else ''
            form = {k: v[0] if v else '' for k, v in parse_qs(raw).items()}
            self._send_json(
                {
                    'ok': True,
                    'contentType': self.headers.get('Content-Type', ''),
                    'trace': self.headers.get('X-Trace-Id', ''),
                    'form': form,
                }
            )
            return
        self._send_json({'ok': False, 'message': 'not found'}, status=404)


def start_mock_server() -> HTTPServer:
    server = HTTPServer((HOST, PORT), MockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


async def run_checks():
    app_dir = Path.cwd()
    cfg = app_dir / 'external_tools.test.json'
    cfg.write_text(
        json.dumps(
            [
                {
                    'name': 'queryRunbookById',
                    'type': 'http',
                    'method': 'GET',
                    'url': f'http://{HOST}:{PORT}/api/runbooks/{{id}}',
                    'description': '按 runbook id 查询处理手册',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'token': {'type': 'string'},
                        },
                        'required': ['id'],
                    },
                    'headers': {
                        'Authorization': 'Bearer {token}',
                    },
                    'include_params_for_get': False,
                },
                {
                    'name': 'submitOpsAction',
                    'type': 'http',
                    'method': 'POST',
                    'url': f'http://{HOST}:{PORT}/api/form_submit',
                    'description': '提交运维动作（表单）',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'action': {'type': 'string'},
                            'target': {'type': 'string'},
                            'traceId': {'type': 'string'},
                        },
                        'required': ['action'],
                    },
                    'headers': {
                        'X-Trace-Id': '{traceId}',
                    },
                    'send_as': 'form',
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    settings.external_tools_config_path = str(cfg)

    dashscope = DashScopeClient()
    vector_store = VectorStore(dashscope)
    tools = AgentTools(vector_store)

    names = [x['function']['name'] for x in tools.definitions]
    assert 'queryRunbookById' in names, 'dynamic tool queryRunbookById not loaded'
    assert 'submitOpsAction' in names, 'dynamic tool submitOpsAction not loaded'

    runbook_raw = await tools.run('queryRunbookById', {'id': 'rb-1001', 'token': 'abc123'})
    runbook = json.loads(runbook_raw)
    assert runbook.get('success') is True, f'runbook tool failed: {runbook_raw}'
    body = runbook.get('body') or {}
    assert body.get('id') == 'rb-1001', f'wrong runbook id: {runbook_raw}'
    assert body.get('auth') == 'Bearer abc123', f'header placeholder not rendered: {runbook_raw}'

    post_raw = await tools.run('submitOpsAction', {'action': 'restart', 'target': 'payment', 'traceId': 'trace-001'})
    post = json.loads(post_raw)
    assert post.get('success') is True, f'post tool failed: {post_raw}'
    body2 = post.get('body') or {}
    assert body2.get('trace') == 'trace-001', f'header placeholder not rendered: {post_raw}'
    assert (body2.get('form') or {}).get('action') == 'restart', f'form body not sent: {post_raw}'

    alias_alerts_raw = await tools.run('query_prometheus_alerts', {})
    alias_alerts = json.loads(alias_alerts_raw)
    assert alias_alerts.get('success') is True, f'builtin alias query_prometheus_alerts failed: {alias_alerts_raw}'

    alias_topics_raw = await tools.run('get_available_log_topics', {})
    alias_topics = json.loads(alias_topics_raw)
    assert alias_topics.get('success') is True, f'builtin alias get_available_log_topics failed: {alias_topics_raw}'

    java_style_time_raw = await tools.run('DateTimeTools.getCurrentDateTime', {})
    assert isinstance(java_style_time_raw, str) and len(java_style_time_raw) > 10, f'java style tool name resolve failed: {java_style_time_raw}'

    java_style_logs_raw = await tools.run(
        'QueryLogsTools.queryLogs',
        {'region': 'ap-guangzhou', 'logTopic': 'application-logs', 'query': 'level:ERROR', 'limit': 2},
    )
    java_style_logs = json.loads(java_style_logs_raw)
    assert java_style_logs.get('success') is True, f'java style logs tool failed: {java_style_logs_raw}'

    cfg.unlink(missing_ok=True)


def main():
    server = start_mock_server()
    try:
        asyncio.run(run_checks())
        print('external_tools_test: PASS')
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
PY
