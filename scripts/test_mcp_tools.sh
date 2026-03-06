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
import os
import sys
from pathlib import Path

from app.core.settings import settings
from app.services.dashscope_client import DashScopeClient
from app.services.tools import AgentTools
from app.services.vector_store import VectorStore


def write_mock_server(path: Path):
    path.write_text(
        r'''import json
import sys

def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        text = line.decode("ascii", errors="ignore").strip()
        if ":" in text:
            k, v = text.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))

def send_message(payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()

while True:
    msg = read_message()
    if msg is None:
        break
    if "id" not in msg:
        continue

    method = msg.get("method")
    req_id = msg.get("id")

    if method == "initialize":
        send_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-mcp", "version": "0.1.0"}
            }
        })
    elif method == "tools/list":
        send_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "mcpEcho",
                        "description": "Echo text from MCP server",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"}
                            },
                            "required": ["text"]
                        }
                    }
                ]
            }
        })
    elif method == "tools/call":
        params = msg.get("params") or {}
        args = params.get("arguments") or {}
        text = str(args.get("text") or "")
        send_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"echo:{text}"}],
                "isError": False
            }
        })
    else:
        send_message({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })
''',
        encoding='utf-8',
    )


async def run_checks():
    app_dir = Path.cwd()
    mock_server = app_dir / 'scripts' / '_mcp_mock_server.py'
    cfg = app_dir / 'mcp_servers.test.json'

    write_mock_server(mock_server)

    cfg.write_text(
        json.dumps(
            [
                {
                    'name': 'mock',
                    'command': sys.executable,
                    'args': [str(mock_server)],
                    'cwd': str(app_dir),
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
    assert 'mcpEcho' in names, f'mcp tool mcpEcho not discovered: {names}'

    out1_raw = await tools.run('mcpEcho', {'text': 'hello'})
    out1 = json.loads(out1_raw)
    assert out1.get('success') is True, f'mcp call failed: {out1_raw}'
    assert 'echo:hello' in (out1.get('content') or ''), f'unexpected mcp output: {out1_raw}'

    out2_raw = await tools.run('mock.mcpEcho', {'text': 'world'})
    out2 = json.loads(out2_raw)
    assert out2.get('success') is True, f'mcp alias call failed: {out2_raw}'
    assert 'echo:world' in (out2.get('content') or ''), f'unexpected mcp alias output: {out2_raw}'

    await tools.close_runtime_tools()

    cfg.unlink(missing_ok=True)
    mock_server.unlink(missing_ok=True)


if __name__ == '__main__':
    asyncio.run(run_checks())
    print('mcp_tools_test: PASS')
PY
