from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


class MCPProtocolError(Exception):
    pass


class MCPBaseSession:
    def __init__(self, name: str, timeout: int = 20) -> None:
        self.name = name
        self.timeout = max(5, int(timeout or 20))
        self._inited = False

    async def start(self) -> None:
        return None

    async def initialize(self) -> None:
        await self.start()
        if self._inited:
            return

        await self.request(
            'initialize',
            {
                'protocolVersion': '2024-11-05',
                'capabilities': {},
                'clientInfo': {'name': 'super-biz-agent-py', 'version': '0.1.0'},
            },
        )
        await self.notify('notifications/initialized', {})
        self._inited = True

    async def list_tools(self) -> List[dict]:
        await self.initialize()
        tools: List[dict] = []
        cursor = None
        while True:
            params = {'cursor': cursor} if cursor else {}
            result = await self.request('tools/list', params)
            page_tools = result.get('tools') or []
            if isinstance(page_tools, list):
                tools.extend([t for t in page_tools if isinstance(t, dict)])
            cursor = result.get('nextCursor') or result.get('next_cursor')
            if not cursor:
                break
        return tools

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        await self.initialize()
        return await self.request('tools/call', {'name': tool_name, 'arguments': arguments or {}})

    async def notify(self, method: str, params: Optional[dict] = None) -> None:
        raise NotImplementedError

    async def request(self, method: str, params: Optional[dict] = None) -> dict:
        raise NotImplementedError

    async def close(self) -> None:
        self._inited = False


class MCPServerSession(MCPBaseSession):
    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: int = 20,
    ) -> None:
        super().__init__(name=name, timeout=timeout)
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._seq = 0
        self._io_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._proc and self._proc.returncode is None:
            return

        proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=(None if not self.env else {**os.environ, **self.env}),
            cwd=self.cwd,
        )
        self._proc = proc
        self._inited = False

    async def notify(self, method: str, params: Optional[dict] = None) -> None:
        payload = {'jsonrpc': '2.0', 'method': method, 'params': params or {}}
        await self._write_message(payload)

    async def request(self, method: str, params: Optional[dict] = None) -> dict:
        async with self._io_lock:
            self._seq += 1
            req_id = self._seq
            payload = {'jsonrpc': '2.0', 'id': req_id, 'method': method, 'params': params or {}}
            await self._write_message(payload)

            while True:
                message = await self._read_message(timeout=self.timeout)
                if not isinstance(message, dict):
                    continue
                if message.get('id') != req_id:
                    continue
                if 'error' in message:
                    err = message.get('error') or {}
                    raise MCPProtocolError(str(err.get('message') or err))
                return message.get('result') or {}

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._inited = False
        if not proc:
            return

        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()

    async def _write_message(self, payload: dict) -> None:
        proc = self._proc
        if not proc or proc.stdin is None:
            raise MCPProtocolError(f'MCP server not running: {self.name}')

        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        header = f'Content-Length: {len(body)}\r\n\r\n'.encode('ascii')
        proc.stdin.write(header + body)
        await proc.stdin.drain()

    async def _read_message(self, timeout: int) -> dict:
        proc = self._proc
        if not proc or proc.stdout is None:
            raise MCPProtocolError(f'MCP server not running: {self.name}')

        async def _read() -> dict:
            headers = await _read_headers(proc.stdout)
            length = _parse_content_length(headers)
            body = await proc.stdout.readexactly(length)
            return json.loads(body.decode('utf-8'))

        return await asyncio.wait_for(_read(), timeout=timeout)


class MCPSseSession(MCPBaseSession):
    def __init__(
        self,
        name: str,
        url: str,
        sse_endpoint: str = '/sse',
        timeout: int = 20,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(name=name, timeout=timeout)
        self.url = str(url or '').rstrip('/')
        self.sse_endpoint = sse_endpoint or '/sse'
        self.headers = headers or {}

        self._seq = 0
        self._request_lock = asyncio.Lock()
        self._pending: Dict[int, asyncio.Future] = {}

        self._client: Optional[httpx.AsyncClient] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._running = False
        self._ready = asyncio.Event()
        self._message_url = _join_url(self.url, '/message')

    async def start(self) -> None:
        if self._listen_task and not self._listen_task.done():
            return

        self._running = True
        self._ready = asyncio.Event()
        self._client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)
        self._listen_task = asyncio.create_task(self._listen_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self.timeout)
        except Exception as e:
            raise MCPProtocolError(f'MCP SSE connection not ready: {self.name}') from e

    async def notify(self, method: str, params: Optional[dict] = None) -> None:
        await self.start()
        payload = {'jsonrpc': '2.0', 'method': method, 'params': params or {}}
        await self._post_message(payload)

    async def request(self, method: str, params: Optional[dict] = None) -> dict:
        await self.start()
        async with self._request_lock:
            self._seq += 1
            req_id = self._seq
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._pending[req_id] = future

            payload = {'jsonrpc': '2.0', 'id': req_id, 'method': method, 'params': params or {}}
            try:
                await self._post_message(payload)
                message = await asyncio.wait_for(future, timeout=self.timeout)
            finally:
                self._pending.pop(req_id, None)

            if not isinstance(message, dict):
                raise MCPProtocolError('invalid MCP SSE response payload')
            if 'error' in message:
                err = message.get('error') or {}
                raise MCPProtocolError(str(err.get('message') or err))
            return message.get('result') or {}

    async def close(self) -> None:
        self._running = False
        self._inited = False
        if self._listen_task:
            self._listen_task.cancel()
            with contextlib.suppress(Exception):
                await self._listen_task
            self._listen_task = None

        for _, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(MCPProtocolError(f'MCP SSE session closed: {self.name}'))
        self._pending.clear()

        if self._client:
            await self._client.aclose()
            self._client = None

    async def _post_message(self, payload: dict) -> None:
        client = self._client
        if client is None:
            raise MCPProtocolError(f'MCP SSE client not ready: {self.name}')

        resp = await client.post(self._message_url, json=payload)
        if resp.status_code >= 400:
            raise MCPProtocolError(f'MCP SSE post failed ({resp.status_code}): {resp.text[:240]}')

    async def _listen_loop(self) -> None:
        sse_url = _join_url(self.url, self.sse_endpoint)
        while self._running:
            try:
                client = self._client
                if client is None:
                    return
                async with client.stream('GET', sse_url, headers={'Accept': 'text/event-stream'}) as resp:
                    resp.raise_for_status()
                    self._ready.set()

                    event_name = 'message'
                    data_lines: List[str] = []

                    async for line in resp.aiter_lines():
                        if line is None:
                            continue
                        if line == '':
                            await self._handle_sse_event(event_name, '\n'.join(data_lines))
                            event_name = 'message'
                            data_lines = []
                            continue
                        if line.startswith(':'):
                            continue
                        if line.startswith('event:'):
                            event_name = line[6:].strip() or 'message'
                            continue
                        if line.startswith('data:'):
                            data_lines.append(line[5:].lstrip())

            except asyncio.CancelledError:
                return
            except Exception:
                if not self._ready.is_set():
                    self._ready.set()
                await asyncio.sleep(0.6)

    async def _handle_sse_event(self, event_name: str, data: str) -> None:
        if not data:
            return

        if event_name == 'endpoint':
            endpoint = data.strip()
            if endpoint:
                self._message_url = _join_url(self.url, endpoint)
            return

        try:
            message = json.loads(data)
        except Exception:
            return

        if not isinstance(message, dict):
            return
        req_id = message.get('id')
        if req_id is None:
            return

        fut = self._pending.get(req_id)
        if fut is not None and not fut.done():
            fut.set_result(message)


class MCPToolBridge:
    def __init__(self, config_path: str, project_root: Path) -> None:
        self.config_path = (config_path or '').strip()
        self.project_root = project_root

        self._sessions: Dict[str, MCPBaseSession] = {}
        self._tools: List[dict] = []
        self._tool_lookup: Dict[str, dict] = {}
        self._discovered = False
        self._discover_lock = asyncio.Lock()

    @property
    def has_config(self) -> bool:
        return bool(self._resolve_config_path())

    @property
    def definitions(self) -> List[dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool['exposed_name'],
                    'description': tool.get('description') or 'MCP tool',
                    'parameters': tool.get('parameters') or {'type': 'object', 'properties': {}, 'required': []},
                },
            }
            for tool in self._tools
        ]

    async def ensure_discovered(self) -> None:
        if self._discovered:
            return

        async with self._discover_lock:
            if self._discovered:
                return

            self._tools = []
            self._tool_lookup = {}
            for server_cfg in self._load_configs():
                transport = str(server_cfg.get('transport') or 'stdio').lower()
                if transport == 'sse':
                    session = MCPSseSession(
                        name=server_cfg['name'],
                        url=server_cfg.get('url') or '',
                        sse_endpoint=server_cfg.get('sse_endpoint') or '/sse',
                        timeout=int(server_cfg.get('timeout') or 20),
                        headers=server_cfg.get('headers') or {},
                    )
                else:
                    session = MCPServerSession(
                        name=server_cfg['name'],
                        command=server_cfg['command'],
                        args=server_cfg.get('args') or [],
                        env=server_cfg.get('env') or {},
                        cwd=server_cfg.get('cwd'),
                        timeout=int(server_cfg.get('timeout') or 20),
                    )
                self._sessions[server_cfg['name']] = session

                try:
                    items = await session.list_tools()
                except Exception:
                    continue

                for item in items:
                    mcp_name = str(item.get('name') or '').strip()
                    if not mcp_name:
                        continue
                    exposed = self._dedupe_exposed_name(mcp_name, server_cfg['name'])
                    record = {
                        'server': server_cfg['name'],
                        'mcp_name': mcp_name,
                        'exposed_name': exposed,
                        'description': item.get('description') or '',
                        'parameters': item.get('inputSchema') or {'type': 'object', 'properties': {}, 'required': []},
                    }
                    self._tools.append(record)
                    self._index_tool(record)

            self._discovered = True

    async def call_tool(self, name: str, arguments: dict) -> Optional[str]:
        await self.ensure_discovered()
        record = self._resolve_record(name)
        if not record:
            return None

        session = self._sessions.get(record['server'])
        if session is None:
            return json.dumps({'success': False, 'message': f'MCP server not found: {record["server"]}'}, ensure_ascii=False)

        try:
            result = await session.call_tool(record['mcp_name'], arguments or {})
            content_text = _extract_content_text(result.get('content'))
            payload = {
                'success': not bool(result.get('isError')),
                'source': 'mcp',
                'server': record['server'],
                'tool': record['mcp_name'],
                'content': content_text,
                'structuredContent': result.get('structuredContent'),
                'isError': bool(result.get('isError')),
            }
            return json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            return json.dumps(
                {
                    'success': False,
                    'source': 'mcp',
                    'server': record['server'],
                    'tool': record['mcp_name'],
                    'message': 'mcp tool call failed',
                    'error': str(e),
                },
                ensure_ascii=False,
            )

    async def close(self) -> None:
        for _, session in list(self._sessions.items()):
            await session.close()
        self._sessions.clear()
        self._tools = []
        self._tool_lookup = {}
        self._discovered = False

    def _resolve_record(self, name: str) -> Optional[dict]:
        raw = str(name or '').strip()
        if not raw:
            return None
        lowered = raw.lower()
        return self._tool_lookup.get(raw) or self._tool_lookup.get(lowered)

    def _index_tool(self, record: dict) -> None:
        exposed = record['exposed_name']
        server = record['server']
        mcp_name = record['mcp_name']

        keys = {
            exposed,
            exposed.lower(),
            mcp_name,
            mcp_name.lower(),
            f'{server}.{mcp_name}',
            f'{server}.{mcp_name}'.lower(),
            f'{server}__{mcp_name}',
            f'{server}__{mcp_name}'.lower(),
        }
        for k in keys:
            self._tool_lookup[k] = record

    def _dedupe_exposed_name(self, mcp_name: str, server_name: str) -> str:
        used = {x['exposed_name'] for x in self._tools}
        if mcp_name not in used:
            return mcp_name
        candidate = f'{server_name}__{mcp_name}'
        if candidate not in used:
            return candidate
        idx = 2
        while f'{candidate}_{idx}' in used:
            idx += 1
        return f'{candidate}_{idx}'

    def _resolve_config_path(self) -> Optional[Path]:
        if not self.config_path:
            return None
        p = Path(self.config_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def _load_configs(self) -> List[dict]:
        cfg_path = self._resolve_config_path()
        if cfg_path is None or not cfg_path.exists():
            return []

        try:
            data = json.loads(cfg_path.read_text(encoding='utf-8'))
        except Exception:
            return []

        if not isinstance(data, list):
            return []

        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name') or '').strip()
            transport = str(item.get('transport') or '').strip().lower()
            command = str(item.get('command') or '').strip()
            url = str(item.get('url') or '').strip()

            if not name:
                continue

            if not transport:
                transport = 'stdio' if command else 'sse'

            if transport == 'sse':
                if not url:
                    continue
                cfg = {
                    'name': name,
                    'transport': 'sse',
                    'url': url,
                    'sse_endpoint': item.get('sse_endpoint') or '/sse',
                    'headers': item.get('headers') or {},
                    'timeout': item.get('timeout') or 20,
                }
            else:
                if not command:
                    continue
                cfg = {
                    'name': name,
                    'transport': 'stdio',
                    'command': command,
                    'args': item.get('args') or [],
                    'env': item.get('env') or {},
                    'cwd': item.get('cwd'),
                    'timeout': item.get('timeout') or 20,
                }
                cwd = cfg.get('cwd')
                if isinstance(cwd, str) and cwd:
                    cwd_path = Path(cwd)
                    if not cwd_path.is_absolute():
                        cfg['cwd'] = str((self.project_root / cwd).resolve())
            out.append(cfg)
        return out


def _extract_content_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    parts.append(str(item.get('text') or ''))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return '\n'.join([x for x in parts if x])
    if isinstance(content, dict):
        if content.get('type') == 'text':
            return str(content.get('text') or '')
        return json.dumps(content, ensure_ascii=False)
    return str(content)


async def _read_headers(reader: asyncio.StreamReader) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line:
            raise MCPProtocolError('unexpected EOF while reading MCP headers')
        if line in (b'\r\n', b'\n'):
            break
        text = line.decode('ascii', errors='ignore').strip()
        if ':' not in text:
            continue
        k, v = text.split(':', 1)
        headers[k.strip().lower()] = v.strip()
    return headers


def _parse_content_length(headers: Dict[str, str]) -> int:
    raw = headers.get('content-length')
    if raw is None:
        raise MCPProtocolError('missing Content-Length header')
    try:
        length = int(raw)
    except Exception as e:
        raise MCPProtocolError('invalid Content-Length header') from e
    if length <= 0:
        raise MCPProtocolError('invalid Content-Length value')
    return length


def _join_url(base: str, path_or_url: str) -> str:
    target = str(path_or_url or '').strip()
    if target.startswith('http://') or target.startswith('https://'):
        return target
    base_norm = str(base or '').rstrip('/')
    if not target:
        return base_norm
    if not target.startswith('/'):
        target = '/' + target
    return base_norm + target
