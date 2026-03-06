from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from app.core.settings import settings
from app.services.mcp_client import MCPToolBridge
from app.services.vector_store import VectorStore


class AgentTools:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self._builtin_definitions = self._build_builtin_definitions()
        self._builtin_handlers = {
            'getCurrentDateTime': self.get_current_datetime,
            'queryInternalDocs': self.query_internal_docs,
            'queryPrometheusAlerts': self.query_prometheus_alerts,
            'getAvailableLogTopics': self.get_available_log_topics,
            'queryLogs': self.query_logs,
        }
        self._builtin_aliases = {
            'get_current_datetime': 'getCurrentDateTime',
            'query_internal_docs': 'queryInternalDocs',
            'query_prometheus_alerts': 'queryPrometheusAlerts',
            'get_available_log_topics': 'getAvailableLogTopics',
            'query_logs': 'queryLogs',
            'querylogs': 'queryLogs',
            'querymetricsalerts': 'queryPrometheusAlerts',
        }
        self._external_tools = self._load_external_tools()
        self._mcp_bridge = MCPToolBridge(settings.mcp_servers_config_path, settings.project_root)

    @property
    def definitions(self) -> list[dict]:
        return self._builtin_definitions + [
            {
                'type': 'function',
                'function': {
                    'name': t['name'],
                    'description': t.get('description') or 'External dynamic tool',
                    'parameters': t.get('parameters') or {'type': 'object', 'properties': {}, 'required': []},
                },
            }
            for t in self._external_tools
        ] + self._mcp_bridge.definitions

    async def ensure_runtime_tools(self) -> None:
        if self._mcp_bridge.has_config:
            await self._mcp_bridge.ensure_discovered()

    async def close_runtime_tools(self) -> None:
        await self._mcp_bridge.close()

    @staticmethod
    def _build_builtin_definitions() -> list[dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'getCurrentDateTime',
                    'description': 'Get the current date and time in user timezone.',
                    'parameters': {'type': 'object', 'properties': {}, 'required': []},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'queryInternalDocs',
                    'description': 'Search internal documentation by semantic retrieval.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'query': {'type': 'string', 'description': 'search text'}},
                        'required': ['query'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'queryPrometheusAlerts',
                    'description': 'Query active alerts from prometheus.',
                    'parameters': {'type': 'object', 'properties': {}, 'required': []},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'getAvailableLogTopics',
                    'description': 'Get available log topics for log query.',
                    'parameters': {'type': 'object', 'properties': {}, 'required': []},
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'queryLogs',
                    'description': 'Query logs by region/topic/query/limit.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'region': {'type': 'string'},
                            'logTopic': {'type': 'string'},
                            'query': {'type': 'string'},
                            'limit': {'type': 'integer'},
                        },
                        'required': ['logTopic'],
                    },
                },
            },
        ]

    async def run(self, name: str, arguments: dict) -> str:
        arguments = arguments or {}
        raw_name = str(name or '').strip()
        canonical_name = self._canonical_tool_name(raw_name)
        if canonical_name in self._builtin_handlers:
            if canonical_name == 'queryInternalDocs':
                return await self.query_internal_docs(arguments.get('query', ''))
            if canonical_name == 'queryPrometheusAlerts':
                return await self.query_prometheus_alerts()
            if canonical_name == 'queryLogs':
                return self.query_logs(
                    region=arguments.get('region') or 'ap-guangzhou',
                    log_topic=arguments.get('logTopic') or 'application-logs',
                    query=arguments.get('query') or '',
                    limit=int(arguments.get('limit') or 20),
                )
            handler = self._builtin_handlers[canonical_name]
            return handler()

        external = self._get_external_tool(canonical_name) or self._get_external_tool(raw_name)
        if external is not None:
            return await self._run_external_tool(external, arguments)

        await self.ensure_runtime_tools()
        mcp_result = await self._mcp_bridge.call_tool(raw_name, arguments)
        if mcp_result is None and canonical_name != raw_name:
            mcp_result = await self._mcp_bridge.call_tool(canonical_name, arguments)
        if mcp_result is not None:
            return mcp_result

        return json.dumps({'success': False, 'message': f'Unknown tool: {raw_name}'}, ensure_ascii=False)

    def _load_external_tools(self) -> list[dict[str, Any]]:
        path_raw = (settings.external_tools_config_path or '').strip() if hasattr(settings, 'external_tools_config_path') else ''
        if not path_raw:
            return []

        cfg_path = Path(path_raw)
        if not cfg_path.is_absolute():
            cfg_path = settings.project_root / cfg_path

        if not cfg_path.exists():
            return []

        try:
            data = json.loads(cfg_path.read_text(encoding='utf-8'))
            if not isinstance(data, list):
                return []
            valid = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                if not item.get('name'):
                    continue
                if (item.get('type') or 'http') != 'http':
                    continue
                if not item.get('url'):
                    continue
                valid.append(item)
            return valid
        except Exception:
            return []

    def _get_external_tool(self, name: str) -> dict[str, Any] | None:
        for tool in self._external_tools:
            if tool.get('name') == name:
                return tool
        return None

    def _canonical_tool_name(self, name: str) -> str:
        raw = str(name or '').strip()
        if not raw:
            return ''

        if raw in self._builtin_handlers:
            return raw

        if '.' in raw:
            suffix = raw.split('.')[-1].strip()
            if suffix in self._builtin_handlers:
                return suffix
            raw = suffix

        normalized = raw.replace('-', '_').strip()
        lowered = normalized.lower()
        if lowered in self._builtin_aliases:
            return self._builtin_aliases[lowered]

        compact = ''.join(ch for ch in lowered if ch.isalnum())
        if compact in self._builtin_aliases:
            return self._builtin_aliases[compact]

        for external in self._external_tools:
            ext_name = str(external.get('name') or '')
            if ext_name == raw:
                return ext_name
            if ext_name.lower() == lowered:
                return ext_name

        return raw

    async def _run_external_tool(self, tool: dict[str, Any], arguments: dict) -> str:
        method = str(tool.get('method') or 'POST').upper()
        timeout = int(tool.get('timeout') or 15)
        headers_raw = tool.get('headers') or {}
        raw_url = str(tool.get('url') or '')
        send_as = str(tool.get('send_as') or 'json').lower()
        include_params_for_get = bool(tool.get('include_params_for_get', True))

        try:
            url = raw_url.format(**(arguments or {}))
        except Exception:
            url = raw_url

        headers = self._render_headers(headers_raw, arguments or {})

        payload = arguments or {}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == 'GET':
                    params = payload if include_params_for_get else None
                    resp = await client.get(url, params=params, headers=headers)
                else:
                    if send_as == 'form':
                        resp = await client.request(method, url, data=payload, headers=headers)
                    else:
                        resp = await client.request(method, url, json=payload, headers=headers)

                text = resp.text
                try:
                    body = resp.json()
                except Exception:
                    body = text

                return json.dumps(
                    {
                        'success': resp.status_code < 400,
                        'tool': tool.get('name'),
                        'statusCode': resp.status_code,
                        'body': body,
                    },
                    ensure_ascii=False,
                )
        except Exception as e:
            return json.dumps(
                {
                    'success': False,
                    'tool': tool.get('name'),
                    'message': 'external tool request failed',
                    'error': str(e),
                },
                ensure_ascii=False,
            )

    @staticmethod
    def _render_headers(headers: dict[str, Any], arguments: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in (headers or {}).items():
            if v is None:
                continue
            raw = str(v)
            try:
                out[str(k)] = raw.format(**arguments)
            except Exception:
                out[str(k)] = raw
        return out

    @staticmethod
    def get_current_datetime() -> str:
        return datetime.now().astimezone().isoformat()

    async def query_internal_docs(self, query: str) -> str:
        if not query.strip():
            return json.dumps({'status': 'error', 'message': 'query is empty'}, ensure_ascii=False)
        rows = await self.vector_store.search(query, settings.rag_top_k)
        payload = [
            {'id': r.id, 'content': r.content, 'score': r.score, 'metadata': r.metadata}
            for r in rows
        ]
        if not payload:
            return json.dumps({'status': 'no_results', 'message': 'No relevant documents found.'}, ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)

    async def query_prometheus_alerts(self) -> str:
        if settings.prometheus_mock_enabled:
            now = datetime.now()
            alerts = [
                {
                    'alert_name': 'HighCPUUsage',
                    'description': 'payment-service CPU 持续 > 80%，当前 92%',
                    'state': 'firing',
                    'active_at': (now - timedelta(minutes=25)).isoformat(),
                    'duration': '25m',
                },
                {
                    'alert_name': 'HighMemoryUsage',
                    'description': 'order-service 内存持续 > 85%，当前 91%',
                    'state': 'firing',
                    'active_at': (now - timedelta(minutes=15)).isoformat(),
                    'duration': '15m',
                },
            ]
            return json.dumps({'success': True, 'alerts': alerts, 'message': f'成功检索到 {len(alerts)} 个活动告警'}, ensure_ascii=False)

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                url = f"{settings.prometheus_base_url}/api/v1/alerts"
                resp = await client.get(url)
                resp.raise_for_status()
                obj = resp.json()

            if obj.get('status') != 'success':
                return json.dumps({'success': False, 'message': 'Prometheus 返回失败', 'error': obj.get('error')}, ensure_ascii=False)

            seen = set()
            out = []
            for a in obj.get('data', {}).get('alerts', []):
                name = (a.get('labels') or {}).get('alertname')
                if not name or name in seen:
                    continue
                seen.add(name)
                out.append(
                    {
                        'alert_name': name,
                        'description': (a.get('annotations') or {}).get('description', ''),
                        'state': a.get('state', ''),
                        'active_at': a.get('activeAt', ''),
                        'duration': 'unknown',
                    }
                )
            return json.dumps({'success': True, 'alerts': out, 'message': f'成功检索到 {len(out)} 个活动告警'}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({'success': False, 'message': '查询失败', 'error': str(e)}, ensure_ascii=False)

    @staticmethod
    def get_available_log_topics() -> str:
        payload = {
            'success': True,
            'topics': [
                {'topicName': 'system-metrics', 'description': '系统指标日志'},
                {'topicName': 'application-logs', 'description': '应用日志'},
                {'topicName': 'database-slow-query', 'description': '数据库慢查询日志'},
                {'topicName': 'system-events', 'description': '系统事件日志'},
            ],
            'availableRegions': ['ap-guangzhou', 'ap-shanghai', 'ap-beijing', 'ap-chengdu'],
            'defaultRegion': 'ap-guangzhou',
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def query_logs(region: str, log_topic: str, query: str, limit: int) -> str:
        limit = max(1, min(limit, 100))
        now = datetime.now()

        samples = []
        if log_topic == 'system-metrics':
            for i in range(min(limit, 5)):
                samples.append(
                    {
                        'timestamp': (now - timedelta(minutes=2 * i)).strftime('%Y-%m-%d %H:%M:%S'),
                        'level': 'WARN',
                        'service': 'payment-service',
                        'message': f'CPU使用率过高: {92 - i * 1.2:.1f}%',
                    }
                )
        else:
            for i in range(min(limit, 5)):
                samples.append(
                    {
                        'timestamp': (now - timedelta(minutes=3 * i)).strftime('%Y-%m-%d %H:%M:%S'),
                        'level': 'ERROR' if i % 2 == 0 else 'WARN',
                        'service': 'order-service',
                        'message': '下游调用超时，可能导致响应变慢',
                    }
                )

        return json.dumps(
            {
                'success': True,
                'region': region,
                'logTopic': log_topic,
                'query': query or 'DEFAULT_QUERY',
                'logs': samples,
                'total': len(samples),
                'message': f'成功查询到 {len(samples)} 条日志',
            },
            ensure_ascii=False,
        )
