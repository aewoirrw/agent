from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional

from app.core.settings import settings


class AiOpsService:
    def __init__(self) -> None:
        self.docs_dir = settings.docs_dir

    def build_report(self) -> str:
        now = datetime.now()
        alert_rows = [
            ('HighCPUUsage', 'warning', 'payment-service', now - timedelta(minutes=25), now, '活跃'),
            ('HighMemoryUsage', 'warning', 'order-service', now - timedelta(minutes=15), now, '活跃'),
        ]

        key_findings = self._summarize_docs()
        table = '\n'.join(
            f"| {a} | {lvl} | {svc} | {s:%Y-%m-%d %H:%M:%S} | {e:%Y-%m-%d %H:%M:%S} | {st} |"
            for a, lvl, svc, s, e, st in alert_rows
        )

        return f"""# 告警分析报告

---

## 📋 活跃告警清单

| 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
|---------|------|----------|-------------|-------------|------|
{table}

---

## 🔍 告警根因分析1 - HighCPUUsage

### 告警详情
- **告警级别**: warning
- **受影响服务**: payment-service
- **持续时间**: 25分钟

### 症状描述
服务 CPU 长时间高位，可能存在热点请求或异常循环。

### 日志证据
近期日志显示业务峰值期间 CPU 接近饱和。

### 根因结论
高并发叠加部分慢逻辑，导致 CPU 使用率持续升高。

---

## 🛠️ 处理方案执行1 - HighCPUUsage

### 已执行的排查步骤
1. 核对近 30 分钟接口流量与慢请求分布
2. 检查主机 CPU 与进程线程趋势

### 处理建议
增加实例副本，优化热点接口，增加限流策略。

### 预期效果
CPU 使用率回落到 70% 以下。

---

## 📊 结论

### 整体评估
当前存在中等风险，建议优先处理 CPU 与内存告警。

### 关键发现
{key_findings}

### 后续建议
1. 建立告警自动关联 Runbook
2. 增加容量预测与压测基线

### 风险评估
风险等级：中。
"""

    def _summarize_docs(self) -> str:
        bullets = []
        if self.docs_dir.exists():
            for p in sorted(self.docs_dir.glob('*.md'))[:3]:
                first = p.read_text(encoding='utf-8').splitlines()[:1]
                hint = first[0] if first else p.stem
                bullets.append(f"- {p.stem}: {hint}")
        if not bullets:
            bullets.append('- 暂无本地运维文档摘要')
        return '\n'.join(bullets)

    def stream_report_chunks(self):
        report = self.build_report()
        for i in range(0, len(report), 80):
            yield json.dumps({'type': 'content', 'data': report[i:i + 80]}, ensure_ascii=False)
        yield json.dumps({'type': 'done', 'data': None}, ensure_ascii=False)

    async def build_report_with_llm(
        self,
        dashscope_client,
        alerts_raw: str,
        logs_raw: str,
        docs_raw: str,
        planner_feedback: Optional[str] = None,
    ) -> str:
        system_prompt = (
            '你是企业级 AIOps 分析专家。请严格基于输入证据生成告警分析报告，禁止编造。'
            '输出必须是 Markdown，结构包含：活跃告警清单、根因分析、处理方案、结论。'
            '若证据不足，必须在结论中明确写出“无法完成”的原因。'
        )
        user_prompt = (
            '请基于以下真实输入生成《告警分析报告》：\n\n'
            f'[alerts]\n{alerts_raw}\n\n'
            f'[logs]\n{logs_raw}\n\n'
            f'[docs]\n{docs_raw}\n\n'
            f'[executor_feedback]\n{planner_feedback or "N/A"}\n\n'
            '要求：\n'
            '1) 先输出“## 📋 活跃告警清单”表格；\n'
            '2) 每个主要告警至少给出一段“根因分析 + 处理建议”；\n'
            '3) 最后输出“## 📊 结论”并给出风险等级。\n'
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

        if dashscope_client.enabled:
            llm_text = await dashscope_client.chat(messages, stream=False)
            if '已切换为本地降级回复' not in llm_text:
                return llm_text

            fallback = self.build_report()
            return f"{llm_text}\n\n---\n\n{fallback}"

        return self.build_report()

    def stream_text_chunks(self, text: str, chunk_size: int = 80):
        for i in range(0, len(text), chunk_size):
            yield json.dumps({'type': 'content', 'data': text[i:i + chunk_size]}, ensure_ascii=False)
        yield json.dumps({'type': 'done', 'data': None}, ensure_ascii=False)

    async def run_planner_executor_loop(
        self,
        dashscope_client,
        tools,
        alerts_raw: str,
        logs_raw: str,
        docs_raw: str,
        max_rounds: int = 3,
    ) -> tuple[str, list[dict[str, Any]]]:
        trace: list[dict[str, Any]] = []
        failed_tool_counts: dict[str, int] = {}

        if not dashscope_client.enabled:
            report = self.build_report()
            trace.append(
                {
                    'round': 0,
                    'decision': 'FINISH',
                    'plannerStatus': 'FINISH',
                    'plannerDecision': 'FINISH',
                    'plannerStep': '模型不可用，进入本地降级路径',
                    'executorStatus': 'SKIPPED',
                    'terminationReason': 'dashscope_disabled',
                    'reason': 'dashscope_disabled',
                }
            )
            return report, trace

        planner_system = (
            '你是 Planner Agent，同时承担 Replanner 角色。\n'
            '你的任务：根据告警/日志/文档证据决定下一步。\n'
            '你必须只输出 JSON，格式如下：\n'
            '{"decision":"PLAN|EXECUTE|FINISH","step":"...","tool":{"name":"...","arguments":{}},"nextHint":"..."}\n'
            '为兼容状态语义，也可额外输出同义字段：plannerDecision/plannerStep/plannerNextHint。\n'
            '若需要执行工具，请输出 decision=EXECUTE 并给出 tool。\n'
            '若信息已足够生成最终报告，请输出 decision=FINISH。\n'
            '调用日志相关工具时，region 优先使用 ap-guangzhou。\n'
            '如果某个工具连续 3 次失败或无有效结果，必须停止该方向并在 nextHint 说明无法完成原因。\n'
            '禁止编造。'
        )

        executor_feedback = 'N/A'

        for round_idx in range(1, max_rounds + 1):
            planner_user = (
                f'当前轮次: {round_idx}\n\n'
                f'[alerts]\n{alerts_raw[:2000]}\n\n'
                f'[logs]\n{logs_raw[:2000]}\n\n'
                f'[docs]\n{docs_raw[:2000]}\n\n'
                f'[executor_feedback]\n{executor_feedback[:1500]}\n\n'
                '请输出下一步决策 JSON。'
            )

            planner_raw = await dashscope_client.chat(
                [
                    {'role': 'system', 'content': planner_system},
                    {'role': 'user', 'content': planner_user},
                ],
                stream=False,
            )

            planner_json = self._extract_json_object(planner_raw)
            decision = self._normalize_decision(
                planner_json.get('decision')
                or planner_json.get('plannerDecision')
                or planner_json.get('planner_status')
            )
            step = str(
                planner_json.get('step')
                or planner_json.get('plannerStep')
                or planner_json.get('planner_step')
                or ''
            ).strip()
            tool_spec = planner_json.get('tool') or {}
            next_hint = str(
                planner_json.get('nextHint')
                or planner_json.get('plannerNextHint')
                or planner_json.get('planner_next_hint')
                or ''
            ).strip()

            trace.append(
                {
                    'round': round_idx,
                    'planner_raw': planner_raw,
                    'decision': decision,
                    'plannerStatus': decision,
                    'plannerDecision': decision,
                    'step': step,
                    'plannerStep': step,
                    'tool': tool_spec,
                    'nextHint': next_hint,
                    'plannerNextHint': next_hint,
                    'executorStatus': 'PENDING',
                    'terminationReason': '',
                }
            )

            if decision == 'EXECUTE':
                tool_name = (tool_spec.get('name') or '').strip()
                tool_args = tool_spec.get('arguments') or {}
                if not tool_name:
                    executor_payload = {
                        'status': 'FAILED',
                        'summary': 'Planner 未提供 tool.name',
                        'evidence': '',
                        'nextHint': '请补充 tool.name 后重试',
                    }
                    executor_feedback = json.dumps(executor_payload, ensure_ascii=False)
                    trace[-1]['executor_feedback'] = executor_feedback
                    trace[-1]['executorStatus'] = 'FAILED'
                    continue

                try:
                    tool_result = await tools.run(tool_name, tool_args)
                    if not str(tool_result).strip() or 'no_results' in str(tool_result).lower():
                        failed_tool_counts[tool_name] = failed_tool_counts.get(tool_name, 0) + 1
                        executor_payload = {
                            'status': 'FAILED',
                            'summary': f'工具执行无有效结果: {tool_name}',
                            'tool': tool_name,
                            'arguments': tool_args,
                            'evidence': str(tool_result)[:2000],
                            'nextHint': '请切换查询方向或缩小范围',
                        }
                    else:
                        failed_tool_counts[tool_name] = 0
                        executor_payload = {
                            'status': 'SUCCESS',
                            'summary': f'已执行工具 {tool_name}',
                            'tool': tool_name,
                            'arguments': tool_args,
                            'evidence': str(tool_result)[:2000],
                            'nextHint': next_hint or '基于证据继续规划下一步',
                        }
                except Exception as e:
                    failed_tool_counts[tool_name] = failed_tool_counts.get(tool_name, 0) + 1
                    executor_payload = {
                        'status': 'FAILED',
                        'summary': f'工具执行失败: {tool_name}',
                        'error': str(e),
                        'tool': tool_name,
                        'arguments': tool_args,
                        'evidence': '',
                        'nextHint': '请检查参数或切换备用工具',
                    }

                executor_feedback = json.dumps(executor_payload, ensure_ascii=False)
                trace[-1]['executor_feedback'] = executor_feedback
                trace[-1]['executorStatus'] = executor_payload.get('status', 'UNKNOWN')

                if tool_name and failed_tool_counts.get(tool_name, 0) >= 3:
                    reason = f'工具 {tool_name} 连续 3 次失败或无有效结果，终止该方向'
                    trace.append(
                        {
                            'round': round_idx,
                            'decision': 'FINISH',
                            'plannerStatus': 'FINISH',
                            'plannerDecision': 'FINISH',
                            'plannerStep': '停止当前方向并输出无法完成原因',
                            'executorStatus': 'FAILED',
                            'terminationReason': reason,
                            'reason': reason,
                        }
                    )
                    break
                continue

            if decision == 'PLAN':
                executor_feedback = json.dumps(
                    {'status': 'INFO', 'summary': step or 'Planner 给出阶段计划，进入下一轮重规划'},
                    ensure_ascii=False,
                )
                trace[-1]['executor_feedback'] = executor_feedback
                trace[-1]['executorStatus'] = 'INFO'
                continue

            if decision == 'FINISH':
                trace[-1]['terminationReason'] = next_hint or 'planner_finish'

            break

        report = await self.build_report_with_llm(
            dashscope_client,
            alerts_raw=alerts_raw,
            logs_raw=logs_raw,
            docs_raw=docs_raw,
            planner_feedback=json.dumps(trace, ensure_ascii=False),
        )

        return report, trace

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        if not text:
            return {}

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return {}

        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    @staticmethod
    def _normalize_decision(decision: Any) -> str:
        raw = str(decision or '').upper().strip()
        if raw in {'PLAN', 'EXECUTE', 'FINISH'}:
            return raw
        return 'PLAN'
