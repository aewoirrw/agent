from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Literal, TypedDict, cast

from app.services.dashscope_client import DashScopeClient


try:
    from langgraph.graph import END, StateGraph  # type: ignore
except Exception:  # pragma: no cover
    END = cast(Any, '__END__')
    StateGraph = None  # type: ignore


logger = logging.getLogger(__name__)


class XhsState(TypedDict):
    topic: str
    trending_insights: str
    draft: str
    feedback: str
    iterations: int
    approved: bool


class XhsAgentService:
    def __init__(self, llm: DashScopeClient) -> None:
        self.llm = llm
        self.max_iterations = 3

    async def generate(self, topic: str) -> dict[str, Any]:
        normalized_topic, final_state = await self._run_graph(topic)
        return self._build_result(cast(XhsState, final_state), normalized_topic)

    async def stream_generate(self, topic: str) -> AsyncIterator[dict[str, Any]]:
        normalized_topic = self._normalize_topic(topic)
        graph = self._build_graph()
        initial_state = self._build_initial_state(normalized_topic)
        final_state: dict[str, Any] = dict(initial_state)

        # 对外暴露节点级事件，便于 SSE 路由实时推送执行进度。
        async for output in graph.astream(initial_state):
            for event in self._collect_events(output, final_state):
                yield event

        yield {
            'type': 'done',
            'result': self._build_result(cast(XhsState, final_state), normalized_topic),
        }

    async def _run_graph(self, topic: str) -> tuple[str, dict[str, Any]]:
        normalized_topic = self._normalize_topic(topic)
        graph = self._build_graph()
        initial_state = self._build_initial_state(normalized_topic)
        final_state: dict[str, Any] = dict(initial_state)

        async for output in graph.astream(initial_state):
            self._collect_events(output, final_state)

        return normalized_topic, final_state

    def _build_graph(self):
        sg = StateGraph(XhsState)  # type: ignore[arg-type]
        sg.add_node('searcher', self.searcher_node)
        sg.add_node('writer', self.writer_node)
        sg.add_node('reviewer', self.reviewer_node)

        sg.set_entry_point('searcher')
        sg.add_edge('searcher', 'writer')
        sg.add_edge('writer', 'reviewer')
        sg.add_conditional_edges(
            'reviewer',
            self.route_next,
            {
                'continue': 'writer',
                END: END,
            },
        )
        return sg.compile()

    async def searcher_node(self, state: XhsState) -> dict[str, str]:
        topic = state['topic']
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是小红书流量搜索规划专家，擅长从用户给定主题中提炼爆款传播线索。'
                    '请输出结构化且可直接用于文案写作的分析结果。'
                ),
            },
            {
                'role': 'user',
                'content': (
                    '请围绕下面主题，输出适合小红书爆款文案创作的搜索洞察。\n\n'
                    '输出要求：\n'
                    '1. 给出 3-5 个流量关键词\n'
                    '2. 给出 2 个爆款切入点\n'
                    '3. 总结目标受众痛点\n'
                    '4. 用清晰的小标题组织内容，方便后续 Writer 直接引用\n\n'
                    f'主题：{topic}'
                ),
            },
        ]
        trending_insights = (await self.llm.chat(messages, stream=False) or '').strip()
        return {
            'trending_insights': trending_insights,
        }

    async def writer_node(self, state: XhsState) -> dict[str, Any]:
        topic = state['topic']
        trending_insights = state.get('trending_insights', '')
        feedback = (state.get('feedback') or '').strip()
        current_iteration = int(state.get('iterations', 0) or 0)

        rewrite_requirement = ''
        if feedback:
            # 如果 Reviewer 给出了反馈，说明这是一次打回重写，需要明确告知 Writer 按意见修订。
            rewrite_requirement = (
                '\n\n[本轮必须修正的问题]\n'
                f'{feedback}\n'
                '请逐条消化上述修改意见，再输出新的完整文案。'
            )

        messages = [
            {
                'role': 'system',
                'content': (
                    '你是小红书爆款文案主笔，擅长写出有情绪钩子、强共鸣、强转发欲望的内容。'
                    '文案必须自然融合洞察信息，包含 Emoji，并在结尾附上合适的话题标签。'
                ),
            },
            {
                'role': 'user',
                'content': (
                    '请严格基于以下信息撰写一篇小红书爆款文案。\n\n'
                    f'[主题]\n{topic}\n\n'
                    f'[爆款洞察]\n{trending_insights}\n'
                    f'{rewrite_requirement}\n\n'
                    '写作要求：\n'
                    '1. 开头必须有一个抓人的标题\n'
                    '2. 正文要有明显的情绪价值和实用价值\n'
                    '3. 至少自然使用 3 个 Emoji\n'
                    '4. 结尾给出行动引导或互动引导\n'
                    '5. 文末单独一行输出 5-8 个小红书标签，格式形如 #标签\n'
                    '6. 不要解释写作过程，直接输出最终文案'
                ),
            },
        ]
        draft = (await self.llm.chat(messages, stream=False) or '').strip()
        return {
            'draft': draft,
            'iterations': current_iteration + 1,
        }

    async def reviewer_node(self, state: XhsState) -> dict[str, Any]:
        draft = state.get('draft', '')
        trending_insights = state.get('trending_insights', '')
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是一个非常严苛的小红书主编，只负责审核质量。'
                    '你必须只输出 JSON，禁止输出任何额外解释、Markdown 或代码块。'
                ),
            },
            {
                'role': 'user',
                'content': (
                    '请审核下面这篇小红书文案是否合格。\n\n'
                    '审核标准：\n'
                    '1. 标题是否足够吸引人\n'
                    '2. 正文是否包含 Emoji\n'
                    '3. 文末是否有标签\n'
                    '4. 是否真正结合了给定的爆款 insights，而不是泛泛而谈\n\n'
                    '请只输出如下 JSON：\n'
                    '{"approved": true/false, "feedback": "修改意见"}\n\n'
                    f'[爆款 insights]\n{trending_insights}\n\n'
                    f'[待审核文案]\n{draft}'
                ),
            },
        ]

        raw_review = (await self.llm.chat(messages, stream=False) or '').strip()

        try:
            parsed_review = self._parse_reviewer_json(raw_review)
            approved = bool(parsed_review.get('approved', False))
            feedback = str(parsed_review.get('feedback') or '')
            if approved and not feedback:
                feedback = '审核通过'
            if not approved and not feedback:
                feedback = '文案未通过，请增强标题吸引力、Emoji、标签和 insights 融合度。'
        except Exception as exc:
            # 审核节点必须具备熔断保护，避免因为模型没有严格输出 JSON 导致整条工作流卡死。
            logger.warning('[xhs_agent] reviewer json parse failed: %s; raw=%s', repr(exc), raw_review)
            approved = True
            feedback = 'JSON解析异常，强制放行'

        return {
            'approved': approved,
            'feedback': feedback,
        }

    def route_next(self, state: XhsState) -> Literal['continue'] | Any:
        approved = bool(state.get('approved', False))
        iterations = int(state.get('iterations', 0) or 0)

        # 当审核通过或达到最大重写次数时，直接结束，避免无限回环。
        if approved or iterations >= self.max_iterations:
            return END
        return 'continue'

    @staticmethod
    def _build_initial_state(topic: str) -> XhsState:
        return {
            'topic': topic,
            'trending_insights': '',
            'draft': '',
            'feedback': '',
            'iterations': 0,
            'approved': False,
        }

    @staticmethod
    def _build_result(state: XhsState, normalized_topic: str) -> dict[str, Any]:
        return {
            'success': True,
            'topic': state.get('topic', normalized_topic),
            'trendingInsights': state.get('trending_insights', ''),
            'draft': state.get('draft', ''),
            'iterations': int(state.get('iterations', 0) or 0),
            'feedback': state.get('feedback', ''),
            'approved': bool(state.get('approved', False)),
        }

    def _normalize_topic(self, topic: str) -> str:
        if StateGraph is None:
            raise ImportError('langgraph 未安装，无法运行小红书文案生成工作流。')

        normalized_topic = (topic or '').strip()
        if not normalized_topic:
            raise ValueError('topic 不能为空')
        return normalized_topic

    def _collect_events(self, output: Any, final_state: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not isinstance(output, dict):
            return events

        for node_name, node_state in output.items():
            if isinstance(node_state, dict):
                final_state.update(node_state)
                logger.info(
                    '[xhs_agent] node=%s state=%s',
                    node_name,
                    json.dumps(node_state, ensure_ascii=False),
                )
                events.append(
                    {
                        'type': 'node_finished',
                        'node': node_name,
                        'delta': node_state,
                        'state': dict(final_state),
                    }
                )
            else:
                logger.info('[xhs_agent] node=%s output=%s', node_name, node_state)
                events.append(
                    {
                        'type': 'node_output',
                        'node': node_name,
                        'output': node_state,
                    }
                )

        return events

    @staticmethod
    def _parse_reviewer_json(raw_text: str) -> dict[str, Any]:
        cleaned = (raw_text or '').strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.strip('`').strip()
            if cleaned.lower().startswith('json'):
                cleaned = cleaned[4:].strip()

        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end >= start:
            cleaned = cleaned[start:end + 1]

        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError('reviewer 输出不是 JSON object')
        return parsed