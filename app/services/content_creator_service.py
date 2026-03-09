from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Optional, TypedDict, cast

from app.services.dashscope_client import DashScopeClient
from app.services.tools import AgentTools


try:
    from langgraph.graph import END, StateGraph  # type: ignore
except Exception:  # pragma: no cover
    END = cast(Any, '__END__')
    StateGraph = None  # type: ignore


class ContentCreatorState(TypedDict, total=False):
    user_goal: str
    platform: str
    audience: str
    seed_topic: str
    current_topic: str
    topic_analysis: str
    trend_keywords: list[str]
    scout_source: str
    visual_concept: str
    image_prompt: str
    generated_image_url: str
    generated_image_result: str
    review_feedback: str
    review_passed: bool
    iteration_count: int
    max_iterations: int
    final_summary: str


class ContentCreatorService:
    def __init__(self, dashscope_client: DashScopeClient, tools: AgentTools) -> None:
        self.dashscope_client = dashscope_client
        self.tools = tools

    async def run(
        self,
        *,
        goal: str,
        platform: str = 'general',
        audience: str = 'general',
        seed_topic: str = '',
        max_iterations: int = 2,
    ) -> dict[str, Any]:
        if StateGraph is None:
            raise ImportError('langgraph 未安装，无法运行内容创作 Agent。')

        graph = self._build_graph()
        initial_state: ContentCreatorState = {
            'user_goal': goal.strip(),
            'platform': platform.strip() or 'general',
            'audience': audience.strip() or 'general',
            'seed_topic': seed_topic.strip(),
            'current_topic': '',
            'topic_analysis': '',
            'trend_keywords': [],
            'scout_source': '',
            'visual_concept': '',
            'image_prompt': '',
            'generated_image_url': '',
            'generated_image_result': '',
            'review_feedback': '',
            'review_passed': False,
            'iteration_count': 0,
            'max_iterations': max(1, int(max_iterations or 1)),
            'final_summary': '',
        }
        final_state = cast(ContentCreatorState, await graph.ainvoke(initial_state))
        return {
            'success': True,
            'topic': final_state.get('current_topic') or '',
            'topicAnalysis': final_state.get('topic_analysis') or '',
            'trendKeywords': final_state.get('trend_keywords') or [],
            'visualConcept': final_state.get('visual_concept') or '',
            'imagePrompt': final_state.get('image_prompt') or '',
            'generatedImageUrl': final_state.get('generated_image_url') or '',
            'generatedImageResult': final_state.get('generated_image_result') or '',
            'reviewFeedback': final_state.get('review_feedback') or '',
            'reviewPassed': bool(final_state.get('review_passed')),
            'iterationCount': int(final_state.get('iteration_count') or 0),
            'maxIterations': int(final_state.get('max_iterations') or 0),
            'summary': final_state.get('final_summary') or '',
            'platform': final_state.get('platform') or '',
            'audience': final_state.get('audience') or '',
            'scoutSource': final_state.get('scout_source') or '',
        }

    def _build_graph(self):
        sg = StateGraph(ContentCreatorState)  # type: ignore[arg-type]
        sg.add_node('scout', self._node_scout)
        sg.add_node('creative', self._node_creative)
        sg.add_node('prompt', self._node_prompt)
        sg.add_node('generate', self._node_generate)
        sg.add_node('review', self._node_review)
        sg.add_node('revise', self._node_revise)
        sg.set_entry_point('scout')
        sg.add_edge('scout', 'creative')
        sg.add_edge('creative', 'prompt')
        sg.add_edge('prompt', 'generate')
        sg.add_edge('generate', 'review')

        def _route(state: ContentCreatorState) -> Literal['revise'] | Any:
            passed = bool(state.get('review_passed'))
            iteration_count = int(state.get('iteration_count') or 0)
            max_iterations = int(state.get('max_iterations') or 1)
            if passed or iteration_count >= max_iterations:
                return END
            return 'revise'

        sg.add_conditional_edges('review', _route, {'revise': 'revise', END: END})
        sg.add_edge('revise', 'generate')
        return sg.compile()

    async def _node_scout(self, state: ContentCreatorState) -> ContentCreatorState:
        goal = str(state.get('user_goal') or '').strip()
        seed_topic = str(state.get('seed_topic') or '').strip()
        platform = str(state.get('platform') or 'general').strip()
        scout_source = await self._discover_trends(goal=goal, platform=platform, seed_topic=seed_topic)
        fallback_topic = seed_topic or goal or '今日热门话题'
        scout_json = await self._ask_json(
            system_prompt=(
                '你是爆款侦察兵 Agent。基于给定素材，提炼一个最值得创作的热点主题。'
                '必须只输出 JSON。'
            ),
            user_prompt=(
                '请输出 JSON，格式如下：\n'
                '{"topic":"...","analysis":"...","keywords":["..."],"sourceSummary":"..."}\n\n'
                f'[goal]\n{goal}\n\n'
                f'[platform]\n{platform}\n\n'
                f'[seed_topic]\n{seed_topic or "N/A"}\n\n'
                f'[search_material]\n{scout_source}\n'
            ),
            default={
                'topic': fallback_topic,
                'analysis': '当前未接入实时热榜工具，使用输入目标作为候选主题。',
                'keywords': [fallback_topic],
                'sourceSummary': scout_source or 'no_external_search_result',
            },
        )
        return {
            **state,
            'current_topic': str(scout_json.get('topic') or fallback_topic),
            'topic_analysis': str(scout_json.get('analysis') or ''),
            'trend_keywords': self._normalize_keywords(scout_json.get('keywords'), fallback_topic),
            'scout_source': str(scout_json.get('sourceSummary') or scout_source or ''),
        }

    async def _node_creative(self, state: ContentCreatorState) -> ContentCreatorState:
        creative_json = await self._ask_json(
            system_prompt=(
                '你是创意总监 Agent。你负责把热点主题转成有传播力的视觉概念。'
                '输出必须是 JSON。'
            ),
            user_prompt=(
                '请输出 JSON，格式如下：\n'
                '{"visualConcept":"...","hook":"...","composition":"..."}\n\n'
                f'[topic]\n{state.get("current_topic") or ""}\n\n'
                f'[analysis]\n{state.get("topic_analysis") or ""}\n\n'
                f'[keywords]\n{json.dumps(state.get("trend_keywords") or [], ensure_ascii=False)}\n\n'
                f'[audience]\n{state.get("audience") or "general"}\n'
            ),
            default={
                'visualConcept': f'围绕“{state.get("current_topic") or "热点主题"}”设计一个强情绪对比的主视觉场景。',
                'hook': '突出情绪冲突和生活共鸣。',
                'composition': '主体居中，前景有记忆点道具，背景简洁。',
            },
        )
        visual_concept = '；'.join(
            part for part in [
                str(creative_json.get('visualConcept') or ''),
                str(creative_json.get('hook') or ''),
                str(creative_json.get('composition') or ''),
            ] if part
        )
        return {
            **state,
            'visual_concept': visual_concept,
        }

    async def _node_prompt(self, state: ContentCreatorState) -> ContentCreatorState:
        prompt_json = await self._ask_json(
            system_prompt=(
                '你是提示词工程师 Agent。请把视觉概念翻译成适合图像生成模型的专业英文 prompt。'
                '输出必须是 JSON。'
            ),
            user_prompt=(
                '请输出 JSON，格式如下：\n'
                '{"imagePrompt":"...","negativePrompt":"...","styleNotes":"..."}\n\n'
                f'[topic]\n{state.get("current_topic") or ""}\n\n'
                f'[visual_concept]\n{state.get("visual_concept") or ""}\n\n'
                f'[platform]\n{state.get("platform") or "general"}\n'
            ),
            default={
                'imagePrompt': (
                    f'{state.get("current_topic") or "viral topic"}, '
                    f'{state.get("visual_concept") or "cinematic storytelling"}, '
                    'highly detailed, cinematic lighting, editorial illustration, social media cover, no text'
                ),
                'negativePrompt': 'blurry, low quality, deformed hands, extra fingers, gibberish text, watermark',
                'styleNotes': '适合封面图，强调主体识别度和情绪。',
            },
        )
        image_prompt = str(prompt_json.get('imagePrompt') or '').strip()
        negative_prompt = str(prompt_json.get('negativePrompt') or '').strip()
        style_notes = str(prompt_json.get('styleNotes') or '').strip()
        merged_prompt = image_prompt
        if negative_prompt:
            merged_prompt = f'{merged_prompt}\n\nNegative prompt: {negative_prompt}'
        if style_notes:
            merged_prompt = f'{merged_prompt}\n\nStyle notes: {style_notes}'
        return {
            **state,
            'image_prompt': merged_prompt,
        }

    async def _node_generate(self, state: ContentCreatorState) -> ContentCreatorState:
        next_iteration = int(state.get('iteration_count') or 0) + 1
        image_prompt = str(state.get('image_prompt') or '')
        topic = str(state.get('current_topic') or '')
        tool_result = await self._run_optional_tools(
            ['generateTrendingImage', 'generateImage', 'textToImage'],
            {
                'prompt': image_prompt,
                'topic': topic,
                'platform': state.get('platform') or 'general',
                'iteration': next_iteration,
            },
        )
        generated_url = ''
        generated_result = ''
        parsed = self._safe_json_loads(tool_result)
        if isinstance(parsed, dict):
            body = parsed.get('body') if isinstance(parsed.get('body'), dict) else parsed
            generated_url = str(
                body.get('image_url')
                or body.get('url')
                or body.get('imageUrl')
                or body.get('data', [{}])[0].get('url') if isinstance(body.get('data'), list) and body.get('data') else ''
            )
            generated_result = json.dumps(parsed, ensure_ascii=False)

        if not generated_url:
            digest = hashlib.sha1(image_prompt.encode('utf-8')).hexdigest()[:16]
            generated_url = f'mock://generated/{digest}.png'
            generated_result = generated_result or json.dumps(
                {
                    'success': True,
                    'mock': True,
                    'message': '未配置图像生成工具，返回 mock 结果用于流程联调。',
                    'image_url': generated_url,
                },
                ensure_ascii=False,
            )

        return {
            **state,
            'iteration_count': next_iteration,
            'generated_image_url': generated_url,
            'generated_image_result': generated_result,
        }

    async def _node_review(self, state: ContentCreatorState) -> ContentCreatorState:
        review_material = await self._run_optional_tools(
            ['reviewGeneratedImage', 'visionReviewImage', 'scoreGeneratedImage'],
            {
                'topic': state.get('current_topic') or '',
                'image_url': state.get('generated_image_url') or '',
                'prompt': state.get('image_prompt') or '',
                'platform': state.get('platform') or 'general',
            },
        )
        default_feedback = self._heuristic_review(state)
        review_json = await self._ask_json(
            system_prompt=(
                '你是艺术总监 Agent。你负责审核图片是否符合热点主题和社媒传播要求。'
                '如果没有真实视觉结果，也要基于 prompt 和生成结果给出严格审查。'
                '输出必须是 JSON。'
            ),
            user_prompt=(
                '请输出 JSON，格式如下：\n'
                '{"passed":true,"feedback":"...","summary":"..."}\n\n'
                f'[topic]\n{state.get("current_topic") or ""}\n\n'
                f'[topic_analysis]\n{state.get("topic_analysis") or ""}\n\n'
                f'[prompt]\n{state.get("image_prompt") or ""}\n\n'
                f'[generated_image_url]\n{state.get("generated_image_url") or ""}\n\n'
                f'[generation_result]\n{state.get("generated_image_result") or ""}\n\n'
                f'[tool_review]\n{review_material}\n\n'
                f'[heuristic_review]\n{json.dumps(default_feedback, ensure_ascii=False)}\n'
            ),
            default=default_feedback,
        )
        passed = bool(review_json.get('passed'))
        feedback = str(review_json.get('feedback') or '')
        summary = str(review_json.get('summary') or '')
        final_summary = (
            f'主题：{state.get("current_topic") or ""}\n'
            f'视觉概念：{state.get("visual_concept") or ""}\n'
            f'审核结论：{"通过" if passed else "需重试"}\n'
            f'说明：{summary or feedback}'
        )
        return {
            **state,
            'review_passed': passed,
            'review_feedback': feedback or summary,
            'final_summary': final_summary,
        }

    async def _node_revise(self, state: ContentCreatorState) -> ContentCreatorState:
        revised_json = await self._ask_json(
            system_prompt=(
                '你是提示词工程师 Agent，正在根据艺术总监反馈修订图像 prompt。'
                '输出必须是 JSON。'
            ),
            user_prompt=(
                '请输出 JSON，格式如下：\n'
                '{"imagePrompt":"...","changeSummary":"..."}\n\n'
                f'[topic]\n{state.get("current_topic") or ""}\n\n'
                f'[current_prompt]\n{state.get("image_prompt") or ""}\n\n'
                f'[review_feedback]\n{state.get("review_feedback") or ""}\n'
            ),
            default={
                'imagePrompt': self._apply_feedback_to_prompt(
                    str(state.get('image_prompt') or ''),
                    str(state.get('review_feedback') or ''),
                ),
                'changeSummary': '已根据审核反馈强化主题一致性并补充负面约束。',
            },
        )
        return {
            **state,
            'image_prompt': str(revised_json.get('imagePrompt') or state.get('image_prompt') or ''),
        }

    async def _discover_trends(self, *, goal: str, platform: str, seed_topic: str) -> str:
        search_query = seed_topic or goal or '今日热门内容'
        result = await self._run_optional_tools(
            ['searchTrendingTopics', 'webSearch', 'searchWeb'],
            {'query': search_query, 'platform': platform},
        )
        if result and 'Unknown tool' not in result:
            return result
        if seed_topic:
            return f'使用用户提供的种子主题: {seed_topic}'
        return f'未配置外部热榜工具，回退到用户目标推演: {search_query}'

    async def _run_optional_tools(self, tool_names: list[str], arguments: dict[str, Any]) -> str:
        for tool_name in tool_names:
            try:
                result = await self.tools.run(tool_name, arguments)
            except Exception as exc:
                result = json.dumps({'success': False, 'message': str(exc)}, ensure_ascii=False)
            if result and 'Unknown tool' not in result:
                return result
        return ''

    async def _ask_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        default: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.dashscope_client.enabled:
            return default
        try:
            raw = await self.dashscope_client.chat(
                [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                stream=False,
            )
        except Exception:
            return default
        parsed = self._extract_json_object(raw)
        if parsed:
            return parsed
        return default

    @staticmethod
    def _safe_json_loads(raw: str) -> Any:
        try:
            return json.loads(raw)
        except Exception:
            return raw

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any]:
        text = (raw or '').strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _normalize_keywords(raw_keywords: Any, fallback: str) -> list[str]:
        if isinstance(raw_keywords, list):
            items = [str(item).strip() for item in raw_keywords if str(item).strip()]
            if items:
                return items[:8]
        return [fallback]

    @staticmethod
    def _apply_feedback_to_prompt(prompt: str, feedback: str) -> str:
        clean_prompt = (prompt or '').strip()
        clean_feedback = (feedback or '').strip()
        if not clean_feedback:
            return clean_prompt
        if 'Negative prompt:' in clean_prompt:
            return f'{clean_prompt}\n\nRevision notes: {clean_feedback}'
        return f'{clean_prompt}\n\nNegative prompt: blurry, low quality, deformed hands, extra fingers, gibberish text, watermark\n\nRevision notes: {clean_feedback}'

    @staticmethod
    def _heuristic_review(state: ContentCreatorState) -> dict[str, Any]:
        prompt = str(state.get('image_prompt') or '').lower()
        topic = str(state.get('current_topic') or '')
        feedback: list[str] = []
        if 'no text' not in prompt and 'gibberish text' not in prompt:
            feedback.append('需要显式约束图片中不要出现乱码或大段文字。')
        if 'extra fingers' not in prompt and 'deformed hands' not in prompt:
            feedback.append('需要补充手部和肢体畸形的负面约束。')
        if len(prompt) < 80:
            feedback.append('当前 prompt 细节偏少，视觉信息不够具体。')
        if topic and topic.lower() not in prompt:
            feedback.append('prompt 对热点主题本身的指向性不够强。')
        passed = not feedback
        return {
            'passed': passed,
            'feedback': '；'.join(feedback) if feedback else '主题一致，约束完整，可以出图。',
            'summary': '文本级审核通过。' if passed else '文本级审核未通过，需要修订 prompt。',
        }