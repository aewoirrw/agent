from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.xhs_agent import XhsAgentService


class XhsAgentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_retries_until_approved(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(
            side_effect=[
                '流量关键词：护肤误区、熬夜脸、早八通勤\n爆款切入点：低成本急救、对比反差\n痛点：想快速见效',
                '普通标题\n正文没有表情\n#标签1 #标签2',
                '{"approved": false, "feedback": "请补充 Emoji、增强标题钩子，并更明显融入爆款洞察"}',
                '熬夜脸逆袭！早八前这样做真的有用吗？✨\n我试了 3 天，变化太明显了😭\n通勤前 5 分钟急救思路直接抄作业！💡\n如果你也是熬夜党，评论区告诉我你的肤况～\n#熬夜护肤 #早八通勤 #护肤急救 #小红书爆款 #学生党护肤',
                '{"approved": true, "feedback": "审核通过"}',
            ]
        )

        service = XhsAgentService(llm)
        result = await service.generate('熬夜后如何快速恢复气色')

        self.assertTrue(result['success'])
        self.assertTrue(result['approved'])
        self.assertEqual(result['iterations'], 2)
        self.assertEqual(result['feedback'], '审核通过')
        self.assertIn('✨', result['draft'])
        self.assertEqual(llm.chat.await_count, 5)

    async def test_generate_allows_pass_when_reviewer_json_is_invalid(self) -> None:
        llm = AsyncMock()
        llm.chat = AsyncMock(
            side_effect=[
                '关键词：租房、独居、收纳\n切入点：小空间改造、低预算提升幸福感\n痛点：空间乱、预算少、改造难',
                '10 平米出租屋也能住出幸福感！🪴\n低预算改造真的救了我！✨\n#租房改造 #小空间收纳 #独居生活 #低预算布置 #出租屋好物',
                '这不是 JSON',
            ]
        )

        service = XhsAgentService(llm)
        result = await service.generate('出租屋改造')

        self.assertTrue(result['success'])
        self.assertTrue(result['approved'])
        self.assertEqual(result['iterations'], 1)
        self.assertEqual(result['feedback'], 'JSON解析异常，强制放行')


class XhsGenerateApiTests(unittest.TestCase):
    def test_generate_route_returns_service_result(self) -> None:
        client = TestClient(app)

        with patch(
            'app.main.xhs_agent_service.generate',
            new=AsyncMock(
                return_value={
                    'success': True,
                    'topic': '夏季防晒',
                    'trendingInsights': '关键词：防晒黑、防晒霜、防晒补涂',
                    'draft': '这篇是最终文案 😎\n#夏季防晒 #防晒攻略 #变白思路 #通勤护肤 #学生党护肤',
                    'iterations': 2,
                    'feedback': '审核通过',
                    'approved': True,
                }
            ),
        ):
            response = client.post('/api/xhs/generate', json={'topic': '夏季防晒'})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['code'], 200)
        self.assertTrue(payload['data']['success'])
        self.assertEqual(payload['data']['topic'], '夏季防晒')
        self.assertEqual(payload['data']['iterations'], 2)

    def test_generate_stream_route_emits_step_and_done(self) -> None:
        client = TestClient(app)

        async def _mock_stream_generate(topic: str):
            yield {
                'type': 'node_finished',
                'node': 'searcher',
                'delta': {'trending_insights': '关键词：防晒黑、防晒补涂'},
                'state': {
                    'topic': topic,
                    'trending_insights': '关键词：防晒黑、防晒补涂',
                    'draft': '',
                    'feedback': '',
                    'iterations': 0,
                    'approved': False,
                },
            }
            yield {
                'type': 'done',
                'result': {
                    'success': True,
                    'topic': topic,
                    'trendingInsights': '关键词：防晒黑、防晒补涂',
                    'draft': '最终文案 😎\n#夏季防晒 #防晒攻略',
                    'iterations': 1,
                    'feedback': '审核通过',
                    'approved': True,
                },
            }

        with patch('app.main.xhs_agent_service.stream_generate', new=_mock_stream_generate):
            response = client.post('/api/xhs/generate_stream', json={'topic': '夏季防晒'})

        self.assertEqual(response.status_code, 200)
        self.assertIn('assistant.xhs.step', response.text)
        self.assertIn('assistant.xhs.result', response.text)
        self.assertIn('assistant.done', response.text)


if __name__ == '__main__':
    unittest.main()