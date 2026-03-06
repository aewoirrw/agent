# Java → Python 行为对齐清单（重构底线：逻辑不变）

## 重构底线（必须遵守）

- 不改变接口语义（URL、入参、出参、SSE 结束条件）。
- 不改变业务判定逻辑（成功/失败条件、回退条件、错误传播边界）。
- 不改变核心状态流转（会话历史窗口、AIOps 闭环状态）。
- 仅允许做结构重组、命名优化、可观测性增强；不得改变功能结果。

---

## 1) 接口层对齐

### 1.1 `POST /api/chat`
- Java：ReactAgent 非流式调用，支持工具调用与会话上下文。
- Python：已支持工具调用、会话历史和统一响应格式。
- 结论：**已基本对齐**。
- 后续动作（不改逻辑）：补充字段级契约测试（`code/message/data.success/answer`）。

### 1.2 `POST /api/chat_stream` (SSE)
- Java：流式输出中包含模型增量与工具执行中间态（节点事件语义）。
- Python：已输出 `content/done/error`，并新增 `tool_started/tool_finished`。
- Python 事件契约补充：新增 `eventKey`（Java 语义映射），同时保留原 `type`，保证兼容。
- 差异：Java 的节点事件类型更细，Python 仍是简化版事件模型。
- 后续动作（不改逻辑）：统一事件字典与映射表，不改已有 `content/done/error` 语义。

#### 1.2.1 SSE 事件映射（签收基线）

| Python `eventKey` | Python `type`（兼容） | Java 节点语义 | 说明 |
|---|---|---|---|
| `assistant.content.delta` | `content` | `MODEL_DELTA` | 模型增量文本分片 |
| `assistant.tool.started` | `tool_started` | `TOOL_START` | 工具开始执行 |
| `assistant.tool.finished` | `tool_finished` | `TOOL_END` | 工具执行完成，含结果预览 |
| `assistant.planner.step` | `planner_step` | `PLANNER_STEP` | 规划/重规划状态推进 |
| `assistant.done` | `done` | `STREAM_DONE` | 当前流结束标记 |
| `assistant.error` | `error` | `STREAM_ERROR` | 当前流异常终止 |

签收规则：前端消费顺序固定为 `eventKey` 优先，缺失时回退 `type`。

### 1.3 `POST /api/ai_ops` (SSE)
- Java：Supervisor + Planner + Executor 原生多 Agent 编排。
- Python：Planner-Executor-Replanner 状态机（语义等价实现）。
- 差异：框架实现不同，但业务闭环已存在。
- 已对齐字段（兼容增强，不改逻辑）：`plannerStatus`、`plannerDecision`、`plannerStep`、`executorStatus`、`terminationReason`。
- Planner 输入/输出兼容：支持 `decision/step/nextHint` 与 `plannerDecision/plannerStep/plannerNextHint` 双格式。
- 后续动作（不改逻辑）：继续收敛提示词细节到 Java 版同义表达。

### 1.4 会话与文件接口
- `POST /api/chat/clear`、`GET /api/chat/session/{sessionId}`、`POST /api/upload`、`GET /milvus/health`
- 结论：**已对齐可用**。

---

## 2) 工具体系对齐

### 当前状态
- Java：ToolCallbackProvider + MCP 工具生态。
- Python：内置工具 + 动态 HTTP 工具注册（JSON 配置）+ MCP 原生协议（stdio）工具桥接。

### 差异
- 扩展机制不同（Provider vs 配置驱动）。
- Python 侧已具备 URL/header 占位符与 form/json 发送能力，并已支持 MCP 原生协议的 `initialize/tools/list/tools/call`。

### 后续动作（不改逻辑）
- 增加“工具能力映射表”（Java 工具名 ↔ Python 工具名）。
- 保持工具输入输出语义一致（字段名、状态码判断）而不是替换业务逻辑。

### 已落地兼容增强
- Python 工具执行层已支持工具名别名解析（如 `query_prometheus_alerts` / `get_available_log_topics` 自动映射到内置标准工具名），用于兼容 Java/Python 不同命名风格输入。

### 工具能力映射表（Java ↔ Python）

| Java 工具名/调用名 | Python 标准工具名 | 兼容别名示例 |
|---|---|---|
| `getCurrentDateTime` | `getCurrentDateTime` | `get_current_datetime`、`DateTimeTools.getCurrentDateTime` |
| `queryInternalDocs` | `queryInternalDocs` | `query_internal_docs`、`InternalDocsTools.queryInternalDocs` |
| `queryPrometheusAlerts` | `queryPrometheusAlerts` | `query_prometheus_alerts`、`QueryMetricsTools.queryPrometheusAlerts` |
| `getAvailableLogTopics` | `getAvailableLogTopics` | `get_available_log_topics`、`QueryLogsTools.getAvailableLogTopics` |
| `queryLogs` | `queryLogs` | `query_logs`、`QueryLogsTools.queryLogs` |

---

## 3) 模型层对齐

### 当前状态
- Java：默认 DashScope。
- Python：已切到 GLM（`zai-sdk`），并保留本地降级。

### 影响判断
- 这属于“实现依赖差异”，不是业务逻辑差异。
- 只要接口行为和工具调用结果语义一致，迁移目标可成立。

### 后续动作（不改逻辑）
- 固化模型错误分类映射（鉴权/限流/超时/服务异常）。
- 保持降级触发条件与返回结构稳定。

---

## 4) 前端可观测性对齐

### 当前状态
- Python 前端已支持执行过程面板（Planner/Tool）与筛选。
- 仍有差异：事件名与 Java 节点事件一一映射未完全固化。

### 后续动作（不改逻辑）
- 增加事件映射常量表（仅映射，不改事件处理主流程）。
- 增加“过程面板显示开关”配置（默认开）。

---

## 5) 验收与回归

### 已有
- `make smoke-py`
- `make smoke-py-strict`
- `make smoke-py-up`
- `make smoke-py-up-strict`
- `make test-external-tools`

### 建议新增（不改逻辑）
- 契约回归：固定 JSON 字段断言（包括 SSE 事件类型和结束标识）。
- 场景回归：
  1. 模型可用路径；
  2. 模型不可用降级路径；
  3. 工具连续失败 3 次终止路径。

---

## 结论

- 目前状态：**功能已高一致，机制仍有差异**。
- 迁移策略：坚持“逻辑不变”，优先做事件契约和状态语义对齐，再做结构重构。
- 执行原则：每次仅收敛一个差异面，改完必须跑严格回归。
