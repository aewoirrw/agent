# SuperBizAgent Python 版

## 启动

1. 安装依赖

```bash
cd python_app
pip install -e .
```

2. 设置环境变量

```bash
export ZHIPU_API_KEY=your-api-key
```

3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload
```

或使用一键启动脚本（会先清理同端口旧 uvicorn 进程，避免端口占用）：

```bash
cd python_app
./start_server.sh
```

可选参数：

```bash
HOST=0.0.0.0 PORT=9900 RELOAD=1 ./start_server.sh
```

## 兼容接口

- `POST /api/chat`
- `POST /api/chat_stream` (SSE)
- `POST /api/ai_ops` (SSE)
- `POST /api/chat/clear`
- `GET /api/chat/session/{sessionId}`
- `POST /api/upload`
- `GET /api/upload/tasks/{taskId}`
- `GET /milvus/health`

## 已迁移能力（第二阶段）

- Chat 接口已接入工具调用（函数调用）
	- `getCurrentDateTime`
	- `queryInternalDocs`
	- `queryPrometheusAlerts`
	- `getAvailableLogTopics`
	- `queryLogs`
- AIOps 接口已增加“告警 + 日志 + 文档”输入拼装，并流式返回报告
- Milvus 不可用时自动降级到内存向量检索

## 第二阶段验收清单（已验证可运行）

### 运行环境

- 使用项目虚拟环境：`python_app/.venv`
- 系统 Python 可能缺少依赖（如 fastapi），请勿直接用系统 Python 启动

### 启动命令

```bash
cd python_app
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload
```

### 验收项

1. 健康检查

```bash
curl http://127.0.0.1:9900/milvus/health
```

期望：HTTP 200；无 Milvus 时返回内存模式集合（如 `in_memory_docs`）。

2. 普通问答

```bash
curl -X POST http://127.0.0.1:9900/api/chat \
	-H "Content-Type: application/json" \
	-d '{"Id":"stage2-check","Question":"你好，简单自我介绍"}'
```

期望：HTTP 200；返回 `code=200` 且 `data.success=true`。

3. 流式问答（SSE）

```bash
curl -N -X POST http://127.0.0.1:9900/api/chat_stream \
	-H "Content-Type: application/json" \
	-d '{"Id":"stage2-check","Question":"现在几点"}'
```

期望：返回 `event: message` 的流式分片，最终包含 `type=done`。

4. AIOps 流式分析（SSE）

```bash
curl -N -X POST http://127.0.0.1:9900/api/ai_ops
```

期望：流式输出“告警 + 日志 + 文档”拼装内容和报告片段。

### 说明

- 未配置或无法访问 GLM（智谱）时，Chat 会自动降级为本地回复。
- 这不影响第二阶段“可启动、可调用、可流式返回”的验收目标。

## Java → Python 功能对照（迁移验收基线）

> 目标：Python 版本对外功能不低于 Java 版本。
> 重构底线：逻辑不变（仅做结构优化与可观测性增强）。

详细行为对齐清单见：`MIGRATION_PARITY_CHECKLIST.md`（含 SSE `eventKey -> type -> Java语义` 映射与签收规则）

### A. 接口兼容（必须一致）

- [x] `POST /api/chat`
- [x] `POST /api/chat_stream`（SSE）
- [x] `POST /api/ai_ops`（SSE）
- [x] `POST /api/chat/clear`
- [x] `GET /api/chat/session/{sessionId}`
- [x] `POST /api/upload`
- [x] `GET /milvus/health`

### B. 核心能力对照

- [x] 多轮会话窗口管理（保留上下文、支持清空）
- [x] Chat 工具调用（时间/文档/告警/日志）
- [x] Chat Stream 流式输出
- [x] Chat Stream 支持工具调用结果返回
- [x] AIOps 输入拼装（alerts + logs + docs）
- [x] AIOps 流式输出最终报告
- [x] AIOps Planner-Executor-Replanner 闭环（基础版）
- [x] 文件上传后自动向量化（已优化为上传秒回 + 后台索引任务）
- [x] Milvus 不可用自动降级内存检索

### C. 稳定性与可观测性

- [x] GLM 调用重试
- [x] GLM 错误分类（鉴权/限流/超时/DNS/5xx）
- [x] 失败降级（大模型失败时仍可返回报告）
- [x] 启动脚本自动清理端口占用（`start_server.sh`）

### D. 尚需持续优化（不影响当前功能等价）

- [ ] AIOps 编排策略进一步贴近 Java 版提示词细节与状态字段（语义一致性）
- [x] 增加自动化 smoke test 脚本，作为回归门禁
- [ ] 前端展示后端错误分类细节（提升排障效率）

### E. 验收标准（建议）

当满足以下条件，可判定“Python 版达到 Java 版功能等价迁移目标”：

1. A/B/C 三部分均通过；
2. 在有网和无网（或模型不可用）两种场景都可完成核心流程；
3. 关键接口可通过命令行复验且结果稳定。

## 自动化冒烟测试（推荐）

已提供脚本：`scripts/smoke_test.sh`

### 用法

1. 启动服务（默认端口 9910）
2. 执行脚本

```bash
cd python_app
./scripts/smoke_test.sh
```

如服务不在 9910 端口，可指定：

```bash
BASE_URL=http://127.0.0.1:9900 ./scripts/smoke_test.sh
```

脚本会检查：

- `GET /milvus/health`
- `POST /api/chat`
- `POST /api/chat_stream`（SSE）
- `POST /api/ai_ops`（SSE）
- `GET /api/chat/session/{id}`
- `POST /api/chat/clear`

## 上传索引优化（RAG）

Python 版上传链路已优化为：

- 上传接口先落盘并立即返回 `taskId`
- embedding / 向量写入在后台异步执行
- 前端通过任务状态接口展示进度条

### 进度查询接口

```bash
curl http://127.0.0.1:9910/api/upload/tasks/{taskId}
```

返回字段包含：

- `status`: `queued|running|success|failed`
- `progress`: 0~100
- `stage`: 当前阶段（如 `chunking` / `embedding` / `writing_vectors`）
- `totalChunks` / `completedChunks`
- `durationMs`

## 上传优化基准采集与对比

为了量化“优化后到底快了多少”，已新增两类脚本：

- `scripts/benchmark_upload.py`：采集上传响应耗时 + 后台索引完成耗时，输出 CSV
- `scripts/compare_upload_benchmarks.py`：对比两份 CSV，计算平均值 / P50 / P95 改善百分比

### 1. 采集当前版本基准

在项目根目录执行：

```bash
make benchmark-upload
```

默认行为：

- 请求地址：`http://127.0.0.1:9910`
- 测试文件：`python_app/uploads/upload_test.md`
- 重复次数：`5`
- 输出文件：`python_app/benchmark_outputs/upload_benchmark_after.csv`

也可覆盖参数：

```bash
FILE=uploads/cpu_high_usage.md REPEAT=10 LABEL=after OUTPUT=benchmark_outputs/after_large.csv make benchmark-upload
```

CSV 包含字段：

- `upload_response_ms`：上传接口返回耗时
- `index_total_ms`：后台索引总耗时
- `total_chunks` / `completed_chunks`
- `storage_mode`
- `status` / `error`

### 1.1 冷启动说明

当前版本会在服务启动后后台预热本地 embedding 模型，以降低首次上传/首次检索时的冷启动抖动。

- 服务刚启动后的前几十秒内，首次索引耗时可能仍略高
- 模型预热完成后，后续上传的 `index_total_ms` 会更稳定
- 该优化不会影响 `/api/upload` 的秒回语义

### 2. 对比优化前后结果

准备两份 CSV 后执行：

```bash
make compare-upload-benchmark BEFORE=benchmark_outputs/before.csv AFTER=benchmark_outputs/upload_benchmark_after.csv
```

输出会分别展示：

- Upload Response Time：接口返回时间
- Background Index Time：后台索引时间

并给出：

- `avg`
- `p50`
- `p95`
- `improvement=xx%`

### 3. 建议对比口径

建议至少保留两组数据：

1. **before**：优化前同步上传/索引版本
2. **after**：当前异步上传 + 后台 embedding/索引版本

如果当前代码已经切到异步版本，建议先用同一文件、同一环境、同一重复次数采集 `after` 基线，后续再与历史 `before` 结果做对比。

## 动态外部工具（可选）

为对齐 Java 版 ToolCallbackProvider 的扩展能力，Python 版支持从 JSON 配置文件动态加载 HTTP 工具。

### 配置步骤

1. 复制示例配置：

```bash
cp external_tools.example.json external_tools.json
```

2. 在 `.env` 设置：

```dotenv
EXTERNAL_TOOLS_CONFIG_PATH=external_tools.json
```

3. 重启服务后，模型可自动使用配置中的工具。

说明：

- 当前支持 `type=http` 工具；
- 支持 `GET/POST`；
- `url` 支持占位符（如 `{id}`）由工具参数填充。
- `headers` 支持占位符（如 `Authorization: Bearer {token}`）。
- 可选 `send_as=form` 用于表单提交（默认 `json`）。

### 回归验证

```bash
cd /zhangqi/mycodes/agent
make test-external-tools
```

该命令会自动启动本地 mock 接口，并验证：

- 动态工具能被注册到工具列表；
- URL 占位符渲染正确；
- Header 占位符渲染正确；
- `send_as=form` 提交正确。

## MCP 原生协议工具（stdio）

Python 版已支持通过 MCP 原生协议（stdio JSON-RPC）接入工具服务器，能力包括：

- `initialize`
- `tools/list`
- `tools/call`

并支持两种传输：

- `transport=stdio`（本地子进程）
- `transport=sse`（SSE + message endpoint）

### 配置步骤

1. 复制示例：

```bash
cp mcp_servers.example.json mcp_servers.json
```

2. 在 `.env` 中配置：

```dotenv
MCP_SERVERS_CONFIG_PATH=mcp_servers.json
```

3. 重启服务。

### 回归验证

```bash
cd /zhangqi/mycodes/agent
make test-mcp-tools
```

该命令会启动本地 mock MCP server，并验证：

- MCP 初始化成功；
- MCP 工具可被发现并注册；
- MCP 工具调用成功；
- `server.tool` 形式别名调用可用。

如需验证 SSE 传输：

```bash
cd /zhangqi/mycodes/agent
make test-mcp-sse-tools
```
