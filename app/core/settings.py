from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(__file__).resolve().parents[2] / '.env'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'SuperBizAgent-Python'
    port: int = 9900
    upload_path: str = './uploads'
    allowed_extensions: str = 'txt,md'

    zhipu_api_key: str = ''
    dashscope_api_key: str = 'your-api-key-here'
    llm_base_url: str = ''
    chat_model: str = 'glm-4.7-flash'
    embedding_model: str = 'text-embedding-v4'
    local_embedding_enabled: bool = True
    local_embedding_model_path: str = './embedding_model'
    # 本地 embedding 运行设备：'cpu' | 'cuda' | 'cuda:0' | 'auto'
    # 当前默认值为 cuda：强制使用 GPU。
    # 若未检测到 CUDA / GPU，则服务会直接报错，不再回退 CPU。
    local_embedding_device: str = 'cuda'
    # 启动时预热本地 embedding 会占用较多 CPU/IO，可能影响首屏加载。
    # 这里提供一个轻量延迟，让服务先能更快响应静态页面与健康检查。
    local_embedding_prewarm_delay_sec: float = 2.0

    milvus_uri: str = ''
    milvus_token: str = ''
    milvus_db_name: str = 'default'
    milvus_host: str = 'localhost'
    milvus_port: int = 19530
    milvus_collection: str = 'vector_store'

    rag_top_k: int = 3
    rerank_enabled: bool = True
    rerank_candidate_k: int = 8
    rerank_timeout_sec: float = 30.0
    modelscope_reranker_base_url: str = 'https://ms-ens-7ed58ba5-37a9.api-inference.modelscope.cn/v1'
    modelscope_reranker_api_key: str = ''
    modelscope_reranker_model: str = 'Qwen/Qwen3-Reranker-4B'
    document_chunk_max_size: int = 800
    document_chunk_overlap: int = 100
    embedding_concurrency: int = 4

    prometheus_base_url: str = 'http://localhost:9090'
    prometheus_mock_enabled: bool = False
    cls_mock_enabled: bool = False
    external_tools_config_path: str = ''
    mcp_servers_config_path: str = ''

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def static_dir(self) -> Path:
        return self.project_root / 'src' / 'main' / 'resources' / 'static'

    @property
    def docs_dir(self) -> Path:
        return self.project_root / 'aiops-docs'

    @property
    def llm_api_key(self) -> str:
        if (self.llm_base_url or '').strip():
            key = (self.dashscope_api_key or '').strip()
            if key and key != 'your-api-key-here':
                return key
        key = (self.zhipu_api_key or '').strip()
        if key:
            return key
        return (self.dashscope_api_key or '').strip()


settings = Settings()
