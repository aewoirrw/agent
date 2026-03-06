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
    chat_model: str = 'glm-4.7-flash'
    embedding_model: str = 'text-embedding-v4'
    local_embedding_enabled: bool = True
    local_embedding_model_path: str = './embedding_model'

    milvus_uri: str = ''
    milvus_token: str = ''
    milvus_db_name: str = 'default'
    milvus_host: str = 'localhost'
    milvus_port: int = 19530
    milvus_collection: str = 'vector_store'

    rag_top_k: int = 3
    document_chunk_max_size: int = 800
    document_chunk_overlap: int = 100

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
        key = (self.zhipu_api_key or '').strip()
        if key:
            return key
        return (self.dashscope_api_key or '').strip()


settings = Settings()
