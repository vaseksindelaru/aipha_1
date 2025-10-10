import yaml
import pytest
from pathlib import Path
from aipha.core.knowledge_manager.manager import AIPHAConfig, VectorDBManager, CaptureSystem, LLMQuerySystem, DevelopmentStep
from datetime import datetime
import uuid
import openai

@pytest.fixture
def config():
    config_path = Path('config.yaml')
    if not config_path.exists():
        # Crea config de test si no existe
        test_config = {
            'system': {'storage_root': './test_storage'},
            'knowledge_manager': {
                'project_root': './',
                'knowledge_db_path': './test_storage/knowledge_base',
                'logs_path': './test_storage/logs',
                'chroma_persist_dir': './test_storage/chroma_db',
                'collection_name': 'test_aipha',
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_dimension': 384,
                'llm_provider': 'openai',
                'llm_model': 'gpt-3.5-turbo',
                'api_key_env_var': 'OPENAI_API_KEY',
                'auto_capture': True,
                'capture_types': ["test"]
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_aipha_config(config):
    aipha_config = AIPHAConfig(config)
    assert aipha_config.PROJECT_ROOT == Path('./')
    assert aipha_config.COLLECTION_NAME == 'test_aipha'

def test_vector_db_manager(config):
    aipha_config = AIPHAConfig(config)
    db_manager = VectorDBManager(aipha_config)
    assert db_manager.collection.name == 'test_aipha'

def test_capture_system(config):
    aipha_config = AIPHAConfig(config)
    db_manager = VectorDBManager(aipha_config)
    capture_system = CaptureSystem(aipha_config, db_manager)
    
    step = DevelopmentStep(
        id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        type="test",
        title="Test Title",
        content="Test Content",
        metadata={"key": "value"}
    )
    capture_system.capture_manual(step)
    
    results = db_manager.search("Test Content")
    assert len(results) > 0
    assert "Test Content" in results[0]['content']

# def test_llm_query_system(config, monkeypatch):
#     # Mock OpenAI para test (evita llamadas reales)
#     class MockResponse:
#         choices = [{'message': {'content': 'Mock LLM response'}}]
#
#     monkeypatch.setattr('openai.ChatCompletion.create', lambda **kwargs: MockResponse())
#
#     aipha_config = AIPHAConfig(config)
#     db_manager = VectorDBManager(aipha_config)
#     llm_query_system = LLMQuerySystem(aipha_config, db_manager)
#
#     result = llm_query_system.query("Test query")
#     assert result == 'Mock LLM response'