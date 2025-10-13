import logging
import yaml
from pathlib import Path
#import shutil  # Comentar si no se usa para limpieza

from aipha.core.redesign_helper import RedesignHelper
from shadow.aipha_shadow import AiphaShadow  # NUEVA importación

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        default_config = {
            'system': {'storage_root': './aipha_memory_storage'},
            'knowledge_manager': {  # Añade si no existe
                'project_root': './',
                'knowledge_db_path': './aipha_memory_storage/knowledge_base',
                'logs_path': './aipha_memory_storage/logs',
                'chroma_persist_dir': './aipha_memory_storage/chroma_db',
                'collection_name': 'aipha_development',
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_dimension': 384,
                'llm_provider': 'openai',
                'llm_model': 'gpt-3.5-turbo',
                'api_key_env_var': 'OPENAI_API_KEY',
                'auto_capture': True,
                'capture_types': ["decision", "architecture", "implementation", "test", "bug_fix", "optimization", "documentation", "principle"]
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f)
        logger.info(f"Config por defecto creada en {config_path}.")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    logging.basicConfig(level=logging.INFO)
    
    config = load_config()
    storage_root = Path(config['system']['storage_root'])
    
    # COMENTADO: No borrar storage_root para preservar ChromaDB y datos persistentes
    # if storage_root.exists():
    #     shutil.rmtree(storage_root)
    #     logger.info(f"Directorio de almacenamiento limpio: {storage_root}")
    # storage_root.mkdir(parents=True, exist_ok=True)
    
    # Instancia RedesignHelper (que integra ContextSentinel y Knowledge Manager)
    aipha_system = RedesignHelper(config)
    
    # NUEVO: Sistema Shadow para consultas
    shadow_system = AiphaShadow()
    
    # Sincronizar shadow con el repositorio actual (comentado para evitar bucle)
    # shadow_system.sync_with_repository()
    
    # Demostración de flujo ATR (o cualquier test)
    aipha_system.demonstrate_atr_proposal_flow()
    
    # NUEVO: Ejemplo de consultas con diferentes LLMs
    print("=== Consultas con AiphaShadow ===")
    # print("OpenAI:", shadow_system.query("¿Qué es Aipha?", "openai"))  # Comentado por compatibilidad OpenAI v1
    print("Gemini:", shadow_system.query("¿Qué es Aipha?", "gemini"))
    
    # Obtener estado del sistema
    status = aipha_system.get_system_status()
    logger.info(f"Estado final del sistema: {status}")

if __name__ == "__main__":
    main()