# main.py

"""
Punto de entrada de Aipha_1.1.
Mantiene mínima lógica, delega toda la orquestación a RedesignHelper.
"""

import sys
import logging
from pathlib import Path
import yaml
import shutil

from aipha.core.redesign_helper import RedesignHelper

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo config.yaml
    
    Returns:
        Diccionario con la configuración del sistema
    """
    if not config_path.exists():
        logger.warning(f"Config no encontrado: {config_path}. Creando por defecto.")
        
        default_config_content = """
system:
  storage_root: "./aipha_memory_storage"

atomic_update_system:
  version_history_file_name: "VERSION_HISTORY.json"
  global_state_file_name: "global_state.json"
  action_history_file_name: "action_history.json"
  dependencies_lock_file_name: "dependencies.lock.json"
  backups_dir_name: "backups"
  config_dir_name: "config"

context_sentinel:
  knowledge_base_db_name: "knowledge_base.db"
  global_state_dir_name: "global_state"
  global_state_file_name: "current_state.json"
  action_history_dir_name: "action_history"
  action_history_file_name: "current_history.json"
"""
        config_path.write_text(default_config_content, encoding='utf-8')
        return yaml.safe_load(default_config_content)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Función principal - mínima lógica de bootstrapping."""
    logger.info("=" * 60)
    logger.info("Iniciando Aipha_1.1 - Sistema de Auto-Construcción")
    logger.info("=" * 60)
    
    # 1. Cargar configuración
    config = load_config(Path("config.yaml"))
    storage_root = Path(config['system']['storage_root'])
    
    # 2. Limpieza opcional (solo para demos/desarrollo)
    # COMENTAR ESTA SECCIÓN EN PRODUCCIÓN
    if storage_root.exists():
        logger.info(f"Limpiando {storage_root} para demostración limpia...")
        try:
            shutil.rmtree(storage_root)
            logger.info("Directorio limpiado exitosamente")
        except OSError as e:
            logger.warning(f"No se pudo limpiar {storage_root}: {e}")
    
    # 3. Crear e inicializar RedesignHelper
    redesign_helper = RedesignHelper(config)
    
    if not redesign_helper.initialize():
        logger.error("Fallo crítico en inicialización del sistema")
        return 1
    
    # 4. Mostrar estado del sistema
    status = redesign_helper.get_system_status()
    logger.info(f"Estado del sistema: {status}")
    
    # 5. Placeholder para ciclo de trabajo principal
    # TODO: Aquí irá la lógica del bucle principal en fases futuras
    logger.info("Sistema Aipha_1.1 inicializado y listo")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())