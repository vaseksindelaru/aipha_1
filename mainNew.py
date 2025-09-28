# main.py

import sys
import os
from datetime import datetime
from pathlib import Path
import logging
import json
import yaml # Necesitarás instalar PyYAML: pip install PyYAML
import shutil
from typing import Dict, Any, List, Optional

# --- Configuración del sistema de Logging ---
logging.basicConfig(
    level=logging.INFO, # Nivel mínimo de mensajes a procesar: INFO, WARNING, ERROR, CRITICAL. DEBUG se ignora.
    format='%(asctime)s - %(levelname)s - %(message)s', # Formato de los mensajes: fecha, nivel, mensaje.
    datefmt='%Y-%m-%dT%H:%M:%SZ' # Formato específico para la marca de tiempo (UTC).
)
logger = logging.getLogger(__name__)

# --- Importaciones de módulos locales ---
# Descomentamos estas líneas ahora que tenemos los archivos atomic_update_system.py y context_sentinel.py.
# La cláusula 'try-except' es una buena práctica para manejar si los archivos no se encuentran.
try:
    from atomic_update_system import CriticalMemoryRules, ChangeProposal, ApprovalStatus
    from context_sentinel import ContextSentinel
except ImportError as e:
    logger.error(f"Error importing system modules: {e}")
    logger.error("Asegúrese de que 'atomic_update_system.py' y 'context_sentinel.py' estén en el mismo directorio que 'main.py'.")
    sys.exit(1)

# --- Función para cargar la configuración desde un archivo YAML ---
def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo YAML.
    Si el archivo no existe, crea uno por defecto con la estructura aprobada.
    """
    if not config_path.exists():
        logger.warning(f"Archivo de configuración no encontrado: {config_path}. Creando uno por defecto.")
        
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

# --- Clase principal del sistema AiphaMemorySystem ---
# Esta clase orquesta los componentes CriticalMemoryRules y ContextSentinel.
class AiphaMemorySystem:
    """Orquestador principal para el sistema de memoria de Aipha_1.1."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa AiphaMemorySystem con la configuración del sistema.
        Parámetros:
            config (Dict[str, Any]): Diccionario de configuración cargado desde config.yaml.
        """
        self.config = config
        # 'storage_root' es la raíz de todos nuestros archivos de persistencia.
        self.storage_root = Path(self.config['system']['storage_root'])
        
        # Instanciamos CriticalMemoryRules, pasándole toda la configuración.
        # CriticalMemoryRules usará su sección de la configuración para saber cómo operar.
        self.critical_memory_rules = CriticalMemoryRules(self.config)
        
        # Instanciamos ContextSentinel, también pasándole toda la configuración.
        # ContextSentinel usará su sección de la configuración.
        self.context_sentinel = ContextSentinel(self.config)
        
        self.system_initialized = False # Flag para saber si el sistema está inicializado.
        logger.info("AiphaMemorySystem: Componentes CriticalMemoryRules y ContextSentinel instanciados.")
        
    def initialize_system(self) -> bool:
        """
        Orquesta la inicialización completa del sistema de memoria,
        verificando la integridad de los componentes y cargando el conocimiento inicial.
        Retorna True si la inicialización es exitosa, False en caso contrario.
        """
        try:
            logger.info("AiphaMemorySystem: Iniciando el sistema de memoria crítica...")
            
            # --- Verificación de integridad de CriticalMemoryRules ---
            # Le pedimos a CriticalMemoryRules que verifique que sus archivos están bien.
            if not self.critical_memory_rules.verify_system_integrity():
                logger.error("La verificación de integridad de CriticalMemoryRules falló. El sistema no puede inicializarse.")
                return False
            
            # --- Verificación de integridad de ContextSentinel ---
            # Le pedimos a ContextSentinel que verifique que su base de conocimiento está bien.
            if not self.context_sentinel.verify_knowledge_base_integrity():
                logger.error("La verificación de integridad de la base de conocimiento de ContextSentinel falló. El sistema no puede inicializarse.")
                return False
            
            # --- Carga de Criterios por Defecto ---
            # ContextSentinel carga los criterios que usaremos para evaluar propuestas.
            self.context_sentinel.add_default_evaluation_criteria()
            
            # --- Añadir Conocimiento Inicial ---
            # ContextSentinel carga la documentación básica del sistema.
            self._add_initial_knowledge()
            
            self.system_initialized = True
            logger.info("AiphaMemorySystem: Sistema de memoria inicializado exitosamente.")
            return True
            
        except Exception as e:
            logger.error(f"AiphaMemorySystem: La inicialización del sistema falló: {e}")
            return False

    def _add_initial_knowledge(self):
        """
        Añade entradas de conocimiento iniciales al ContextSentinel.
        Este es el "conocimiento base" que el sistema tendrá al arrancar.
        """
        logger.info("AiphaMemorySystem: Añadiendo entradas de conocimiento iniciales...")
        self.context_sentinel.add_knowledge_entry(
            category="architecture",
            title="Aipha_1.1 Layer Architecture",
            content="Aipha_1.1 se estructura en 5 capas principales: Núcleo (redesignHelper), Context Sentinel, Data Preprocessor, Trading Flow, y Oracles.",
            metadata={"importance": "critical", "layer": "architecture", "version": "1.1.0"}
        )
        self.context_sentinel.add_knowledge_entry(
            category="protocol",
            title="Atomic Update Protocol",
            content="El protocolo de actualización atómica asegura cambios controlados con aprobación explícita, previniendo reescrituras completas.",
            metadata={"type": "protocol", "critical": True, "version": "1.1.0"}
        )
        # Añade más entradas de conocimiento aquí si lo deseas.
        logger.info("AiphaMemorySystem: Entradas de conocimiento iniciales añadidas.")


# --- Función principal de ejecución del programa ---
def main():
    """
    Función principal que orquesta la inicialización y ejecución del sistema Aipha_1.1.
    """
    logger.info("Iniciando el sistema de reglas de memoria críticas de Build_Aipha_1.1...")
    
    config_file_path = Path("config.yaml")
    config = load_config(config_file_path) 
    storage_root = Path(config['system']['storage_root'])
    
    # --- Limpieza Opcional del Directorio de Almacenamiento ---
    if storage_root.exists():
        logger.info(f"Limpiando la ruta raíz de almacenamiento: '{storage_root}' para una demostración limpia...")
        try:
            shutil.rmtree(storage_root) 
            logger.info(f"Directorio '{storage_root}' limpiado exitosamente.")
        except OSError as e:
            logger.error(f"Error al eliminar el directorio '{storage_root}': {e}")
            logger.warning("Procediendo con el almacenamiento existente, lo que podría afectar la consistencia de la demostración.")
    
    # --- Inicialización del Sistema Aipha ---
    # Creamos una instancia de nuestra clase AiphaMemorySystem, pasándole la configuración.
    aipha_system = AiphaMemorySystem(config) 
    
    # Llamamos al método initialize_system() para que orqueste la configuración de sus componentes.
    if not aipha_system.initialize_system(): 
        logger.error("La inicialización del sistema AiphaMemorySystem falló. Terminando programa.")
        return 1 # Devolvemos 1 para indicar que el programa terminó con un error.
    
    # Si la inicialización fue exitosa, simplemente informamos y terminamos por ahora.
    logger.info("El sistema Aipha_1.1 está listo para operar (inicialización completada).")
    
    return 0 # Devolvemos 0 para indicar que el programa terminó con éxito.

# --- Bloque de ejecución principal (Entry Point) ---
if __name__ == "__main__":
    sys.exit(main())