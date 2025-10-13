import os
import shutil
import yaml
from pathlib import Path
import logging

from aipha.core.redesign_helper import RedesignHelper

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Punto de entrada para el ciclo de automejora interactivo."""
    # Asegúrate de que tu clave de API esté configurada como variable de entorno
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: La variable de entorno OPENAI_API_KEY no está configurada.")
        print("Por favor, ejecútala con: export OPENAI_API_KEY='tu-clave-aqui'")
        return

    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error(f"El archivo de configuración '{config_path}' no se encontró.")
        return

    config = load_config(config_path)

    # Limpieza opcional del directorio de almacenamiento para una ejecución limpia
    storage_root = Path(config['system']['storage_root'])
    if storage_root.exists():
        logger.warning(f"Limpiando el directorio de almacenamiento '{storage_root}' para una ejecución limpia.")
        shutil.rmtree(storage_root)

    try:
        # Inicializar el orquestador principal
        helper = RedesignHelper(config)
        helper.initialize()

        # Directiva de alto nivel para el LLM
        directive = "Revisa el documento CRITICAL_CONTRACTS.md y propón una mejora para clarificar el rol del 'ProposalEvaluator' en el protocolo de actualización atómica, detallando sus responsabilidades."

        # Ejecutar el ciclo interactivo
        helper.run_interactive_change_cycle(directive, use_llm=True)

    except Exception as e:
        logger.critical(f"Ha ocurrido un error fatal durante la ejecución: {e}", exc_info=True)

if __name__ == "__main__":
    main()