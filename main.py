"""
Aipha Main Entry Point - Sistema de Automejora Inteligente

Este módulo inicializa y ejecuta el sistema Aipha completo, incluyendo:
- RedesignHelper para automejora del sistema
- AiphaShadow para consultas contextuales
- Gestión de configuración y logging
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from aipha.core.redesign_helper import RedesignHelper
from shadow.aipha_shadow import AiphaShadow

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load system configuration from YAML file with fallback defaults.

    Args:
        config_path (str): Path to configuration file. Defaults to 'config.yaml'.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.

    Side effects:
        - Creates default config file if it doesn't exist.
        - Logs configuration loading status.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        default_config = {
            'system': {
                'storage_root': './aipha_memory_storage',
                'log_level': 'INFO'
            },
            'knowledge_manager': {
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
                'capture_types': [
                    "decision", "architecture", "implementation",
                    "test", "bug_fix", "optimization", "documentation", "principle"
                ]
            }
        }
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Default configuration created at {config_path}")
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
            sys.exit(1)

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Configuration file is empty")
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging based on system configuration.

    Args:
        config (Dict[str, Any]): System configuration dictionary.

    Side effects:
        - Configures global logging settings.
    """
    log_level = getattr(logging, config.get('system', {}).get('log_level', 'INFO').upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_environment() -> bool:
    """
    Validate that the runtime environment meets system requirements.

    Returns:
        bool: True if environment is valid, False otherwise.

    Side effects:
        - Logs validation results and warnings.
    """
    try:
        # Check Python version
        if sys.version_info < (3, 11):
            logger.warning(f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
                         "Recommended: Python 3.11+")

        # Check required packages
        required_packages = ['chromadb', 'sentence_transformers', 'openai', 'yaml']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error("Install with: pip install chromadb sentence-transformers openai pyyaml")
            return False

        logger.info("Environment validation passed")
        return True

    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def main() -> int:
    """
    Main entry point for the Aipha system.

    Returns:
        int: Exit code (0 for success, 1 for failure).

    Side effects:
        - Initializes and runs the complete Aipha system.
        - Logs system status and operations.
    """
    try:
        # Load configuration
        config = load_config()

        # Setup logging
        setup_logging(config)

        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed. Exiting.")
            return 1

        storage_root = Path(config['system']['storage_root'])

        # Initialize core systems
        logger.info("Initializing Aipha core systems...")

        # RedesignHelper (integrates ContextSentinel and Knowledge Manager)
        aipha_system = RedesignHelper(config)

        # AiphaShadow for contextual queries
        shadow_system = AiphaShadow()

        # Demonstrate ATR proposal flow
        logger.info("Demonstrating ATR proposal flow...")
        aipha_system.demonstrate_atr_proposal_flow()

        # Example queries with different LLMs
        print("\n=== AiphaShadow Contextual Queries ===")
        try:
            # OpenAI query (commented due to API compatibility)
            # print("OpenAI:", shadow_system.query("¿Qué es Aipha?", "openai"))
            print("Gemini:", shadow_system.query("¿Qué es Aipha?", "gemini"))
        except Exception as e:
            logger.warning(f"Shadow system query failed: {e}")

        # Get final system status
        status = aipha_system.get_system_status()
        logger.info(f"System initialization completed successfully")
        logger.info(f"Final system status: {status}")

        return 0

    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)