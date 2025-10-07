# tests/conftest.py

"""
Fixtures compartidos para la suite de tests de Aipha_1.1
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import yaml
import sys

# Asegurar que el path incluye la raíz del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_config():
    """
    Crea una configuración temporal para tests.
    Se limpia automáticamente después de cada test.
    """
    temp_dir = tempfile.mkdtemp()
    
    config = {
        'system': {
            'storage_root': str(Path(temp_dir) / 'aipha_test_storage')
        },
        'atomic_update_system': {
            'version_history_file_name': 'VERSION_HISTORY.json',
            'global_state_file_name': 'global_state.json',
            'action_history_file_name': 'action_history.json',
            'dependencies_lock_file_name': 'dependencies.lock.json',
            'backups_dir_name': 'backups',
            'config_dir_name': 'config'
        },
        'context_sentinel': {
            'knowledge_base_db_name': 'knowledge_base.db',
            'global_state_dir_name': 'global_state',
            'global_state_file_name': 'current_state.json',
            'action_history_dir_name': 'action_history',
            'action_history_file_name': 'current_history.json'
        }
    }
    
    yield config
    
    # Cleanup después del test
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def critical_memory(temp_config):
    """Fixture de CriticalMemoryRules para tests."""
    from aipha.core.atomic_update_system import CriticalMemoryRules
    return CriticalMemoryRules(temp_config)


@pytest.fixture
def context_sentinel(temp_config):
    """Fixture de ContextSentinel para tests."""
    from aipha.core.context_sentinel import ContextSentinel
    return ContextSentinel(temp_config)


@pytest.fixture
def redesign_helper(temp_config):
    """Fixture de RedesignHelper para tests."""
    from aipha.core.redesign_helper import RedesignHelper
    helper = RedesignHelper(temp_config)
    helper.initialize()  # Pre-inicializar para la mayoría de tests
    return helper