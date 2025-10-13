"""
Shared test fixtures for Aipha_1.1 test suite.

This module provides pytest fixtures for testing Aipha system components,
including temporary configurations, isolated storage, and pre-initialized
system instances.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import sys
from typing import Dict, Any, Generator

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_config() -> Generator[Dict[str, Any], None, None]:
    """
    Create temporary configuration for tests with isolated storage.

    This fixture provides a complete configuration dictionary with temporary
    directories that are automatically cleaned up after each test.

    Yields:
        Dict[str, Any]: Configuration dictionary with temporary paths.

    Side effects:
        - Creates temporary directories during test execution.
        - Automatically cleans up temporary directories after test completion.
    """
    temp_dir = Path(tempfile.mkdtemp())

    config = {
        'system': {
            'storage_root': str(temp_dir / 'aipha_test_storage'),
            'log_level': 'WARNING'  # Reduce log noise in tests
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
        },
        'knowledge_manager': {
            'project_root': str(temp_dir),
            'knowledge_db_path': str(temp_dir / 'knowledge_base'),
            'logs_path': str(temp_dir / 'logs'),
            'chroma_persist_dir': str(temp_dir / 'chroma_db'),
            'collection_name': 'test_collection',
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_dimension': 384,
            'llm_provider': 'openai',
            'llm_model': 'gpt-3.5-turbo',
            'api_key_env_var': 'OPENAI_API_KEY',
            'auto_capture': False,  # Disable auto-capture in tests
            'capture_types': ["test", "architecture"]
        }
    }

    yield config

    # Cleanup after test
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")


@pytest.fixture
def critical_memory(temp_config):
    """
    Fixture providing CriticalMemoryRules instance for tests.

    Args:
        temp_config: Temporary configuration fixture.

    Returns:
        CriticalMemoryRules: Initialized instance for testing.
    """
    from aipha.core.atomic_update_system import CriticalMemoryRules
    return CriticalMemoryRules(temp_config)


@pytest.fixture
def context_sentinel(temp_config):
    """
    Fixture providing ContextSentinel instance for tests.

    Args:
        temp_config: Temporary configuration fixture.

    Returns:
        ContextSentinel: Initialized instance for testing.
    """
    from aipha.core.context_sentinel import ContextSentinel
    return ContextSentinel(temp_config)


@pytest.fixture
def redesign_helper(temp_config):
    """
    Fixture providing RedesignHelper instance for tests.

    This fixture pre-initializes the helper to ensure consistent test state.

    Args:
        temp_config: Temporary configuration fixture.

    Returns:
        RedesignHelper: Fully initialized instance ready for testing.
    """
    from aipha.core.redesign_helper import RedesignHelper
    helper = RedesignHelper(temp_config)
    # Note: RedesignHelper auto-initializes in __init__, no need for manual initialize()
    return helper


@pytest.fixture
def change_proposer(temp_config):
    """
    Fixture providing ChangeProposer instance for tests.

    Args:
        temp_config: Temporary configuration fixture.

    Returns:
        ChangeProposer: Initialized instance for testing proposal generation.
    """
    from aipha.core.tools.change_proposer import ChangeProposer
    return ChangeProposer(temp_config)