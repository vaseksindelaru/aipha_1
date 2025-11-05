#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_shadow_integration.py - Suite de tests para integración Shadow

Tests unitarios y de integración para validar:
- Registro de eventos Git en Shadow Memory
- Bridge AiphaLab y consultas
- Integridad de cadena de hashes
- Export de contexto para LLM

Autor: Shadow System
Versión: 1.0.0
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Añadir directorio shadow al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shadow'))

from shadow.aiphalab_bridge import AiphaLabBridge
from shadow.log_git_event import GitEventLogger


class TestShadowIntegration:
    """Tests de integración para el sistema Shadow"""

    @pytest.fixture
    def temp_memory_dir(self):
        """Directorio temporal para memoria de Shadow"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_memory_data(self):
        """Datos de memoria de ejemplo"""
        return [
            {
                "entry_id": "test-123",
                "timestamp": "2025-11-03T10:00:00.000000",
                "version_id": "Aipha_0.0.1",
                "source_component": "Git_Hook",
                "entry_category": "GIT_EVENT",
                "data_payload": {
                    "event_type": "commit",
                    "commit_hash": "abc123",
                    "commit_message": "Test commit",
                    "files_changed": ["test.py"]
                },
                "previous_entry_hash": "",
                "entry_hash": "hash123"
            }
        ]

    def test_log_git_event_commit(self, temp_memory_dir):
        """Test registro de evento commit"""
        # Crear logger y ejecutar
        logger = GitEventLogger(temp_memory_dir)
        event_data = {
            "event_type": "commit",
            "commit_hash": "test123",
            "commit_message": "Test commit message",
            "files_changed": ["test.py"]
        }
        result = logger.log_git_event(event_data)

        # Verificar resultado
        assert result is True

        # Verificar archivo creado
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        assert os.path.exists(memory_file)

        # Verificar contenido
        with open(memory_file, 'r') as f:
            data = json.load(f)

        assert len(data) == 1
        entry = data[0]
        assert entry['entry_category'] == 'GIT_EVENT'
        assert entry['data_payload']['event_type'] == 'commit'
        assert entry['data_payload']['commit_hash'] == 'test123'
        assert entry['data_payload']['commit_message'] == 'Test commit message'
        assert entry['data_payload']['files_changed'] == ['test.py']

    def test_log_git_event_push(self, temp_memory_dir):
        """Test registro de evento push"""
        logger = GitEventLogger(temp_memory_dir)
        event_data = {
            "event_type": "push",
            "commit_hash": "push123",
            "github_url": "https://github.com/user/repo/commit/push123"
        }
        result = logger.log_git_event(event_data)

        assert result is True

        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'r') as f:
            data = json.load(f)

        assert len(data) == 1
        entry = data[0]
        assert entry['data_payload']['event_type'] == 'push'
        assert entry['data_payload']['github_url'] == 'https://github.com/user/repo/commit/push123'

    def test_aiphalab_bridge_initialization(self, temp_memory_dir):
        """Test inicialización del bridge AiphaLab"""
        # Crear archivo de memoria vacío
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump([], f)

        bridge = AiphaLabBridge(temp_memory_dir)
        assert bridge.shadow_memory_path == temp_memory_dir
        assert bridge.memory_file == memory_file

    def test_aiphalab_bridge_query_empty_memory(self, temp_memory_dir):
        """Test consulta en memoria vacía"""
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump([], f)

        bridge = AiphaLabBridge(temp_memory_dir)
        result = bridge.query_shadow_memory({})

        assert result['status'] == 'success'
        assert result['total_entries'] == 0
        assert result['filtered_entries'] == 0
        assert result['data'] == []

    def test_aiphalab_bridge_query_with_data(self, temp_memory_dir, sample_memory_data):
        """Test consulta con datos existentes"""
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump(sample_memory_data, f)

        bridge = AiphaLabBridge(temp_memory_dir)
        result = bridge.query_shadow_memory({})

        assert result['status'] == 'success'
        assert result['total_entries'] == 1
        assert result['filtered_entries'] == 1
        assert len(result['data']) == 1

    def test_aiphalab_bridge_filter_by_category(self, temp_memory_dir, sample_memory_data):
        """Test filtro por categoría"""
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump(sample_memory_data, f)

        bridge = AiphaLabBridge(temp_memory_dir)

        # Consulta con filtro
        result = bridge.query_shadow_memory({'category': 'GIT_EVENT'})
        assert result['filtered_entries'] == 1

        # Consulta con filtro que no coincide
        result = bridge.query_shadow_memory({'category': 'NON_EXISTENT'})
        assert result['filtered_entries'] == 0

    def test_aiphalab_bridge_filter_by_component(self, temp_memory_dir, sample_memory_data):
        """Test filtro por componente"""
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump(sample_memory_data, f)

        bridge = AiphaLabBridge(temp_memory_dir)

        result = bridge.query_shadow_memory({'component': 'Git_Hook'})
        assert result['filtered_entries'] == 1

        result = bridge.query_shadow_memory({'component': 'NonExistent'})
        assert result['filtered_entries'] == 0

    def test_aiphalab_bridge_search_text(self, temp_memory_dir, sample_memory_data):
        """Test búsqueda por texto"""
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump(sample_memory_data, f)

        bridge = AiphaLabBridge(temp_memory_dir)

        result = bridge.query_shadow_memory({'search_text': 'Test commit'})
        assert result['filtered_entries'] == 1

        result = bridge.query_shadow_memory({'search_text': 'nonexistent text'})
        assert result['filtered_entries'] == 0

    def test_aiphalab_bridge_limit_results(self, temp_memory_dir):
        """Test límite de resultados"""
        # Crear múltiples entradas
        entries = []
        for i in range(10):
            entries.append({
                "entry_id": f"test-{i}",
                "timestamp": f"2025-11-03T10:{i:02d}:00.000000",
                "version_id": "Aipha_0.0.1",
                "source_component": "Test",
                "entry_category": "TEST_EVENT",
                "data_payload": {"test": f"entry_{i}"},
                "previous_entry_hash": "",
                "entry_hash": f"hash{i}"
            })

        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump(entries, f)

        bridge = AiphaLabBridge(temp_memory_dir)

        result = bridge.query_shadow_memory({'limit': 5})
        assert len(result['data']) == 5

        result = bridge.query_shadow_memory({'limit': 3})
        assert len(result['data']) == 3

    def test_aiphalab_bridge_export_context(self, temp_memory_dir, sample_memory_data):
        """Test export de contexto para AiphaLab"""
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump(sample_memory_data, f)

        bridge = AiphaLabBridge(temp_memory_dir)
        context = bridge.get_context_for_aiphalab()

        # Verificar que contiene elementos esperados
        assert "# 🔍 CONTEXTO SHADOW MEMORY" in context
        assert "Estado general del proyecto" in context
        assert "GIT_EVENT" in context
        assert "Test commit" in context

    def test_memory_integrity_validation(self, temp_memory_dir, sample_memory_data):
        """Test validación de integridad de memoria"""
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')
        with open(memory_file, 'w') as f:
            json.dump(sample_memory_data, f)

        bridge = AiphaLabBridge(temp_memory_dir)

        # Ejecutar consulta para activar validación
        result = bridge.query_shadow_memory({'include_metadata': True})

        # Verificar que incluye metadatos de integridad
        assert 'metadata' in result
        assert 'memory_integrity' in result['metadata']

        integrity = result['metadata']['memory_integrity']
        assert 'valid_entries' in integrity
        assert 'invalid_entries' in integrity
        assert 'chain_valid' in integrity

    def test_corrupted_memory_handling(self, temp_memory_dir):
        """Test manejo de memoria corrupta"""
        memory_file = os.path.join(temp_memory_dir, 'current_history.json')

        # Archivo corrupto (no es JSON válido)
        with open(memory_file, 'w') as f:
            f.write("not valid json {")

        bridge = AiphaLabBridge(temp_memory_dir)
        result = bridge.query_shadow_memory({})

        assert result['status'] == 'error'
        assert 'No se pudo cargar la memoria de Shadow' in result['message']

    def test_missing_memory_file(self, temp_memory_dir):
        """Test cuando no existe archivo de memoria"""
        bridge = AiphaLabBridge(temp_memory_dir)
        result = bridge.query_shadow_memory({})

        assert result['status'] == 'error'
        assert 'No se pudo cargar la memoria de Shadow' in result['message']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])