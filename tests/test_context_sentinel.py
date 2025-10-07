# tests/test_context_sentinel.py

"""
Tests para context_sentinel.py - ContextSentinel
"""

import pytest
from pathlib import Path
from aipha.core.context_sentinel import ContextSentinel


class TestInitialization:
    """Tests de inicialización de ContextSentinel"""
    
    def test_creates_database(self, context_sentinel):
        """Verifica que se crea la base de datos"""
        assert context_sentinel.db_file.exists()
    
    def test_creates_directories(self, context_sentinel):
        """Verifica que se crean los directorios necesarios"""
        assert context_sentinel.global_state_dir.exists()
        assert context_sentinel.action_history_dir.exists()
    
    def test_creates_json_files(self, context_sentinel):
        """Verifica que se crean los archivos JSON"""
        assert context_sentinel.global_state_file.exists()
        assert context_sentinel.action_history_file.exists()


class TestKnowledgeEntries:
    """Tests de entradas de conocimiento"""
    
    def test_add_knowledge_entry(self, context_sentinel):
        """Verifica que se pueden añadir entradas"""
        entry_id = context_sentinel.add_knowledge_entry(
            category="test",
            title="Test Entry",
            content="Test content",
            metadata={"importance": "low"}
        )
        
        assert entry_id is not None
        assert len(entry_id) == 12  # Hash MD5 truncado
    
    def test_retrieve_knowledge_entry(self, context_sentinel):
        """Verifica que se pueden recuperar entradas"""
        context_sentinel.add_knowledge_entry(
            category="test",
            title="Test Entry",
            content="Test content"
        )
        
        entries = context_sentinel.get_knowledge_entries(category="test")
        assert len(entries) == 1
        assert entries[0].title == "Test Entry"
    
    def test_filter_by_category(self, context_sentinel):
        """Verifica filtrado por categoría"""
        context_sentinel.add_knowledge_entry(
            category="cat1",
            title="Entry 1",
            content="Content 1"
        )
        context_sentinel.add_knowledge_entry(
            category="cat2",
            title="Entry 2",
            content="Content 2"
        )
        
        cat1_entries = context_sentinel.get_knowledge_entries(category="cat1")
        assert len(cat1_entries) == 1
        assert cat1_entries[0].title == "Entry 1"


class TestEvaluationCriteria:
    """Tests de criterios de evaluación"""
    
    def test_add_criteria(self, context_sentinel):
        """Verifica que se pueden añadir criterios"""
        criteria_id = context_sentinel.add_evaluation_criteria(
            name="Test Criterion",
            weight=0.5,
            description="Test description",
            examples_positive=["good"],
            examples_negative=["bad"],
            category="test"
        )
        
        assert criteria_id is not None
    
    def test_upsert_updates_existing(self, context_sentinel):
        """Verifica que upsert actualiza en lugar de duplicar"""
        # Añadir inicial
        context_sentinel.add_evaluation_criteria(
            name="Updatable",
            weight=0.3,
            description="Original",
            examples_positive=["old"],
            examples_negative=["old"],
            category="test"
        )
        
        # Actualizar
        context_sentinel.add_evaluation_criteria(
            name="Updatable",
            weight=0.8,
            description="Updated",
            examples_positive=["new"],
            examples_negative=["new"],
            category="test"
        )
        
        criteria = context_sentinel.get_evaluation_criteria(category="test")
        assert len(criteria) == 1
        assert criteria[0]['weight'] == 0.8
        assert criteria[0]['description'] == "Updated"


class TestIntegrity:
    """Tests de integridad de base de conocimiento"""
    
    def test_fresh_db_passes_integrity(self, context_sentinel):
        """Verifica que DB recién creada pasa integridad"""
        assert context_sentinel.verify_knowledge_base_integrity() is True