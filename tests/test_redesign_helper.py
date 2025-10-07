# tests/test_redesign_helper.py

"""
Tests para redesign_helper.py - RedesignHelper
"""

import pytest
from pathlib import Path
from aipha.core.redesign_helper import RedesignHelper


class TestInitialization:
    """Tests de inicialización de RedesignHelper"""
    
    def test_creates_components(self, temp_config):
        """Verifica que se crean los componentes base"""
        helper = RedesignHelper(temp_config)
        
        assert helper.critical_memory is not None
        assert helper.context is not None
        assert helper.initialized is False  # Aún no inicializado
    
    def test_initialize_succeeds(self, temp_config):
        """Verifica que la inicialización completa es exitosa"""
        helper = RedesignHelper(temp_config)
        result = helper.initialize()
        
        assert result is True
        assert helper.initialized is True
    
    def test_loads_base_knowledge(self, redesign_helper):
        """Verifica que se carga el conocimiento base"""
        # redesign_helper fixture ya está inicializado
        entries = redesign_helper.context.get_knowledge_entries()
        
        # Debe haber al menos las entradas base
        assert len(entries) >= 10  # Arquitectura, roles, protocolos, principios
        
        # Verificar categorías clave
        categories = {entry.category for entry in entries}
        assert 'architecture' in categories
        assert 'agent_roles' in categories
        assert 'protocol_flow' in categories
        assert 'coding_principles' in categories


class TestSystemStatus:
    """Tests del estado del sistema"""
    
    def test_status_before_initialization(self, temp_config):
        """Verifica estado antes de inicializar"""
        helper = RedesignHelper(temp_config)
        status = helper.get_system_status()
        
        assert status['initialized'] is False
    
    def test_status_after_initialization(self, redesign_helper):
        """Verifica estado después de inicializar"""
        status = redesign_helper.get_system_status()
        
        assert status['initialized'] is True
        assert 'current_version' in status
        assert 'storage_root' in status
        assert status['knowledge_entries'] > 0
        assert status['action_history_size'] > 0


class TestKnowledgeContent:
    """Tests del contenido de conocimiento cargado"""
    
    def test_has_architecture_knowledge(self, redesign_helper):
        """Verifica conocimiento sobre arquitectura"""
        entries = redesign_helper.context.get_knowledge_entries(category='architecture')
        
        assert len(entries) >= 2
        titles = [e.title for e in entries]
        assert any('Layer Architecture' in t for t in titles)
        assert any('Componentes del Núcleo' in t for t in titles)
    
    def test_has_all_agent_roles(self, redesign_helper):
        """Verifica que están todos los roles de agentes"""
        entries = redesign_helper.context.get_knowledge_entries(category='agent_roles')
        
        required_agents = [
            'ChangeProposer',
            'ProposalEvaluator',
            'CodecraftSage',
            'MetaImprover',
            'Data Postprocessor'
        ]
        
        titles = [e.title for e in entries]
        for agent in required_agents:
            assert any(agent in t for t in titles), f"Falta rol de {agent}"
    
    def test_has_safety_first_principle(self, redesign_helper):
        """Verifica que existe el principio Safety-First"""
        entries = redesign_helper.context.get_knowledge_entries(category='coding_principles')
        
        titles = [e.title for e in entries]
        assert any('Safety-First' in t for t in titles)
        
        # Verificar que tiene importancia crítica
        safety_entry = next(e for e in entries if 'Safety-First' in e.title)
        assert safety_entry.metadata['importance'] == 'critical'