# tests/test_integration.py

"""
Tests de integración - flujo completo del sistema
"""

import pytest
from pathlib import Path
from aipha.core.redesign_helper import RedesignHelper
from aipha.core.atomic_update_system import ApprovalStatus


class TestFullInitializationFlow:
    """Tests del flujo completo de inicialización"""
    
    def test_end_to_end_initialization(self, temp_config):
        """
        Test de integración completo desde config hasta sistema listo.
        Simula el flujo de main.py.
        """
        # 1. Crear RedesignHelper
        helper = RedesignHelper(temp_config)
        assert helper is not None
        
        # 2. Inicializar
        result = helper.initialize()
        assert result is True
        
        # 3. Verificar estado
        status = helper.get_system_status()
        assert status['initialized'] is True
        assert status['knowledge_entries'] > 0
        
        # 4. Verificar integridad de componentes
        assert helper.critical_memory.verify_system_integrity() is True
        assert helper.context.verify_knowledge_base_integrity() is True
        
        # 5. Verificar que se pueden consultar datos
        knowledge = helper.context.get_knowledge_entries()
        assert len(knowledge) > 0
        
        criteria = helper.context.get_evaluation_criteria()
        assert len(criteria) > 0
        
        history = helper.critical_memory.get_version_history()
        assert len(history) > 0


class TestProposalLifecycle:
    """Tests del ciclo de vida completo de una propuesta"""
    
    def test_create_approve_apply_proposal(self, redesign_helper):
        """
        Test de integración del ciclo completo:
        Crear -> Aprobar -> Aplicar propuesta
        """
        # 1. Crear propuesta
        proposal = redesign_helper.critical_memory.create_change_proposal(
            description="Test Integration Proposal",
            justification="Integration testing",
            files_affected=["test.py"],
            diff_content="+ print('integration test')",
            author="IntegrationTest"
        )
        
        assert proposal.change_id is not None
        assert proposal.status == ApprovalStatus.PENDING
        
        # 2. Aprobar
        result = redesign_helper.critical_memory.approve_change(
            proposal,
            "IntegrationTestDeveloper"
        )
        assert result is True
        assert proposal.status == ApprovalStatus.APPROVED
        
        # 3. Aplicar
        result = redesign_helper.critical_memory.apply_atomic_update(proposal)
        assert result is True
        
        # 4. Verificar que se actualizó la versión
        new_version = redesign_helper.critical_memory.get_current_version()
        assert "1.1.1" in new_version  # Versión incrementada
        
        # 5. Verificar integridad post-actualización
        assert redesign_helper.critical_memory.verify_system_integrity() is True