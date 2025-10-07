# tests/test_atomic_update_system.py

"""
Tests para atomic_update_system.py - CriticalMemoryRules
"""

import pytest
from pathlib import Path
from aipha.core.atomic_update_system import (
    CriticalMemoryRules,
    ChangeProposal,
    ApprovalStatus,
    VersionInfo
)


class TestInitialization:
    """Tests de inicialización de CriticalMemoryRules"""
    
    def test_creates_storage_directory(self, critical_memory, temp_config):
        """Verifica que se crea el directorio de almacenamiento"""
        storage_root = Path(temp_config['system']['storage_root'])
        assert storage_root.exists()
    
    def test_creates_all_critical_files(self, critical_memory):
        """Verifica que se crean todos los archivos críticos"""
        assert critical_memory.version_history_file.exists()
        assert critical_memory.global_state_file.exists()
        assert critical_memory.action_history_file.exists()
        assert critical_memory.dependencies_file.exists()
    
    def test_initial_version_is_1_1_0(self, critical_memory):
        """Verifica que la versión inicial es 1.1.0"""
        assert critical_memory.current_version.major == 1
        assert critical_memory.current_version.minor == 1
        assert critical_memory.current_version.patch == 0
    
    def test_version_history_has_initial_entry(self, critical_memory):
        """Verifica que VERSION_HISTORY.json tiene entrada inicial"""
        history = critical_memory.get_version_history()
        assert len(history) == 1
        assert history[0]['status'] == 'active'
        assert history[0]['change_id'] == 'INIT_000'


class TestChangeProposalCreation:
    """Tests de creación de propuestas de cambio"""
    
    def test_creates_proposal_with_unique_id(self, critical_memory):
        """Verifica que cada propuesta tiene un ID único"""
        proposal1 = critical_memory.create_change_proposal(
            description="Test 1",
            justification="Testing",
            files_affected=["test.py"],
            diff_content="+ test",
            author="TestSystem"
        )
        
        proposal2 = critical_memory.create_change_proposal(
            description="Test 2",
            justification="Testing",
            files_affected=["test.py"],
            diff_content="+ test",
            author="TestSystem"
        )
        
        assert proposal1.change_id != proposal2.change_id
    
    def test_proposal_starts_as_pending(self, critical_memory):
        """Verifica que las propuestas inician como PENDING"""
        proposal = critical_memory.create_change_proposal(
            description="Test",
            justification="Test",
            files_affected=["test.py"],
            diff_content="test",
            author="Test"
        )
        
        assert proposal.status == ApprovalStatus.PENDING
    
    def test_proposal_has_incremented_version(self, critical_memory):
        """Verifica que la propuesta incrementa el patch"""
        initial_version = critical_memory.current_version
        
        proposal = critical_memory.create_change_proposal(
            description="Test",
            justification="Test",
            files_affected=["test.py"],
            diff_content="test",
            author="Test"
        )
        
        # La versión de la propuesta debe ser patch + 1
        assert "1.1.1" in proposal.version


class TestApprovalProcess:
    """Tests del proceso de aprobación"""
    
    def test_approve_pending_proposal(self, critical_memory):
        """Verifica que se puede aprobar una propuesta PENDING"""
        proposal = critical_memory.create_change_proposal(
            description="Test",
            justification="Test",
            files_affected=["test.py"],
            diff_content="test",
            author="Test"
        )
        
        result = critical_memory.approve_change(proposal, "TestDeveloper")
        
        assert result is True
        assert proposal.status == ApprovalStatus.APPROVED
        assert proposal.approved_by == "TestDeveloper"
        assert proposal.approval_timestamp is not None
    
    def test_cannot_approve_already_approved(self, critical_memory):
        """Verifica que no se puede aprobar dos veces"""
        proposal = critical_memory.create_change_proposal(
            description="Test",
            justification="Test",
            files_affected=["test.py"],
            diff_content="test",
            author="Test"
        )
        
        critical_memory.approve_change(proposal, "Dev1")
        result = critical_memory.approve_change(proposal, "Dev2")
        
        assert result is False
        assert proposal.approved_by == "Dev1"  # No cambió
    
    def test_reject_pending_proposal(self, critical_memory):
        """Verifica el rechazo de propuestas"""
        proposal = critical_memory.create_change_proposal(
            description="Bad proposal",
            justification="Will break things",
            files_affected=["test.py"],
            diff_content="bad code",
            author="Test"
        )
        
        result = critical_memory.reject_change(
            proposal,
            "TestDeveloper",
            "Violates safety principles"
        )
        
        assert result is True
        assert proposal.status == ApprovalStatus.REJECTED


class TestAtomicUpdate:
    """Tests del protocolo de actualización atómica"""
    
    def test_cannot_apply_unapproved_proposal(self, critical_memory):
        """Verifica que no se puede aplicar propuesta sin aprobar"""
        proposal = critical_memory.create_change_proposal(
            description="Test",
            justification="Test",
            files_affected=["test.py"],
            diff_content="test",
            author="Test"
        )
        
        result = critical_memory.apply_atomic_update(proposal)
        assert result is False
    
    def test_creates_backup_before_update(self, critical_memory, temp_config):
        """Verifica que se crea backup antes de aplicar cambio"""
        proposal = critical_memory.create_change_proposal(
            description="Test",
            justification="Test",
            files_affected=["test.py"],
            diff_content="test",
            author="Test"
        )
        critical_memory.approve_change(proposal, "Test")
        
        critical_memory.apply_atomic_update(proposal)
        
        # Verificar que hay al menos un backup
        backups_dir = Path(temp_config['system']['storage_root']) / 'backups'
        assert backups_dir.exists()
        backup_dirs = list(backups_dir.iterdir())
        assert len(backup_dirs) >= 1
    
    def test_updates_version_history(self, critical_memory):
        """Verifica que se actualiza el historial de versiones"""
        initial_history_len = len(critical_memory.get_version_history())
        
        proposal = critical_memory.create_change_proposal(
            description="Test",
            justification="Test",
            files_affected=["test.py"],
            diff_content="test",
            author="Test"
        )
        critical_memory.approve_change(proposal, "Test")
        critical_memory.apply_atomic_update(proposal)
        
        new_history = critical_memory.get_version_history()
        assert len(new_history) == initial_history_len + 1
        assert new_history[-1]['change_id'] == proposal.change_id


class TestSystemIntegrity:
    """Tests de verificación de integridad"""
    
    def test_fresh_system_passes_integrity_check(self, critical_memory):
        """Verifica que sistema recién inicializado pasa integridad"""
        assert critical_memory.verify_system_integrity() is True
    
    def test_detects_missing_file(self, critical_memory):
        """Verifica que detecta archivos faltantes"""
        # Eliminar un archivo crítico
        critical_memory.global_state_file.unlink()
        
        assert critical_memory.verify_system_integrity() is False