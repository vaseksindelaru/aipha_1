"""
Tests for CodecraftSage - Code generation and implementation agent.

This module tests the CodecraftSage agent functionality including:
- Code generation for ATR implementations
- Test generation for trading engines
- Template-based code creation
- Error handling and validation
- Integration with RedesignHelper
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import os

from aipha.core.tools.codecraft_sage import CodecraftSage, ImplementationResult
from aipha.core.tools.change_proposer import ChangeProposal


class TestImplementationResult:
    """Test the ImplementationResult dataclass."""

    def test_implementation_result_creation(self):
        """Test basic ImplementationResult creation."""
        result = ImplementationResult(
            code="print('hello')",
            test_code="def test_hello(): pass",
            success=True,
            message="Code generated successfully"
        )

        assert result.code == "print('hello')"
        assert result.test_code == "def test_hello(): pass"
        assert result.success is True
        assert result.message == "Code generated successfully"
        assert result.files_modified == []
        assert result.test_files_created == []

    def test_implementation_result_with_files(self):
        """Test ImplementationResult with file lists."""
        result = ImplementationResult(
            code="class Engine: pass",
            test_code="def test_engine(): pass",
            success=True,
            message="Engine created",
            files_modified=["engine.py"],
            test_files_created=["test_engine.py"]
        )

        assert result.files_modified == ["engine.py"]
        assert result.test_files_created == ["test_engine.py"]


class TestCodecraftSage:
    """Test the CodecraftSage class."""

    @pytest.fixture
    def sample_atr_proposal(self):
        """Create a sample ATR proposal for testing."""
        return ChangeProposal(
            id="test-atr-001",
            title="Implementación de Barreras Dinámicas con ATR",
            description="Implement ATR-based dynamic barriers",
            justification="Improve risk management",
            component="aipha/trading_flow/labelers/potential_capture_engine.py",
            params={
                "atr_period": 20,
                "tp_multiplier": 5.0,
                "sl_multiplier": 3.0,
                "time_limit": 15
            },
            priority="high",
            estimated_impact="significant"
        )

    @pytest.fixture
    def sample_generic_proposal(self):
        """Create a generic proposal for testing."""
        return ChangeProposal(
            id="test-generic-001",
            title="Generic Feature Implementation",
            description="Add a generic feature",
            justification="General improvement",
            component="generic_module.py",
            params={"param1": "value1"}
        )

    @pytest.fixture
    def codecraft_sage(self, temp_config):
        """Create a CodecraftSage instance for testing."""
        return CodecraftSage(temp_config)

    def test_codecraft_sage_initialization(self, temp_config, codecraft_sage):
        """Test CodecraftSage initialization."""
        assert codecraft_sage.config == temp_config
        assert isinstance(codecraft_sage.templates, dict)

    def test_implement_atr_proposal(self, codecraft_sage, sample_atr_proposal):
        """Test ATR proposal implementation."""
        result = codecraft_sage.implement_change(sample_atr_proposal)

        # Verify result structure
        assert isinstance(result, ImplementationResult)
        assert result.success is True
        assert "ATR" in result.message
        assert "PotentialCaptureEngine" in result.code
        assert "test_" in result.test_code
        assert len(result.files_modified) > 0
        assert len(result.test_files_created) > 0

    def test_implement_unsupported_proposal(self, codecraft_sage, sample_generic_proposal):
        """Test implementation of unsupported proposal type."""
        result = codecraft_sage.implement_change(sample_generic_proposal)

        assert result.success is False
        assert "unsupported proposal type" in result.message.lower()
        assert result.code == ""
        assert result.test_code == ""

    def test_atr_code_generation(self, codecraft_sage, sample_atr_proposal):
        """Test ATR code generation specifically."""
        result = codecraft_sage.implement_change(sample_atr_proposal)

        # Check that generated code contains expected components
        code = result.code
        assert "class PotentialCaptureEngine" in code
        assert "atr_period" in code
        assert "tp_multiplier" in code
        assert "sl_multiplier" in code
        assert "def label_events" in code
        assert "_calculate_atr" in code

    def test_atr_test_generation(self, codecraft_sage, sample_atr_proposal):
        """Test ATR test generation."""
        result = codecraft_sage.implement_change(sample_atr_proposal)

        # Check that test code contains expected components
        test_code = result.test_code
        assert "class TestPotentialCaptureEngineATR" in test_code
        assert "def test_" in test_code
        assert "PotentialCaptureEngine" in test_code
        assert "label_events" in test_code
        assert "pytest" in test_code

    def test_code_validation_passes(self, codecraft_sage, sample_atr_proposal):
        """Test that generated code passes validation."""
        result = codecraft_sage.implement_change(sample_atr_proposal)

        assert result.success is True
        assert "successfully" in result.message

    def test_error_handling(self, codecraft_sage, sample_atr_proposal):
        """Test error handling in implementation."""
        # Create a proposal that will cause an error
        bad_proposal = ChangeProposal(
            id="bad-proposal",
            title="Bad Proposal",
            description="This will fail",
            justification="Testing error handling",
            component="bad.py",
            params=None  # This might cause issues
        )

        result = codecraft_sage.implement_change(bad_proposal)

        # Should handle gracefully
        assert isinstance(result, ImplementationResult)
        assert result.success is False
        assert "unsupported proposal type" in result.message.lower()

    def test_template_fallback(self, codecraft_sage, sample_atr_proposal):
        """Test fallback to default templates when templates are not available."""
        # Clear templates to force fallback
        original_templates = codecraft_sage.templates
        codecraft_sage.templates = {}

        try:
            result = codecraft_sage.implement_change(sample_atr_proposal)

            # Should still work with default implementations
            assert result.success is True
            assert "PotentialCaptureEngine" in result.code

        finally:
            # Restore templates
            codecraft_sage.templates = original_templates

    def test_parameter_extraction(self, codecraft_sage):
        """Test parameter extraction from proposals."""
        proposal = ChangeProposal(
            id="param-test",
            title="Implementación de Barreras Dinámicas con ATR",
            description="Test params",
            justification="Test",
            component="test.py",
            params={
                "atr_period": 14,
                "tp_multiplier": 3.0,
                "sl_multiplier": 2.0,
                "time_limit": 10
            }
        )

        result = codecraft_sage.implement_change(proposal)

        # Check that parameters are used in generated code
        assert "atr_period=14" in result.code
        assert "tp_multiplier=3.0" in result.code
        assert "sl_multiplier=2.0" in result.code

    def test_generated_code_structure(self, codecraft_sage, sample_atr_proposal):
        """Test that generated code has proper structure."""
        result = codecraft_sage.implement_change(sample_atr_proposal)

        code = result.code

        # Check for proper Python class structure
        assert "import pandas as pd" in code
        assert "import numpy as np" in code
        assert "class PotentialCaptureEngine:" in code
        assert "def __init__(" in code
        assert "def _calculate_atr(" in code
        assert "def label_events(" in code
        assert "def get_parameters(" in code

        # Check for docstrings
        assert '"""' in code

    def test_generated_test_structure(self, codecraft_sage, sample_atr_proposal):
        """Test that generated test code has proper structure."""
        result = codecraft_sage.implement_change(sample_atr_proposal)

        test_code = result.test_code

        # Check for proper test structure
        assert "import pytest" in test_code
        assert "import pandas as pd" in test_code
        assert "class TestPotentialCaptureEngineATR:" in test_code
        assert "def test_engine_initialization(" in test_code
        assert "def test_atr_calculation(" in test_code
        assert "def test_take_profit_hit(" in test_code
        assert "def test_stop_loss_hit(" in test_code
        assert "def test_timeout_scenario(" in test_code


class TestCodecraftSageIntegration:
    """Integration tests for CodecraftSage with other components."""

    def test_with_redesign_helper(self, redesign_helper):
        """Test CodecraftSage integration with RedesignHelper."""
        # Get the CodecraftSage instance from RedesignHelper
        sage = redesign_helper.codecraft_sage

        # Create a proposal
        proposal = redesign_helper.change_proposer.generate_proposal("ATR")

        # Implement using the integrated CodecraftSage
        result = sage.implement_change(proposal)

        # Verify integration works
        assert isinstance(result, ImplementationResult)
        assert result.success is True
        assert "ATR" in result.message

    def test_full_flow_integration(self, redesign_helper):
        """Test the complete flow: propose -> evaluate -> implement."""
        # Generate proposal
        proposal = redesign_helper.change_proposer.generate_proposal("ATR")
        assert proposal is not None

        # Evaluate proposal (may not be approved due to score threshold)
        evaluation = redesign_helper.evaluate_proposal(proposal)
        assert isinstance(evaluation, dict)
        assert 'approved' in evaluation

        # If approved, implement (in this case it won't be due to score)
        if evaluation['approved']:
            implementation = redesign_helper.codecraft_sage.implement_change(proposal)
            assert implementation.success is True
        else:
            # Even if not approved, we can still test implementation directly
            implementation = redesign_helper.codecraft_sage.implement_change(proposal)
            assert implementation.success is True

    def test_knowledge_base_integration(self, redesign_helper):
        """Test that CodecraftSage can access knowledge base templates."""
        # Check that templates are loaded from knowledge base
        sage = redesign_helper.codecraft_sage

        # The templates should be loaded during initialization
        # (even if empty, the mechanism should work)
        assert hasattr(sage, 'templates')
        assert isinstance(sage.templates, dict)


# Manual test function for direct execution
def run_manual_test():
    """Run manual test of CodecraftSage."""
    import yaml
    from pathlib import Path

    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'system': {'storage_root': './test_storage'}}

    try:
        # Create CodecraftSage
        sage = CodecraftSage(config)

        # Create ATR proposal
        from aipha.core.tools.change_proposer import ChangeProposal
        proposal = ChangeProposal(
            id="manual-test-001",
            title="Implementación de Barreras Dinámicas con ATR",
            description="Manual test of ATR implementation",
            justification="Testing CodecraftSage manually",
            component="aipha/trading_flow/labelers/potential_capture_engine.py",
            params={
                "atr_period": 20,
                "tp_multiplier": 5.0,
                "sl_multiplier": 3.0,
                "time_limit": 20
            }
        )

        # Implement
        result = sage.implement_change(proposal)

        print("=== CodecraftSage Manual Test Result ===")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Code length: {len(result.code)} characters")
        print(f"Test code length: {len(result.test_code)} characters")
        print(f"Files to modify: {result.files_modified}")
        print(f"Test files to create: {result.test_files_created}")

        if result.success:
            print("\n=== Sample Generated Code ===")
            lines = result.code.split('\n')
            for i, line in enumerate(lines[:20]):  # First 20 lines
                print("2d")
            if len(lines) > 20:
                print("...")

        print("\nManual test completed successfully!")

    except Exception as e:
        print(f"Manual test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_manual_test()