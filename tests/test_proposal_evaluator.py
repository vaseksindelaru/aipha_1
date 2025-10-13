"""
Tests for ProposalEvaluator - Rule-based evaluation with RAG support.

This module tests the ProposalEvaluator agent functionality including:
- Rule-based evaluation scoring
- RAG integration with knowledge base
- LLM fallback behavior
- Error handling and edge cases
- Integration with RedesignHelper
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from aipha.core.tools.proposal_evaluator import ProposalEvaluator, EvaluationResult
from aipha.core.tools.change_proposer import ChangeProposal
from aipha.core.context_sentinel import ContextSentinel


class TestEvaluationResult:
    """Test the EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Test basic EvaluationResult creation and properties."""
        result = EvaluationResult(
            score=0.75,
            feasibility=0.8,
            impact=0.9,
            risk=0.3,
            justification="Good proposal",
            criteria_used=["feasibility", "impact"]
        )

        assert result.score == 0.75
        assert result.feasibility == 0.8
        assert result.impact == 0.9
        assert result.risk == 0.3
        assert result.justification == "Good proposal"
        assert result.criteria_used == ["feasibility", "impact"]

    def test_evaluation_result_defaults(self):
        """Test EvaluationResult with default values."""
        result = EvaluationResult(
            score=0.6,
            feasibility=0.7,
            impact=0.5,
            risk=0.4,
            justification="Average proposal"
        )

        assert result.criteria_used == []
        assert result.is_approved() is False  # Below default threshold of 0.7

    def test_is_approved_method(self):
        """Test the is_approved method with different thresholds."""
        result = EvaluationResult(
            score=0.8,
            feasibility=0.8,
            impact=0.8,
            risk=0.2,
            justification="Approved proposal"
        )

        assert result.is_approved() is True  # Above default 0.7
        assert result.is_approved(0.9) is False  # Below custom threshold
        assert result.is_approved(0.7) is True  # At threshold


class TestProposalEvaluator:
    """Test the ProposalEvaluator class."""

    @pytest.fixture
    def mock_context_sentinel(self):
        """Create a mock ContextSentinel for testing."""
        mock_cs = Mock(spec=ContextSentinel)

        # Mock get_knowledge_entries to return evaluation criteria
        mock_cs.get_knowledge_entries.return_value = [
            {"id": "crit1", "content": "Feasibility criteria content", "metadata": {"type": "evaluation_criteria"}},
            {"id": "crit2", "content": "Impact criteria content", "metadata": {"type": "evaluation_criteria"}}
        ]

        # Mock llm_query_system as None initially (for fallback testing)
        mock_cs.llm_query_system = None

        return mock_cs

    @pytest.fixture
    def sample_proposal(self):
        """Create a sample ChangeProposal for testing."""
        return ChangeProposal(
            id="test-001",
            title="Test ATR Implementation",
            description="Implement ATR-based dynamic barriers",
            justification="Improve trading accuracy",
            component="aipha/trading_flow/labelers/potential_capture_engine.py",
            params={"atr_period": 20, "tp_multiplier": 5.0, "sl_multiplier": 3.0},
            priority="high",
            estimated_impact="significant"
        )

    @pytest.fixture
    def evaluator(self, temp_config, mock_context_sentinel):
        """Create a ProposalEvaluator instance for testing."""
        return ProposalEvaluator(temp_config, mock_context_sentinel)

    def test_evaluator_initialization(self, temp_config, mock_context_sentinel):
        """Test ProposalEvaluator initialization."""
        evaluator = ProposalEvaluator(temp_config, mock_context_sentinel)

        assert evaluator.config == temp_config
        assert evaluator.context_sentinel == mock_context_sentinel
        assert evaluator.evaluation_weights == {"feasibility": 0.3, "impact": 0.4, "risk": 0.3}

    def test_rule_based_evaluation(self, evaluator, sample_proposal, mock_context_sentinel):
        """Test rule-based evaluation when LLM is not available."""
        # Ensure LLM is not available
        mock_context_sentinel.llm_query_system = None

        result = evaluator.evaluate_proposal(sample_proposal)

        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.feasibility <= 1.0
        assert 0.0 <= result.impact <= 1.0
        assert 0.0 <= result.risk <= 1.0
        assert isinstance(result.justification, str)
        assert len(result.justification) > 0

        # Verify criteria retrieval was called
        mock_context_sentinel.get_knowledge_entries.assert_called_with(
            category="evaluation_criteria",
            limit=10
        )

    @patch('aipha.core.tools.proposal_evaluator.logger')
    def test_llm_evaluation_success(self, mock_logger, evaluator, sample_proposal, mock_context_sentinel):
        """Test successful LLM evaluation."""
        # Mock LLM system
        mock_llm = Mock()
        mock_llm.query.return_value = """
        Feasibility: 0.8
        Impact: 0.9
        Risk: 0.2
        Justification: This is a well-structured proposal with clear benefits.
        """
        mock_context_sentinel.llm_query_system = mock_llm

        result = evaluator.evaluate_proposal(sample_proposal)

        # Verify LLM was called
        mock_llm.query.assert_called_once()
        assert result.feasibility == 0.8
        assert result.impact == 0.9
        assert result.risk == 0.2
        assert "well-structured" in result.justification

        # Verify logging
        mock_logger.info.assert_called()

    def test_llm_evaluation_failure_fallback(self, evaluator, sample_proposal, mock_context_sentinel):
        """Test fallback to rule-based when LLM fails."""
        # Mock LLM that raises exception
        mock_llm = Mock()
        mock_llm.query.side_effect = Exception("API Error")
        mock_context_sentinel.llm_query_system = mock_llm

        result = evaluator.evaluate_proposal(sample_proposal)

        # Should still return valid result using rule-based fallback
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.score <= 1.0
        assert "Rule-based assessment" in result.justification

    def test_parse_llm_response(self, evaluator):
        """Test parsing of LLM responses."""
        # Test valid response
        response = """
        Feasibility: 0.85
        Impact: 0.75
        Risk: 0.3
        Justification: Good technical approach.
        """

        scores = evaluator._parse_llm_response(response)

        assert scores['feasibility'] == 0.85
        assert scores['impact'] == 0.75
        assert scores['risk'] == 0.3

    def test_parse_llm_response_invalid_scores(self, evaluator):
        """Test parsing with invalid scores."""
        response = """
        Feasibility: invalid
        Impact: 0.8
        Risk: 1.5
        Justification: Test.
        """

        scores = evaluator._parse_llm_response(response)

        # Invalid scores should use defaults
        assert scores['feasibility'] == 0.5  # Default
        assert scores['impact'] == 0.8  # Valid
        assert scores['risk'] == 1.0  # Clamped from 1.5 to max 1.0

    def test_calculate_overall_score(self, evaluator):
        """Test overall score calculation."""
        scores = {'feasibility': 0.8, 'impact': 0.9, 'risk': 0.2}

        overall = evaluator._calculate_overall_score(scores)

        # Expected: (0.8 * 0.3) + (0.9 * 0.4) + ((1-0.2) * 0.3) = 0.24 + 0.36 + 0.24 = 0.84
        expected = (0.8 * 0.3) + (0.9 * 0.4) + ((1 - 0.2) * 0.3)
        assert abs(overall - expected) < 0.001

    def test_error_handling(self, evaluator, sample_proposal, mock_context_sentinel):
        """Test error handling in evaluation."""
        # Make get_knowledge_entries raise exception
        mock_context_sentinel.get_knowledge_entries.side_effect = Exception("DB Error")

        result = evaluator.evaluate_proposal(sample_proposal)

        # Should return conservative fallback result
        assert result.score == 0.5
        assert "Evaluation failed" in result.justification

    def test_get_default_criteria(self, evaluator):
        """Test default criteria retrieval."""
        criteria = evaluator._get_default_criteria()

        assert isinstance(criteria, str)
        assert "Feasibility" in criteria
        assert "Impact" in criteria
        assert "Risk" in criteria

    def test_build_justification(self, evaluator, sample_proposal):
        """Test justification building."""
        scores = {'feasibility': 0.8, 'impact': 0.7, 'risk': 0.3}
        llm_response = "Justification: Good proposal with solid technical foundation."

        justification = evaluator._build_justification(sample_proposal, scores, llm_response)

        assert sample_proposal.title in justification
        assert "0.80" in justification  # Feasibility score
        assert "0.70" in justification  # Impact score
        assert "0.30" in justification  # Risk score
        assert "solid technical foundation" in justification


class TestProposalEvaluatorIntegration:
    """Integration tests for ProposalEvaluator with real components."""

    def test_with_real_config_and_context(self, temp_config):
        """Test with real ContextSentinel instance."""
        # Ensure storage directory exists for ContextSentinel
        import os
        storage_root = temp_config['system']['storage_root']
        os.makedirs(storage_root, exist_ok=True)

        # Create real ContextSentinel
        context_sentinel = ContextSentinel(temp_config)

        # Create evaluator
        evaluator = ProposalEvaluator(temp_config, context_sentinel)

        # Create proposal
        proposal = ChangeProposal(
            id="integration-test-001",
            title="Integration Test Proposal",
            description="Test proposal for integration testing",
            justification="Verify integration works",
            component="test_component.py",
            params={"test_param": "value"}
        )

        # Evaluate
        result = evaluator.evaluate_proposal(proposal)

        # Verify result
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.score, float)
        assert isinstance(result.justification, str)
        assert len(result.justification) > 0

    def test_redesign_helper_integration(self, redesign_helper):
        """Test integration with RedesignHelper."""
        # Create a sample proposal for testing
        sample_proposal = ChangeProposal(
            id="integration-test-002",
            title="Integration Test Proposal",
            description="Test proposal for RedesignHelper integration",
            justification="Verify RedesignHelper integration works",
            component="test_component.py",
            params={"test_param": "value"}
        )

        # Evaluate through RedesignHelper
        evaluation = redesign_helper.evaluate_proposal(sample_proposal)

        # Verify evaluation structure
        required_keys = ['approved', 'score', 'feasibility', 'impact', 'risk', 'justification', 'proposal_id']
        for key in required_keys:
            assert key in evaluation

        assert isinstance(evaluation['approved'], bool)
        assert isinstance(evaluation['score'], float)
        assert 0.0 <= evaluation['score'] <= 1.0


# Run basic functionality test if executed directly
if __name__ == "__main__":
    import yaml

    # Load config for testing
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Mock config for testing
        config = {
            'system': {'storage_root': './test_storage'},
            'knowledge_manager': {
                'project_root': './',
                'chroma_persist_dir': './test_chroma',
                'collection_name': 'test_collection',
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        }

    # Create test proposal
    proposal = ChangeProposal(
        id="manual-test-001",
        title="Manual Test Proposal",
        description="Test proposal for manual testing",
        justification="Verify manual test works",
        component="test_component.py",
        params={"test_param": "value"}
    )

    try:
        # Create evaluator
        context_sentinel = ContextSentinel(config)
        evaluator = ProposalEvaluator(config, context_sentinel)

        # Evaluate proposal
        result = evaluator.evaluate_proposal(proposal)

        print("=== Manual Test Result ===")
        print(f"Score: {result.score:.2f}")
        print(f"Feasibility: {result.feasibility:.2f}")
        print(f"Impact: {result.impact:.2f}")
        print(f"Risk: {result.risk:.2f}")
        print(f"Approved: {result.is_approved()}")
        print(f"Justification:\n{result.justification}")
        print("Manual test completed successfully!")

    except Exception as e:
        print(f"Manual test failed: {e}")
        import traceback
        traceback.print_exc()