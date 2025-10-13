"""
ProposalEvaluator - Agent for evaluating change proposals using rule-based criteria with RAG support.

This module provides the ProposalEvaluator agent that assesses change proposals based on
feasibility, impact, and risk criteria. It combines rule-based evaluation with contextual
information retrieved via RAG from the knowledge base.
"""

import logging
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from ..context_sentinel import ContextSentinel
from .change_proposer import ChangeProposal

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """
    Structured result of proposal evaluation.

    Attributes:
        score (float): Overall evaluation score (0.0 to 1.0, higher is better).
        feasibility (float): Feasibility score (0.0 to 1.0).
        impact (float): Expected impact score (0.0 to 1.0).
        risk (float): Risk assessment (0.0 to 1.0, lower is better).
        justification (str): Detailed justification for the evaluation.
        criteria_used (list): List of evaluation criteria applied.
    """
    score: float
    feasibility: float
    impact: float
    risk: float
    justification: str
    criteria_used: list[str] = None

    def __post_init__(self):
        """Initialize criteria_used if not provided."""
        if self.criteria_used is None:
            self.criteria_used = []

    def is_approved(self, threshold: float = 0.7) -> bool:
        """
        Check if proposal meets approval threshold.

        Args:
            threshold (float): Minimum score required for approval. Defaults to 0.7.

        Returns:
            bool: True if proposal is approved based on score.
        """
        return self.score >= threshold

class ProposalEvaluator:
    """
    Agent responsible for evaluating change proposals using rule-based criteria with RAG support.

    This evaluator combines structured rule-based assessment with contextual information
    retrieved from the knowledge base to provide comprehensive proposal evaluations.

    Attributes:
        config (Dict[str, Any]): System configuration dictionary.
        context_sentinel (ContextSentinel): Context sentinel for knowledge retrieval.
        evaluation_weights (Dict[str, float]): Weights for scoring components.
    """

    def __init__(self, config: Dict[str, Any], context_sentinel: ContextSentinel):
        """
        Initialize the ProposalEvaluator agent.

        Args:
            config (Dict[str, Any]): System configuration containing evaluation settings.
            context_sentinel (ContextSentinel): Context sentinel instance for knowledge access.

        Side effects:
            - Logs initialization message.
        """
        self.config = config
        self.context_sentinel = context_sentinel

        # Default weights for scoring components
        self.evaluation_weights = {
            'feasibility': 0.3,
            'impact': 0.4,
            'risk': 0.3  # Risk is inverted (lower risk = higher score)
        }

        logger.info("ProposalEvaluator inicializado (basado en reglas con RAG).")

    def evaluate_proposal(self, proposal: ChangeProposal) -> EvaluationResult:
        """
        Evaluate a change proposal using rule-based criteria with RAG contextualization.

        Retrieves evaluation criteria from the knowledge base and uses LLM for contextual
        assessment, then combines with rule-based scoring.

        Args:
            proposal (ChangeProposal): The proposal to evaluate.

        Returns:
            EvaluationResult: Structured evaluation result with scores and justification.

        Side effects:
            - Queries knowledge base for evaluation criteria.
            - May call LLM for contextual assessment if available.
            - Logs evaluation process and results.
        """
        try:
            # Retrieve evaluation criteria from knowledge base
            criteria_docs = self.context_sentinel.get_knowledge_entries(
                category="evaluation_criteria",
                limit=10
            )

            criteria_context = ""
            criteria_used = []

            if criteria_docs:
                criteria_context = "\n".join([doc.get('content', '') for doc in criteria_docs])
                criteria_used = [doc.get('id', '') for doc in criteria_docs if doc.get('id')]
                logger.info(f"Retrieved {len(criteria_docs)} evaluation criteria documents")
            else:
                logger.warning("No evaluation criteria found in knowledge base, using defaults")
                criteria_context = self._get_default_criteria()
                criteria_used = ["default_criteria"]

            # Use LLM for contextual evaluation if available
            llm_assessment = self._get_llm_assessment(proposal, criteria_context)

            # Parse LLM response to extract scores
            scores = self._parse_llm_response(llm_assessment)

            # Calculate overall score using weighted average
            overall_score = self._calculate_overall_score(scores)

            # Create comprehensive justification
            justification = self._build_justification(proposal, scores, llm_assessment)

            result = EvaluationResult(
                score=overall_score,
                feasibility=scores.get('feasibility', 0.5),
                impact=scores.get('impact', 0.5),
                risk=scores.get('risk', 0.5),
                justification=justification,
                criteria_used=criteria_used
            )

            logger.info(f"Proposal {proposal.id} evaluated: score={result.score:.2f}, "
                       f"feasibility={result.feasibility:.2f}, impact={result.impact:.2f}, "
                       f"risk={result.risk:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error evaluating proposal {proposal.id}: {e}")
            # Return conservative default evaluation on error
            return EvaluationResult(
                score=0.5,
                feasibility=0.5,
                impact=0.5,
                risk=0.5,
                justification=f"Evaluation failed due to error: {str(e)}. Manual review required.",
                criteria_used=["error_fallback"]
            )

    def _get_llm_assessment(self, proposal: ChangeProposal, criteria_context: str) -> str:
        """
        Get LLM assessment of the proposal using RAG context.

        Args:
            proposal (ChangeProposal): Proposal to assess.
            criteria_context (str): Retrieved evaluation criteria context.

        Returns:
            str: LLM response with assessment.
        """
        if not hasattr(self.context_sentinel, 'llm_query_system') or not self.context_sentinel.llm_query_system:
            logger.warning("LLM system not available, using rule-based evaluation only")
            return self._rule_based_assessment(proposal)

        prompt = f"""
        Evaluate the following change proposal based on the provided evaluation criteria:

        PROPOSAL:
        Title: {proposal.title}
        Description: {proposal.description}
        Justification: {proposal.justification}
        Component: {proposal.component}
        Parameters: {proposal.params}
        Priority: {getattr(proposal, 'priority', 'medium')}
        Estimated Impact: {getattr(proposal, 'estimated_impact', 'moderate')}

        EVALUATION CRITERIA:
        {criteria_context}

        Please provide scores for:
        - Feasibility (0.0-1.0): How technically feasible is this change?
        - Impact (0.0-1.0): What is the expected positive impact?
        - Risk (0.0-1.0): What is the risk level? (0.0 = no risk, 1.0 = high risk)

        Format your response as:
        Feasibility: [score]
        Impact: [score]
        Risk: [score]
        Justification: [detailed explanation]
        """

        try:
            response = self.context_sentinel.llm_query_system.query(prompt)
            logger.info("LLM assessment completed successfully")
            return response
        except Exception as e:
            logger.warning(f"LLM assessment failed: {e}, falling back to rule-based evaluation")
            return self._rule_based_assessment(proposal)

    def _rule_based_assessment(self, proposal: ChangeProposal) -> str:
        """
        Provide rule-based assessment when LLM is not available.

        Args:
            proposal (ChangeProposal): Proposal to assess.

        Returns:
            str: Formatted assessment string.
        """
        # Simple rule-based scoring based on proposal attributes
        feasibility = 0.8 if len(proposal.params) > 0 else 0.6
        impact = 0.7 if getattr(proposal, 'priority', 'medium') == 'high' else 0.5
        risk = 0.3 if 'test' in proposal.component.lower() else 0.5

        return f"""
        Feasibility: {feasibility}
        Impact: {impact}
        Risk: {risk}
        Justification: Rule-based assessment - {proposal.title} appears technically feasible with moderate impact and acceptable risk.
        """

    def _parse_llm_response(self, response: str) -> Dict[str, float]:
        """
        Parse LLM response to extract numerical scores.

        Args:
            response (str): Raw LLM response.

        Returns:
            Dict[str, float]: Parsed scores with defaults.
        """
        scores = {
            'feasibility': 0.5,
            'impact': 0.5,
            'risk': 0.5
        }

        try:
            # Extract scores using regex patterns
            patterns = {
                'feasibility': r'Feasibility:\s*([0-9.]+)',
                'impact': r'Impact:\s*([0-9.]+)',
                'risk': r'Risk:\s*([0-9.]+)'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        # Ensure score is within valid range
                        scores[key] = max(0.0, min(1.0, score))
                    except ValueError:
                        logger.warning(f"Could not parse {key} score from: {match.group(1)}")

            logger.info(f"Parsed scores: {scores}")
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}, using default scores")

        return scores

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate overall evaluation score using weighted components.

        Args:
            scores (Dict[str, float]): Individual component scores.

        Returns:
            float: Weighted overall score.
        """
        feasibility_score = scores.get('feasibility', 0.5)
        impact_score = scores.get('impact', 0.5)
        risk_score = scores.get('risk', 0.5)

        # Risk is inverted: lower risk = higher contribution
        risk_contribution = 1.0 - risk_score

        overall = (
            feasibility_score * self.evaluation_weights['feasibility'] +
            impact_score * self.evaluation_weights['impact'] +
            risk_contribution * self.evaluation_weights['risk']
        )

        return round(overall, 3)

    def _build_justification(self, proposal: ChangeProposal, scores: Dict[str, float],
                           llm_response: str) -> str:
        """
        Build comprehensive justification from scores and LLM response.

        Args:
            proposal (ChangeProposal): The evaluated proposal.
            scores (Dict[str, float]): Parsed scores.
            llm_response (str): Raw LLM response.

        Returns:
            str: Formatted justification.
        """
        justification_parts = [
            f"Proposal '{proposal.title}' evaluation:",
            f"- Feasibility: {scores.get('feasibility', 0.5):.2f} (technical implementation ease)",
            f"- Impact: {scores.get('impact', 0.5):.2f} (expected benefits)",
            f"- Risk: {scores.get('risk', 0.5):.2f} (potential negative consequences)",
            f"- Overall Score: {self._calculate_overall_score(scores):.2f}"
        ]

        # Extract justification from LLM response if available
        if "Justification:" in llm_response:
            llm_justification = llm_response.split("Justification:", 1)[1].strip()
            if llm_justification:
                justification_parts.append(f"\nDetailed Analysis: {llm_justification}")

        return "\n".join(justification_parts)

    def _get_default_criteria(self) -> str:
        """
        Provide default evaluation criteria when none are available in knowledge base.

        Returns:
            str: Default evaluation criteria text.
        """
        return """
        Default Evaluation Criteria:
        - Feasibility: Technical complexity, resource requirements, implementation timeline
        - Impact: Business value, user experience improvement, system performance gains
        - Risk: Potential for bugs, system instability, security vulnerabilities, rollback difficulty
        - Overall: Balance of benefits vs. costs and risks
        """

# Test functionality
if __name__ == "__main__":
    import yaml
    from pathlib import Path

    # Load config for testing
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
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

    # Create mock context sentinel for testing
    from ..context_sentinel import ContextSentinel

    try:
        context_sentinel = ContextSentinel(config)

        # Create evaluator
        evaluator = ProposalEvaluator(config, context_sentinel)

        # Create mock proposal
        from .change_proposer import ChangeProposal
        mock_proposal = ChangeProposal(
            id="test-001",
            title="Test ATR Implementation",
            description="Implement ATR-based dynamic barriers",
            justification="Improve trading accuracy",
            component="aipha/trading_flow/labelers/potential_capture_engine.py",
            params={"atr_period": 20, "tp_multiplier": 5.0, "sl_multiplier": 3.0},
            priority="high",
            estimated_impact="significant"
        )

        # Evaluate proposal
        result = evaluator.evaluate_proposal(mock_proposal)

        print("=== Proposal Evaluation Result ===")
        print(f"Overall Score: {result.score:.2f}")
        print(f"Feasibility: {result.feasibility:.2f}")
        print(f"Impact: {result.impact:.2f}")
        print(f"Risk: {result.risk:.2f}")
        print(f"Approved: {result.is_approved()}")
        print(f"Justification:\n{result.justification}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()