"""
ChangeProposer - Agent for generating structured change proposals in Aipha system.

This module provides the ChangeProposer agent that generates formal ChangeProposal
objects for system improvements. Currently specialized for ATR-based trading improvements.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

@dataclass
class ChangeProposal:
    """
    Structured representation of a system change proposal.

    Attributes:
        id (str): Unique identifier for the proposal.
        title (str): Descriptive title of the proposed change.
        description (str): Detailed description of what the change entails.
        justification (str): Rationale for why this change is beneficial.
        component (str): Target component/module path for implementation.
        params (Dict[str, Any]): Configuration parameters for the change.
        created_at (str): ISO timestamp of proposal creation.
        priority (str): Priority level ("low", "medium", "high", "critical").
        estimated_impact (str): Expected impact assessment.
    """
    id: str
    title: str
    description: str
    justification: str
    component: str
    params: Dict[str, Any]
    created_at: str = ""
    priority: str = "medium"
    estimated_impact: str = "moderate"

    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

class ChangeProposer:
    """
    Agent responsible for generating structured change proposals for system improvements.

    Currently implements hardcoded proposals for ATR-based trading enhancements.
    Designed to be extensible for dynamic proposal generation in future versions.

    Attributes:
        config (Dict[str, Any]): System configuration dictionary.
        supported_types (List[str]): List of proposal types this agent can generate.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ChangeProposer agent.

        Args:
            config (Dict[str, Any]): System configuration containing proposal settings.

        Side effects:
            - Logs initialization message.
            - Validates configuration if needed.
        """
        self.config = config
        self.supported_types = ["ATR"]  # Expandable for future proposal types
        logger.info("ChangeProposer inicializado (hardcodeado para ATR).")

    def generate_proposal(self, proposal_type: str, **kwargs) -> ChangeProposal:
        """
        Generate a structured change proposal for the specified type.

        Args:
            proposal_type (str): Type of proposal to generate (currently "ATR").
            **kwargs: Additional parameters for proposal customization.

        Returns:
            ChangeProposal: Structured proposal object with all required fields.

        Raises:
            ValueError: If proposal_type is not supported.
            RuntimeError: If proposal generation fails.

        Example:
            >>> proposer = ChangeProposer(config)
            >>> proposal = proposer.generate_proposal("ATR")
            >>> print(proposal.title)
            Implementación de Barreras Dinámicas con ATR
        """
        if proposal_type not in self.supported_types:
            available = ", ".join(self.supported_types)
            raise ValueError(f"Proposal type '{proposal_type}' not supported. Available: {available}")

        try:
            if proposal_type == "ATR":
                return self._generate_atr_proposal(**kwargs)
            else:
                # Future expansion point for dynamic proposal generation
                raise ValueError(f"Unsupported proposal type: {proposal_type}")
        except Exception as e:
            logger.error(f"Failed to generate {proposal_type} proposal: {e}")
            raise RuntimeError(f"Proposal generation failed: {e}")

    def _generate_atr_proposal(self, **kwargs) -> ChangeProposal:
        """
        Generate ATR-based dynamic barriers proposal.

        Args:
            **kwargs: Override parameters (atr_period, tp_multiplier, sl_multiplier).

        Returns:
            ChangeProposal: ATR trading improvement proposal.
        """
        # Allow parameter overrides for flexibility
        atr_period = kwargs.get('atr_period', 20)
        tp_multiplier = kwargs.get('tp_multiplier', 5.0)
        sl_multiplier = kwargs.get('sl_multiplier', 3.0)

        # Validate parameters
        if not isinstance(atr_period, int) or atr_period <= 0:
            raise ValueError("atr_period must be a positive integer")
        if not all(isinstance(x, (int, float)) and x > 0 for x in [tp_multiplier, sl_multiplier]):
            raise ValueError("multipliers must be positive numbers")

        proposal_id = f"pce-atr-{uuid.uuid4().hex[:6]}"

        return ChangeProposal(
            id=proposal_id,
            title="Implementación de Barreras Dinámicas con ATR",
            description="Reemplazar TP/SL fijos por barreras dinámicas basadas en Average True Range para adaptación automática a volatilidad del mercado.",
            justification="Adaptar el motor de trading a diferentes regímenes de volatilidad para reducir señales falsas y mejorar la captura de beneficios en mercados dinámicos.",
            component="aipha/trading_flow/labelers/potential_capture_engine.py",
            params={
                "atr_period": atr_period,
                "tp_multiplier": tp_multiplier,
                "sl_multiplier": sl_multiplier,
                "enable_dynamic_barriers": True
            },
            priority="high",
            estimated_impact="significant"
        )

    def validate_proposal(self, proposal: ChangeProposal) -> bool:
        """
        Validate a change proposal for completeness and correctness.

        Args:
            proposal (ChangeProposal): Proposal to validate.

        Returns:
            bool: True if proposal is valid, False otherwise.

        Side effects:
            - Logs validation issues if any.
        """
        required_fields = ['id', 'title', 'description', 'justification', 'component', 'params']
        missing_fields = [field for field in required_fields if not getattr(proposal, field, None)]

        if missing_fields:
            logger.warning(f"Proposal validation failed: missing fields {missing_fields}")
            return False

        if not isinstance(proposal.params, dict):
            logger.warning("Proposal validation failed: params must be a dictionary")
            return False

        logger.info(f"Proposal {proposal.id} validated successfully")
        return True

    def list_supported_types(self) -> list[str]:
        """
        Get list of supported proposal types.

        Returns:
            List[str]: Available proposal types.
        """
        return self.supported_types.copy()

# Test al final (opcional)
if __name__ == "__main__":
    config = {}  # Mock config
    proposer = ChangeProposer(config)
    proposal = proposer.generate_proposal("ATR")
    print(proposal)