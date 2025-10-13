import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChangeProposal:
    id: str
    title: str
    description: str
    justification: str
    component: str
    params: Dict[str, Any]

class ChangeProposer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("ChangeProposer inicializado (hardcodeado para ATR).")

    def generate_proposal(self, proposal_type: str) -> ChangeProposal:
        if proposal_type == "ATR":
            return ChangeProposal(
                id="pce-atr-001",
                title="Implementación de Barreras Dinámicas con ATR",
                description="Reemplazar TP/SL fijos por barreras dinámicas basadas en Average True Range.",
                justification="Adaptar el motor a diferentes regímenes de volatilidad para reducir falsas señales y mejorar la captura de beneficios.",
                component="aipha/trading_flow/labelers/potential_capture_engine.py",
                params={"atr_period": 20, "tp_multiplier": 5, "sl_multiplier": 3}
            )
        raise ValueError("Proposal type not supported.")

# Test al final (opcional)
if __name__ == "__main__":
    config = {}  # Mock config
    proposer = ChangeProposer(config)
    proposal = proposer.generate_proposal("ATR")
    print(proposal)