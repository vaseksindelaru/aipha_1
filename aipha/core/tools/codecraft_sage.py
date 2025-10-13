"""
CodecraftSage - Agent for implementing code changes and generating tests.

This module provides the CodecraftSage agent that implements approved change proposals
by generating code modifications and corresponding tests. It uses template-based code
generation with configurable parameters.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import re

try:
    from .change_proposer import ChangeProposal
except ImportError:
    # For direct execution
    from aipha.core.tools.change_proposer import ChangeProposal

logger = logging.getLogger(__name__)

@dataclass
class ImplementationResult:
    """
    Result of a code implementation attempt.

    Attributes:
        code (str): Generated implementation code.
        test_code (str): Generated test code.
        success (bool): Whether implementation was successful.
        message (str): Human-readable status message.
        files_modified (list): List of files that would be modified.
        test_files_created (list): List of test files that would be created.
    """
    code: str
    test_code: str
    success: bool
    message: str
    files_modified: Optional[list[str]] = None
    test_files_created: Optional[list[str]] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.files_modified is None:
            self.files_modified = []
        if self.test_files_created is None:
            self.test_files_created = []


class CodecraftSage:
    """
    Agent responsible for implementing approved change proposals.

    This agent generates code modifications and tests based on approved proposals.
    It uses template-based generation with configurable parameters to create
    production-ready code implementations.

    Attributes:
        config (Dict[str, Any]): System configuration dictionary.
        templates (Dict[str, str]): Code templates for different implementation types.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CodecraftSage agent.

        Args:
            config (Dict[str, Any]): System configuration containing implementation settings.

        Side effects:
            - Loads code templates.
            - Logs initialization message.
        """
        self.config = config
        self.templates = self._load_templates()

        logger.info("CodecraftSage inicializado (simulado para ATR con templates).")

    def implement_change(self, proposal: ChangeProposal) -> ImplementationResult:
        """
        Implement a change proposal by generating code and tests.

        Args:
            proposal (ChangeProposal): The approved proposal to implement.

        Returns:
            ImplementationResult: Result containing generated code, tests, and status.

        Side effects:
            - Logs implementation process.
            - May validate generated code structure.
        """
        try:
            if "Implementación de Barreras Dinámicas con ATR" in proposal.title:
                return self._implement_atr_bars(proposal)
            elif "ATR" in proposal.title.upper():
                return self._implement_atr_bars(proposal)
            else:
                return ImplementationResult(
                    code="",
                    test_code="",
                    success=False,
                    message=f"Unsupported proposal type: {proposal.title}",
                    files_modified=[],
                    test_files_created=[]
                )

        except Exception as e:
            logger.error(f"Error implementing proposal {proposal.id}: {e}")
            return ImplementationResult(
                code="",
                test_code="",
                success=False,
                message=f"Implementation failed: {str(e)}",
                files_modified=[],
                test_files_created=[]
            )

    def _implement_atr_bars(self, proposal: ChangeProposal) -> ImplementationResult:
        """
        Implement ATR-based dynamic barriers for PotentialCaptureEngine.

        Args:
            proposal (ChangeProposal): ATR proposal to implement.

        Returns:
            ImplementationResult: Complete implementation with code and tests.
        """
        # Extract parameters from proposal
        params = proposal.params or {}
        atr_period = params.get('atr_period', 20)
        tp_multiplier = params.get('tp_multiplier', 5.0)
        sl_multiplier = params.get('sl_multiplier', 3.0)
        time_limit = params.get('time_limit', 20)

        # Generate implementation code
        code = self._generate_atr_implementation_code(
            atr_period=atr_period,
            tp_multiplier=tp_multiplier,
            sl_multiplier=sl_multiplier,
            time_limit=time_limit
        )

        # Generate test code
        test_code = self._generate_atr_test_code(
            atr_period=atr_period,
            tp_multiplier=tp_multiplier,
            sl_multiplier=sl_multiplier
        )

        # Validate generated code structure
        if self._validate_generated_code(code, test_code):
            return ImplementationResult(
                code=code,
                test_code=test_code,
                success=True,
                message="ATR dynamic barriers implementation generated successfully with comprehensive tests.",
                files_modified=["aipha/trading_flow/labelers/potential_capture_engine.py"],
                test_files_created=["tests/test_potential_capture_engine_atr.py"]
            )
        else:
            return ImplementationResult(
                code=code,
                test_code=test_code,
                success=False,
                message="Generated code failed validation checks.",
                files_modified=[],
                test_files_created=[]
            )

    def _generate_atr_implementation_code(self, atr_period: int, tp_multiplier: float,
                                        sl_multiplier: float, time_limit: int) -> str:
        """
        Generate ATR implementation code using template.

        Args:
            atr_period (int): ATR calculation period.
            tp_multiplier (float): Take profit multiplier.
            sl_multiplier (float): Stop loss multiplier.
            time_limit (int): Maximum bars to hold position.

        Returns:
            str: Generated implementation code.
        """
        template = self.templates.get('atr_implementation', '')
        if not template:
            return self._get_default_atr_code(atr_period, tp_multiplier, sl_multiplier, time_limit)

        # Fill template with parameters
        code = template.format(
            atr_period=atr_period,
            tp_multiplier=tp_multiplier,
            sl_multiplier=sl_multiplier,
            time_limit=time_limit
        )

        return code

    def _generate_atr_test_code(self, atr_period: int, tp_multiplier: float, sl_multiplier: float) -> str:
        """
        Generate comprehensive test code for ATR implementation.

        Args:
            atr_period (int): ATR calculation period.
            tp_multiplier (float): Take profit multiplier.
            sl_multiplier (float): Stop loss multiplier.

        Returns:
            str: Generated test code.
        """
        template = self.templates.get('atr_test', '')
        if not template:
            return self._get_default_atr_test_code(atr_period, tp_multiplier, sl_multiplier)

        # Fill template with parameters
        test_code = template.format(
            atr_period=atr_period,
            tp_multiplier=tp_multiplier,
            sl_multiplier=sl_multiplier
        )

        return test_code

    def _validate_generated_code(self, code: str, test_code: str) -> bool:
        """
        Validate that generated code has required structure.

        Args:
            code (str): Implementation code to validate.
            test_code (str): Test code to validate.

        Returns:
            bool: True if code passes basic validation.
        """
        try:
            # Check for required components in implementation code
            required_impl = [
                'class PotentialCaptureEngine',
                'def __init__',
                'def label_events',
                'atr_period',
                'tp_multiplier',
                'sl_multiplier'
            ]

            for req in required_impl:
                if req not in code:
                    logger.warning(f"Missing required component in implementation: {req}")
                    return False

            # Check for required components in test code
            required_test = [
                'def test_',
                'PotentialCaptureEngine',
                'label_events',
                'assert'
            ]

            for req in required_test:
                if req not in test_code:
                    logger.warning(f"Missing required component in test: {req}")
                    return False

            # Check for basic syntax (very basic check)
            if code.count('def ') < 2:  # At least __init__ and label_events
                logger.warning("Implementation code missing required methods")
                return False

            if test_code.count('def test_') < 1:
                logger.warning("Test code missing test functions")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating generated code: {e}")
            return False

    def _load_templates(self) -> Dict[str, str]:
        """
        Load code templates from configuration or use defaults.

        Returns:
            Dict[str, str]: Dictionary of code templates.
        """
        # In a real implementation, these would come from config files
        # For now, return empty dict to use default implementations
        return {}

    def _get_default_atr_code(self, atr_period: int, tp_multiplier: float,
                            sl_multiplier: float, time_limit: int) -> str:
        """
        Get default ATR implementation code.

        Args:
            atr_period (int): ATR calculation period.
            tp_multiplier (float): Take profit multiplier.
            sl_multiplier (float): Stop loss multiplier.
            time_limit (int): Maximum bars to hold position.

        Returns:
            str: Default ATR implementation code.
        """
        return f'''
import pandas as pd
import numpy as np

class PotentialCaptureEngine:
    """
    Potential Capture Engine with ATR-based dynamic barriers.

    This engine implements dynamic take-profit and stop-loss levels based on
    Average True Range (ATR) volatility measurements.
    """

    def __init__(self, atr_period: int = {atr_period}, tp_multiplier: float = {tp_multiplier},
                 sl_multiplier: float = {sl_multiplier}, time_limit: int = {time_limit}, **kwargs):
        """
        Initialize the Potential Capture Engine with ATR parameters.

        Args:
            atr_period (int): Period for ATR calculation.
            tp_multiplier (float): Multiplier for take-profit level (ATR * multiplier).
            sl_multiplier (float): Multiplier for stop-loss level (ATR * multiplier).
            time_limit (int): Maximum bars to hold position.
        """
        self.atr_period = atr_period
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.time_limit = time_limit

        # Validate parameters
        if atr_period <= 0:
            raise ValueError("ATR period must be positive")
        if tp_multiplier <= 0 or sl_multiplier <= 0:
            raise ValueError("Multipliers must be positive")
        if time_limit <= 0:
            raise ValueError("Time limit must be positive")

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            high (pd.Series): High prices.
            low (pd.Series): Low prices.
            close (pd.Series): Close prices.

        Returns:
            pd.Series: ATR values.
        """
        # True Range calculation
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR as rolling mean of TR
        atr = tr.rolling(window=self.atr_period, min_periods=1).mean()
        return atr

    def label_events(self, prices: pd.DataFrame, t_events: pd.Series) -> pd.Series:
        """
        Label trading events with ATR-based dynamic barriers.

        Args:
            prices (pd.DataFrame): Price data with 'high', 'low', 'close' columns.
            t_events (pd.Series): Entry timestamps.

        Returns:
            pd.Series: Labels (-1: loss, 0: timeout, 1: profit).
        """
        if not all(col in prices.columns for col in ['high', 'low', 'close']):
            raise ValueError("Price data must contain 'high', 'low', 'close' columns")

        df = prices.copy()

        # Calculate ATR
        df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])

        # Filter valid events
        valid_events = t_events.dropna().unique()
        valid_events = pd.Series(valid_events)[pd.Series(valid_events).isin(df.index)]

        if len(valid_events) == 0:
            return pd.Series([], dtype=int)

        # Initialize labels
        labels = pd.Series(0, index=valid_events, dtype=int)

        for t0 in valid_events:
            try:
                entry_price = df.loc[t0, 'close']
                atr_value = df.loc[t0, 'atr']

                if pd.isna(atr_value) or atr_value <= 0:
                    continue  # Skip if ATR not available

                # Calculate dynamic barriers
                sl_level = entry_price - (self.sl_multiplier * atr_value)
                tp_level = entry_price + (self.tp_multiplier * atr_value)

                # Define analysis window
                start_idx = df.index.get_loc(t0)
                end_idx = min(start_idx + self.time_limit + 1, len(df))
                path = df.iloc[start_idx + 1:end_idx]

                if len(path) == 0:
                    continue

                # Check exit conditions
                outcome = 0  # Default: timeout

                for t1, row in path.iterrows():
                    # Check take profit
                    if row['high'] >= tp_level:
                        outcome = 1  # Profit
                        break
                    # Check stop loss
                    elif row['low'] <= sl_level:
                        outcome = -1  # Loss
                        break

                labels[t0] = outcome

            except Exception as e:
                logger.warning(f"Error processing event at {t0}: {e}")
                continue

        return labels

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current engine parameters.

        Returns:
            Dict[str, Any]: Current configuration parameters.
        """
        return {{
            'atr_period': self.atr_period,
            'tp_multiplier': self.tp_multiplier,
            'sl_multiplier': self.sl_multiplier,
            'time_limit': self.time_limit
        }}
'''

    def _get_default_atr_test_code(self, atr_period: int, tp_multiplier: float, sl_multiplier: float) -> str:
        """
        Get default ATR test code.

        Args:
            atr_period (int): ATR calculation period.
            tp_multiplier (float): Take profit multiplier.
            sl_multiplier (float): Stop loss multiplier.

        Returns:
            str: Default ATR test code.
        """
        return f'''
import pytest
import pandas as pd
import numpy as np
from aipha.trading_flow.labelers.potential_capture_engine import PotentialCaptureEngine


class TestPotentialCaptureEngineATR:
    """Test suite for ATR-based Potential Capture Engine."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data with trends and volatility
        base_price = 100.0
        prices = []
        for i in range(20):
            # Add some trend and noise
            trend = i * 0.1
            noise = np.random.normal(0, 2.0)
            close = base_price + trend + noise

            # Generate high/low around close
            volatility = abs(np.random.normal(0, 1.0))
            high = close + volatility
            low = close - volatility

            prices.append({{
                'close': round(close, 2),
                'high': round(high, 2),
                'low': round(low, 2)
            }})

        return pd.DataFrame(prices, index=dates)

    @pytest.fixture
    def engine(self):
        """Create engine instance with test parameters."""
        return PotentialCaptureEngine(
            atr_period={atr_period},
            tp_multiplier={tp_multiplier},
            sl_multiplier={sl_multiplier},
            time_limit=10
        )

    def test_engine_initialization(self, engine):
        """Test engine initializes with correct parameters."""
        params = engine.get_parameters()

        assert params['atr_period'] == {atr_period}
        assert params['tp_multiplier'] == {tp_multiplier}
        assert params['sl_multiplier'] == {sl_multiplier}
        assert params['time_limit'] == 10

    def test_atr_calculation(self, engine, sample_prices):
        """Test ATR calculation."""
        atr = engine._calculate_atr(sample_prices['high'], sample_prices['low'], sample_prices['close'])

        assert len(atr) == len(sample_prices)
        assert not atr.isna().all()  # Should have some valid values
        assert (atr >= 0).all()  # ATR should be non-negative

    def test_take_profit_hit(self, engine):
        """Test scenario where take profit is hit."""
        # Create price data where TP is hit quickly
        prices = pd.DataFrame({{
            'close': [100.0, 100.0, 100.0],
            'high': [100.0, 120.0, 120.0],  # TP hit on second bar
            'low': [95.0, 95.0, 95.0]
        }}, index=pd.date_range('2020-01-01', periods=3))

        t_events = pd.Series([pd.Timestamp('2020-01-01')])

        # Mock ATR calculation to return predictable value
        engine._calculate_atr = lambda h, l, c: pd.Series([5.0, 5.0, 5.0], index=prices.index)

        labels = engine.label_events(prices, t_events)

        assert labels.iloc[0] == 1  # Should be profit (TP hit)

    def test_stop_loss_hit(self, engine):
        """Test scenario where stop loss is hit."""
        # Create price data where SL is hit
        prices = pd.DataFrame({{
            'close': [100.0, 100.0, 100.0],
            'high': [105.0, 105.0, 105.0],
            'low': [85.0, 85.0, 85.0]  # SL hit on first bar
        }}, index=pd.date_range('2020-01-01', periods=3))

        t_events = pd.Series([pd.Timestamp('2020-01-01')])

        # Mock ATR calculation
        engine._calculate_atr = lambda h, l, c: pd.Series([5.0, 5.0, 5.0], index=prices.index)

        labels = engine.label_events(prices, t_events)

        assert labels.iloc[0] == -1  # Should be loss (SL hit)

    def test_timeout_scenario(self, engine):
        """Test scenario where position times out without hitting TP/SL."""
        # Create price data that stays within range
        prices = pd.DataFrame({{
            'close': [100.0] * 15,  # Same price throughout
            'high': [102.0] * 15,   # Never hits TP
            'low': [98.0] * 15      # Never hits SL
        }}, index=pd.date_range('2020-01-01', periods=15))

        t_events = pd.Series([pd.Timestamp('2020-01-01')])

        # Mock ATR calculation
        engine._calculate_atr = lambda h, l, c: pd.Series([2.0] * 15, index=prices.index)

        labels = engine.label_events(prices, t_events)

        assert labels.iloc[0] == 0  # Should be timeout

    def test_invalid_price_data(self, engine):
        """Test handling of invalid price data."""
        # Missing required columns
        invalid_prices = pd.DataFrame({{
            'close': [100.0, 101.0],
            'high': [101.0, 102.0]
            # Missing 'low' column
        }})

        t_events = pd.Series([pd.Timestamp('2020-01-01')])

        with pytest.raises(ValueError, match="must contain"):
            engine.label_events(invalid_prices, t_events)

    def test_parameter_validation(self):
        """Test parameter validation in initialization."""
        # Invalid ATR period
        with pytest.raises(ValueError, match="ATR period must be positive"):
            PotentialCaptureEngine(atr_period=0)

        # Invalid multipliers
        with pytest.raises(ValueError, match="Multipliers must be positive"):
            PotentialCaptureEngine(tp_multiplier=-1.0)

        # Invalid time limit
        with pytest.raises(ValueError, match="Time limit must be positive"):
            PotentialCaptureEngine(time_limit=0)

    def test_empty_events(self, engine, sample_prices):
        """Test handling of empty events series."""
        empty_events = pd.Series([], dtype=object)
        labels = engine.label_events(sample_prices, empty_events)

        assert len(labels) == 0
        assert labels.dtype == int

    def test_events_outside_price_range(self, engine, sample_prices):
        """Test handling of events outside price data range."""
        # Event before price data starts
        past_event = pd.Series([pd.Timestamp('2019-01-01')])
        labels = engine.label_events(sample_prices, past_event)

        assert len(labels) == 0  # Should be filtered out

    @pytest.mark.parametrize("atr_period,tp_mult,sl_mult", [
        (5, 2.0, 1.0),
        (20, 3.0, 2.0),
        (50, 4.0, 1.5)
    ])
    def test_different_parameters(self, atr_period, tp_mult, sl_mult):
        """Test engine with different parameter combinations."""
        engine = PotentialCaptureEngine(
            atr_period=atr_period,
            tp_multiplier=tp_mult,
            sl_multiplier=sl_mult
        )

        params = engine.get_parameters()

        assert params['atr_period'] == atr_period
        assert params['tp_multiplier'] == tp_mult
        assert params['sl_multiplier'] == sl_mult
'''


# Test functionality
if __name__ == "__main__":
    import yaml
    from pathlib import Path
    import sys
    import os

    # Add parent directories to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Load config for testing
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Mock config for testing
        config = {
            'system': {'storage_root': './test_storage'}
        }

    try:
        # Create CodecraftSage
        sage = CodecraftSage(config)

        # Create mock ATR proposal
        from aipha.core.tools.change_proposer import ChangeProposal
        mock_proposal = ChangeProposal(
            id="test-atr-001",
            title="Implementación de Barreras Dinámicas con ATR",
            description="Implement ATR-based dynamic barriers for better risk management",
            justification="ATR provides better volatility-adjusted position sizing",
            component="aipha/trading_flow/labelers/potential_capture_engine.py",
            params={
                "atr_period": 20,
                "tp_multiplier": 5.0,
                "sl_multiplier": 3.0,
                "time_limit": 20
            },
            priority="high",
            estimated_impact="significant"
        )

        # Implement the change
        result = sage.implement_change(mock_proposal)

        print("=== CodecraftSage Implementation Result ===")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Files to modify: {result.files_modified}")
        print(f"Test files to create: {result.test_files_created}")
        print(f"Code length: {len(result.code)} characters")
        print(f"Test code length: {len(result.test_code)} characters")

        if result.success:
            print("\n=== Generated Code Preview ===")
            print(result.code[:500] + "..." if len(result.code) > 500 else result.code)

            print("\n=== Generated Test Preview ===")
            print(result.test_code[:500] + "..." if len(result.test_code) > 500 else result.test_code)

        print("\nCodecraftSage test completed successfully!")

    except Exception as e:
        print(f"CodecraftSage test failed: {e}")
        import traceback
        traceback.print_exc()