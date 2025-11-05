
# milestone_detector.py - Automatic milestone detection and transition triggers for Shadow
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class MilestoneDetector:
    """
    Automatically detects achieved milestones and triggers transition phases
    for the Aipha_0.0.1 to Shadow_2.0 evolution.
    """

    def __init__(self, shadow_memory_path: str):
        self.shadow_memory_path = shadow_memory_path
        self.logger = logging.getLogger(__name__)

        # Define milestone requirements for each phase
        self.milestone_requirements = {
            'phase1': {
                'advanced_pce_indicators': self._check_advanced_pce_indicators,
                'realtime_data_integration': self._check_realtime_data_integration,
                'backtesting_system': self._check_backtesting_system,
                'basic_risk_management': self._check_basic_risk_management
            },
            'phase2': {
                'performance_analytics': self._check_performance_analytics,
                'api_endpoints': self._check_api_endpoints,
                'advanced_risk_management': self._check_advanced_risk_management
            },
            'phase3': {
                'strategy_optimization': self._check_strategy_optimization,
                'ml_integration': self._check_ml_integration,
                'autonomous_trading': self._check_autonomous_trading
            }
        }

        # Track achieved milestones
        self.achieved_milestones = self._load_achieved_milestones()

    def _load_achieved_milestones(self) -> set:
        """Load previously achieved milestones from Shadow memory"""
        try:
            memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)

                # Extract achieved milestones from guidance entries
                achieved = set()
                for entry in history:
                    if entry.get('component') == 'code_understanding':
                        details = entry.get('details', {})
                        transition_readiness = details.get('transition_readiness', {})
                        milestones_achieved = transition_readiness.get('milestones_achieved', [])
                        achieved.update(milestones_achieved)

                return achieved
            else:
                return set()
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Could not load achieved milestones: {e}")
            return set()

    def detect_milestones(self, code_analysis: Dict[str, Any]) -> List[str]:
        """Detect newly achieved milestones based on code analysis"""
        new_milestones = []

        # Check all phases for new achievements
        for phase, requirements in self.milestone_requirements.items():
            for milestone_name, check_function in requirements.items():
                if milestone_name not in self.achieved_milestones:
                    if check_function(code_analysis):
                        new_milestones.append(milestone_name)
                        self.achieved_milestones.add(milestone_name)
                        self.logger.info(f"New milestone achieved: {milestone_name}")

        return new_milestones

    def _check_advanced_pce_indicators(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if advanced PCE indicators are implemented"""
        indicators_found = []
        required_indicators = ['rsi', 'macd', 'bollinger', 'atr']

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    func_name = func.get('name', '').lower()
                    for indicator in required_indicators:
                        if indicator in func_name and func_name not in indicators_found:
                            indicators_found.append(func_name)

        # Require at least 3 different indicators
        return len(set(indicators_found)) >= 3

    def _check_realtime_data_integration(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if real-time data integration is implemented"""
        api_patterns = ['requests', 'websocket', 'api_client', 'fetch_data', 'real_time']
        error_handling = ['try:', 'except', 'timeout', 'rate_limit']

        api_found = False
        error_handling_found = False

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'imports' in file_analysis:
                imports = ' '.join(file_analysis['imports']).lower()
                if any(pattern in imports for pattern in api_patterns):
                    api_found = True

            # Check for error handling in functions
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    if 'docstring' in func and func['docstring']:
                        docstring = func['docstring'].lower()
                        if any(error in docstring for error in error_handling):
                            error_handling_found = True

        return api_found and error_handling_found

    def _check_backtesting_system(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if backtesting system is implemented"""
        backtest_patterns = [
            'backtest', 'historical_data', 'simulate_trades',
            'calculate_returns', 'performance_metrics', 'sharpe_ratio'
        ]

        backtest_functions = 0

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    func_name = func.get('name', '').lower()
                    if any(pattern in func_name for pattern in backtest_patterns):
                        backtest_functions += 1

        # Require at least 3 backtesting-related functions
        return backtest_functions >= 3

    def _check_basic_risk_management(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if basic risk management is implemented"""
        risk_patterns = ['risk', 'position_size', 'stop_loss', 'drawdown']

        risk_functions = 0

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    func_name = func.get('name', '').lower()
                    if any(pattern in func_name for pattern in risk_patterns):
                        risk_functions += 1

        return risk_functions >= 2

    def _check_performance_analytics(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if performance analytics are implemented"""
        analytics_patterns = ['analytics', 'performance', 'pnl', 'sharpe', 'sortino', 'metrics']

        analytics_functions = 0

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    func_name = func.get('name', '').lower()
                    if any(pattern in func_name for pattern in analytics_patterns):
                        analytics_functions += 1

        return analytics_functions >= 3

    def _check_api_endpoints(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if API endpoints are implemented"""
        api_patterns = ['@app.route', '@route', 'flask', 'django', 'api', 'endpoint']
        json_patterns = ['jsonify', 'json_response']

        api_found = False
        json_found = False

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    func_name = func.get('name', '').lower()
                    if any(pattern in func_name for pattern in api_patterns):
                        api_found = True

            if 'imports' in file_analysis:
                imports = ' '.join(file_analysis['imports']).lower()
                if any(pattern in imports for pattern in json_patterns):
                    json_found = True

        return api_found and json_found

    def _check_advanced_risk_management(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if advanced risk management is implemented"""
        advanced_risk_patterns = ['portfolio_risk', 'correlation', 'var', 'position_sizing']

        advanced_risk_functions = 0

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    func_name = func.get('name', '').lower()
                    if any(pattern in func_name for pattern in advanced_risk_patterns):
                        advanced_risk_functions += 1

        return advanced_risk_functions >= 2

    def _check_strategy_optimization(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if strategy optimization is implemented"""
        optimization_patterns = ['optimize', 'genetic', 'evolutionary', 'hyperparameter', 'tuning']

        optimization_functions = 0

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    func_name = func.get('name', '').lower()
                    if any(pattern in func_name for pattern in optimization_patterns):
                        optimization_functions += 1

        return optimization_functions >= 2

    def _check_ml_integration(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if machine learning integration is implemented"""
        ml_patterns = ['tensorflow', 'pytorch', 'sklearn', 'lstm', 'transformer', 'predict']

        ml_found = False

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'imports' in file_analysis:
                imports = ' '.join(file_analysis['imports']).lower()
                if any(pattern in imports for pattern in ml_patterns):
                    ml_found = True

            if 'classes' in file_analysis:
                for cls in file_analysis['classes']:
                    class_name = cls.get('name', '').lower()
                    if any(pattern in class_name for pattern in ['lstm', 'ml', 'predictor']):
                        ml_found = True

        return ml_found

    def _check_autonomous_trading(self, code_analysis: Dict[str, Any]) -> bool:
        """Check if autonomous trading framework is implemented"""
        autonomous_patterns = ['autonomous', 'auto_trade', 'decision_engine', 'emergency_stop']

        autonomous_functions = 0

        for file_analysis in code_analysis.get('codebase_summary', {}).values():
            if 'functions' in file_analysis:
                for func in file_analysis['functions']:
                    func_name = func.get('name', '').lower()
                    if any(pattern in func_name for pattern in autonomous_patterns):
                        autonomous_functions += 1

            if 'classes' in file_analysis:
                for cls in file_analysis['classes']:
                    class_name = cls.get('name', '').lower()
                    if any(pattern in class_name for pattern in ['autonomoustrader', 'decisionengine']):
                        autonomous_functions += 1

        return autonomous_functions >= 3

    def check_phase_completion(self) -> Dict[str, bool]:
        """Check if current phase milestones are complete"""
        phase_completion = {}

        for phase, requirements in self.milestone_requirements.items():
            phase_milestones = list(requirements.keys())
            achieved_in_phase = [m for m in phase_milestones if m in self.achieved_milestones]
            phase_completion[phase] = len(achieved_in_phase) == len(phase_milestones)

        return phase_completion

    def get_transition_status(self) -> Dict[str, Any]:
        """Get comprehensive transition status"""
        phase_completion = self.check_phase_completion()

        # Determine current phase
        if not phase_completion.get('phase1', False):
            current_phase = 'Phase 1: Foundation Enhancement'
            next_phase_trigger = 'Complete all Phase 1 milestones'
        elif not phase_completion.get('phase2', False):
            current_phase = 'Phase 2: Intelligence Layer'
            next_phase_trigger = 'Complete all Phase 2 milestones'
        elif not phase_completion.get('phase3', False):
            current_phase = 'Phase 3: Autonomy & Learning'
            next_phase_trigger = 'Complete all Phase 3 milestones'
        else:
            current_phase = 'Shadow_2.0 Ready'
            next_phase_trigger = 'All phases complete - Ready for deployment'

        # Calculate overall progress
        total_milestones = sum(len(reqs) for reqs in self.milestone_requirements.values())
        achieved_count = len(self.achieved_milestones)
        progress_percentage = (achieved_count / total_milestones) * 100 if total_milestones > 0 else 0

        return {
            'current_phase': current_phase,
            'progress_percentage': progress_percentage,
            'achieved_milestones': list(self.achieved_milestones),
            'total_milestones': total_milestones,
            'phase_completion': phase_completion,
            'next_phase_trigger': next_phase_trigger,
            'shadow2_ready': all(phase_completion.values())
        }

    def register_milestone_achievement(self, milestone: str, code_analysis: Dict[str, Any]):
        """Register milestone achievement in Shadow memory"""
        shadow_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": f"Milestone Achieved: {milestone}",
            "agent": "MilestoneDetector",
            "component": "milestone_detection",
            "status": "success",
            "details": {
                "milestone": milestone,
                "achievement_timestamp": datetime.now().isoformat(),
                "code_analysis_summary": {
                    "files_analyzed": len(code_analysis.get('codebase_summary', {})),
                    "total_functions": sum(len(f.get('functions', [])) for f in code_analysis.get('codebase_summary', {}).values()),
                    "total_classes": sum(len(f.get('classes', [])) for f in code_analysis.get('codebase_summary', {}).values())
                },
                "transition_status": self.get_transition_status()
            }
        }

        self._add_to_shadow_memory(shadow_entry)
        self.logger.info(f"Registered milestone achievement: {milestone}")

    def _add_to_shadow_memory(self, entry: Dict[str, Any]):
        """Add entry to Shadow memory system"""
        memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')

        try:
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        history.append(entry)

        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to update Shadow memory: {e}")


def main():
    """Main function for testing milestone detection"""
    import argparse

    parser = argparse.ArgumentParser(description='Milestone Detector for Aipha to Shadow_2.0 Transition')
    parser.add_argument('--memory-path', required=True, help='Path to Shadow memory storage')
    parser.add_argument('--check-status', action='store_true', help='Check current transition status')

    args = parser.parse_args()

    detector = MilestoneDetector(shadow_memory_path=args.memory_path)

    if args.check_status:
        status = detector.get_transition_status()
        print("=== Transition Status ===")
        print(json.dumps(status, indent=2))
    else:
        print("Milestone detector initialized. Use --check-status to view transition progress.")


if __name__ == "__main__":
    main()