# local_code_monitor.py - Local file system monitoring for Shadow code understanding
import os
import sys
import time
import hashlib
import logging
import ast
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeAnalyzer:
    """Analyzes Python code to understand functionality and changes"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file and extract key information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            analysis = {
                'file_path': file_path,
                'file_type': 'python',
                'functions': [],
                'classes': [],
                'imports': [],
                'complexity': 0,
                'lines_of_code': len(content.split('\n')),
                'summary': ''
            }

            # Parse AST for detailed analysis
            try:
                tree = ast.parse(content)
                analysis.update(self._analyze_ast(tree))
            except SyntaxError as e:
                analysis['syntax_error'] = str(e)

            # Extract imports
            analysis['imports'] = self._extract_imports(content)

            # Generate summary
            analysis['summary'] = self._generate_file_summary(analysis)

            return analysis

        except Exception as e:
            return {
                'file_path': file_path,
                'error': f'Analysis failed: {str(e)}'
            }

    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze AST to extract functions, classes, etc."""
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'line': node.lineno,
                    'docstring': ast.get_docstring(node) or ''
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases],
                    'line': node.lineno,
                    'docstring': ast.get_docstring(node) or ''
                })

        return {
            'functions': functions,
            'classes': classes,
            'complexity': len(functions) + len(classes) * 2  # Simple complexity metric
        }

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from code"""
        imports = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)

        return imports

    def _generate_file_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the file"""
        summary_parts = []

        if analysis.get('classes'):
            class_names = [c['name'] for c in analysis['classes']]
            summary_parts.append(f"Defines classes: {', '.join(class_names)}")

        if analysis.get('functions'):
            func_names = [f['name'] for f in analysis['functions']]
            summary_parts.append(f"Contains functions: {', '.join(func_names)}")

        if analysis.get('imports'):
            key_imports = [imp for imp in analysis['imports'] if not imp.startswith('import os') and not imp.startswith('import sys')]
            if key_imports:
                summary_parts.append(f"Uses libraries: {', '.join(key_imports[:3])}")

        if not summary_parts:
            summary_parts.append("Utility or configuration file")

        return '. '.join(summary_parts)

    def analyze_changes(self, changed_files: List[str], repo_path: str) -> Dict[str, Any]:
        """Analyze all changed files and provide comprehensive understanding"""
        analysis = {
            'changed_files': [],
            'code_understanding': {},
            'impact_assessment': '',
            'functionality_changes': []
        }

        for file_path in changed_files:
            full_path = os.path.join(repo_path, file_path)

            if os.path.exists(full_path) and file_path.endswith('.py'):
                file_analysis = self.analyze_file(full_path)
                analysis['changed_files'].append(file_path)
                analysis['code_understanding'][file_path] = file_analysis

                # Extract functionality changes
                if 'functions' in file_analysis:
                    for func in file_analysis['functions']:
                        analysis['functionality_changes'].append({
                            'file': file_path,
                            'type': 'function',
                            'name': func['name'],
                            'description': func.get('docstring', '').split('.')[0] if func.get('docstring') else 'No description'
                        })

                if 'classes' in file_analysis:
                    for cls in file_analysis['classes']:
                        analysis['functionality_changes'].append({
                            'file': file_path,
                            'type': 'class',
                            'name': cls['name'],
                            'description': cls.get('docstring', '').split('.')[0] if cls.get('docstring') else 'No description'
                        })

        # Generate impact assessment
        analysis['impact_assessment'] = self._assess_impact(analysis)

        return analysis

    def _assess_impact(self, analysis: Dict[str, Any]) -> str:
        """Assess the overall impact of the changes"""
        changes = analysis.get('functionality_changes', [])

        if not changes:
            return "Minor changes or non-functional updates"

        # Count different types of changes
        new_functions = len([c for c in changes if 'new' in c.get('description', '').lower()])
        new_classes = len([c for c in changes if c['type'] == 'class'])
        total_changes = len(changes)

        if new_classes > 0:
            return f"Major architectural changes: {new_classes} new classes, {total_changes} total functionality changes"
        elif new_functions > 2:
            return f"Significant feature additions: {new_functions} new functions, {total_changes} total changes"
        elif total_changes > 5:
            return f"Moderate enhancements: {total_changes} functionality changes"
        else:
            return f"Minor improvements: {total_changes} functionality changes"


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events for code monitoring"""

    def __init__(self, monitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.py', '.json', '.md')):
            self.logger.info(f"File modified: {event.src_path}")
            self.monitor.handle_file_change(event.src_path, 'modified')

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.py', '.json', '.md')):
            self.logger.info(f"File created: {event.src_path}")
            self.monitor.handle_file_change(event.src_path, 'created')

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(('.py', '.json', '.md')):
            self.logger.info(f"File deleted: {event.src_path}")
            self.monitor.handle_file_change(event.src_path, 'deleted')


class LocalCodeMonitor:
    """
    Monitors local file system for code changes and provides Shadow with real-time understanding
    """

    def __init__(self, local_path: str, shadow_memory_path: str):
        self.local_path = local_path
        self.shadow_memory_path = shadow_memory_path
        self.code_analyzer = CodeAnalyzer()
        self.file_hashes = {}
        self.observer = None
        self.event_handler = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize monitoring
        self._initialize_monitoring()

    def _initialize_monitoring(self):
        """Initialize file monitoring system"""
        self.logger.info(f"Initializing local code monitoring for: {self.local_path}")

        # Build initial file database
        self._build_file_hashes()
        self._analyze_initial_codebase()

        # Setup file system monitoring
        self.event_handler = FileChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.local_path, recursive=True)

    def _build_file_hashes(self):
        """Build database of current file hashes"""
        self.logger.info("Building file hash database...")

        for root, dirs, files in os.walk(self.local_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]

            for file in files:
                if file.endswith(('.py', '.json', '.md', '.yaml', '.yml')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.local_path)

                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            file_hash = hashlib.md5(content).hexdigest()
                            self.file_hashes[relative_path] = file_hash
                    except Exception as e:
                        self.logger.warning(f"Could not hash {relative_path}: {e}")

        self.logger.info(f"Built hash database for {len(self.file_hashes)} files")

    def _analyze_initial_codebase(self):
        """Analyze the entire codebase initially for Shadow understanding"""
        self.logger.info("Performing initial codebase analysis...")

        codebase_analysis = {
            'timestamp': datetime.now().isoformat(),
            'repository': 'Aipha_0.0.1_local',
            'analysis_type': 'LOCAL_CODEBASE_ANALYSIS',
            'files_analyzed': 0,
            'codebase_summary': {},
            'architecture_overview': '',
            'transition_readiness': self._assess_transition_readiness()
        }

        for root, dirs, files in os.walk(self.local_path):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.local_path)

                    analysis = self.code_analyzer.analyze_file(file_path)
                    codebase_analysis['codebase_summary'][relative_path] = analysis
                    codebase_analysis['files_analyzed'] += 1

        # Generate architecture overview
        codebase_analysis['architecture_overview'] = self._generate_architecture_overview(codebase_analysis)

        # Register initial understanding in Shadow memory
        self._register_codebase_understanding(codebase_analysis)

    def _assess_transition_readiness(self) -> Dict[str, Any]:
        """Assess readiness for transition to Shadow_2.0"""
        readiness = {
            'current_stage': 'Aipha_0.0.1',
            'milestones_achieved': [],
            'next_milestones': [],
            'transition_triggers': [],
            'shadow_2_0_readiness': 0.0
        }

        # Check for basic system components
        main_exists = os.path.exists(os.path.join(self.local_path, 'main.py'))
        config_exists = os.path.exists(os.path.join(self.local_path, 'config.json'))
        shadow_exists = os.path.exists(os.path.join(self.local_path, 'shadow.py'))
        pce_exists = os.path.exists(os.path.join(self.local_path, 'potential_capture_engine.py'))

        if main_exists:
            readiness['milestones_achieved'].append('Main execution system implemented')
        if config_exists:
            readiness['milestones_achieved'].append('Configuration system centralized')
        if shadow_exists:
            readiness['milestones_achieved'].append('Shadow knowledge layer implemented')
        if pce_exists:
            readiness['milestones_achieved'].append('Potential Capture Engine implemented')

        # Define next milestones for Shadow_2.0 transition
        readiness['next_milestones'] = [
            'Implement advanced PCE with multiple indicators',
            'Add real-time data integration',
            'Create automated backtesting system',
            'Implement risk management layer',
            'Add performance analytics and reporting',
            'Create API endpoints for external access',
            'Implement automated strategy optimization',
            'Add machine learning components'
        ]

        # Define transition triggers
        readiness['transition_triggers'] = [
            'All core PCE indicators implemented (RSI, MACD, Bollinger Bands)',
            'Real-time market data integration completed',
            'Automated backtesting system operational',
            'Risk management with position sizing implemented',
            'Performance analytics dashboard functional',
            'API endpoints for strategy access available',
            'Machine learning optimization pipeline active'
        ]

        # Calculate readiness percentage
        total_milestones = len(readiness['milestones_achieved']) + len(readiness['next_milestones'])
        readiness['shadow_2_0_readiness'] = len(readiness['milestones_achieved']) / total_milestones

        return readiness

    def _generate_architecture_overview(self, analysis: Dict[str, Any]) -> str:
        """Generate high-level architecture overview"""
        summary = analysis.get('codebase_summary', {})

        # Count different types of files and components
        total_files = len(summary)
        classes = sum(len(file_info.get('classes', [])) for file_info in summary.values())
        functions = sum(len(file_info.get('functions', [])) for file_info in summary.values())

        # Identify main components
        main_components = []
        for file_path, file_info in summary.items():
            if file_info.get('classes'):
                for cls in file_info['classes']:
                    main_components.append(f"{cls['name']} ({file_path})")

        overview = f"Local codebase contains {total_files} Python files with {classes} classes and {functions} functions. "

        if main_components:
            overview += f"Main components: {', '.join(main_components[:5])}"
            if len(main_components) > 5:
                overview += f" and {len(main_components) - 5} more"

        return overview

    def handle_file_change(self, file_path: str, change_type: str):
        """Handle file system change events"""
        try:
            relative_path = os.path.relpath(file_path, self.local_path)

            # Skip if not a monitored file type
            if not any(relative_path.endswith(ext) for ext in ['.py', '.json', '.md', '.yaml', '.yml']):
                return

            self.logger.info(f"Processing {change_type} event for: {relative_path}")

            # Analyze the change
            change_analysis = self._analyze_file_change(relative_path, change_type)

            # Register change understanding in Shadow memory
            self._register_change_understanding(change_analysis, change_type)

            # Update file hash
            if change_type in ['modified', 'created']:
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        self.file_hashes[relative_path] = hashlib.md5(content).hexdigest()
                except Exception as e:
                    self.logger.warning(f"Could not update hash for {relative_path}: {e}")
            elif change_type == 'deleted':
                self.file_hashes.pop(relative_path, None)

        except Exception as e:
            self.logger.error(f"Error handling file change: {e}")

    def _analyze_file_change(self, relative_path: str, change_type: str) -> Dict[str, Any]:
        """Analyze a specific file change"""
        analysis = {
            'file_path': relative_path,
            'change_type': change_type,
            'timestamp': datetime.now().isoformat(),
            'code_understanding': {},
            'functionality_changes': [],
            'impact_assessment': ''
        }

        if change_type in ['modified', 'created']:
            full_path = os.path.join(self.local_path, relative_path)
            if os.path.exists(full_path):
                file_analysis = self.code_analyzer.analyze_file(full_path)
                analysis['code_understanding'] = file_analysis

                # Extract functionality changes
                if 'functions' in file_analysis:
                    for func in file_analysis['functions']:
                        analysis['functionality_changes'].append({
                            'type': 'function',
                            'name': func['name'],
                            'description': func.get('docstring', '').split('.')[0] if func.get('docstring') else 'No description'
                        })

                if 'classes' in file_analysis:
                    for cls in file_analysis['classes']:
                        analysis['functionality_changes'].append({
                            'type': 'class',
                            'name': cls['name'],
                            'description': cls.get('docstring', '').split('.')[0] if cls.get('docstring') else 'No description'
                        })

        # Assess impact
        analysis['impact_assessment'] = self._assess_change_impact(analysis)

        return analysis

    def _assess_change_impact(self, analysis: Dict[str, Any]) -> str:
        """Assess the impact of a file change"""
        change_type = analysis.get('change_type', '')
        functionality_changes = analysis.get('functionality_changes', [])

        if change_type == 'deleted':
            return "File deletion - potential functionality loss"
        elif change_type == 'created':
            return f"New file created with {len(functionality_changes)} functionality elements"
        elif change_type == 'modified':
            if not functionality_changes:
                return "Minor modifications or non-functional changes"
            else:
                return f"Modified file with {len(functionality_changes)} functionality elements"

        return "Unknown change impact"

    def _register_codebase_understanding(self, analysis: Dict[str, Any]):
        """Register comprehensive codebase understanding in Shadow memory"""
        shadow_entry = {
            "timestamp": analysis['timestamp'],
            "action": f"Shadow Code Understanding: Local codebase analysis for Aipha_0.0.1",
            "agent": "LocalCodeMonitor",
            "component": "code_understanding",
            "status": "success",
            "details": {
                "repository": analysis['repository'],
                "analysis_type": analysis['analysis_type'],
                "files_analyzed": analysis['files_analyzed'],
                "architecture_overview": analysis['architecture_overview'],
                "codebase_summary": analysis['codebase_summary'],
                "transition_readiness": analysis['transition_readiness'],
                "understanding_status": "LOCAL_CODEBASE_COMPLETE"
            }
        }

        self._add_to_shadow_memory(shadow_entry)
        self.logger.info(f"Registered local codebase understanding: {analysis['files_analyzed']} files analyzed")

    def _register_change_understanding(self, change_analysis: Dict[str, Any], change_type: str):
        """Register file change understanding in Shadow memory"""
        shadow_entry = {
            "timestamp": change_analysis['timestamp'],
            "action": f"Shadow Code Understanding: Local file {change_type} - {change_analysis['file_path']}",
            "agent": "LocalCodeMonitor",
            "component": "change_understanding",
            "status": "success",
            "details": {
                "file_path": change_analysis['file_path'],
                "change_type": change_type,
                "analysis_type": "LOCAL_FILE_CHANGE_ANALYSIS",
                "code_understanding": change_analysis['code_understanding'],
                "functionality_changes": change_analysis['functionality_changes'],
                "impact_assessment": change_analysis['impact_assessment'],
                "understanding_status": "LOCAL_CHANGE_ANALYZED"
            }
        }

        self._add_to_shadow_memory(shadow_entry)
        self.logger.info(f"Registered change understanding: {change_analysis['file_path']} ({change_type})")

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

    def start_monitoring(self):
        """Start the file system monitoring"""
        self.logger.info("Starting local code monitoring...")
        self.observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop the file system monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.logger.info("Local code monitoring stopped")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "monitoring_active": self.observer is not None and self.observer.is_alive(),
            "local_path": self.local_path,
            "shadow_memory_path": self.shadow_memory_path,
            "tracked_files": len(self.file_hashes),
            "last_check": datetime.now().isoformat(),
            "codebase_analyzed": True,
            "shadow_understanding": "ACTIVE"
        }


def main():
    """Main function to run the local code monitor"""
    import argparse

    parser = argparse.ArgumentParser(description='Local Code Monitor for Shadow Real-time Understanding')
    parser.add_argument('--local-path', required=True, help='Local path to monitor')
    parser.add_argument('--memory-path', required=True, help='Path to Shadow memory storage')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')

    args = parser.parse_args()

    # Create monitor
    monitor = LocalCodeMonitor(
        local_path=args.local_path,
        shadow_memory_path=args.memory_path
    )

    if args.continuous:
        # Start continuous monitoring
        monitor.start_monitoring()
    else:
        # Single status check
        status = monitor.get_monitoring_status()
        print("Local Code Monitoring Status:")
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()