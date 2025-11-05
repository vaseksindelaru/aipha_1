# enhanced_github_monitor.py - Enhanced GitHub monitor with code understanding for Shadow
import os
import sys
import subprocess
import json
import hashlib
import logging
import ast
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

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


class EnhancedGitHubRepositoryMonitor:
    """
    Enhanced GitHub monitor with code understanding capabilities for Shadow
    """

    def __init__(self, repo_url: str, local_path: str, shadow_memory_path: str):
        self.repo_url = repo_url
        self.local_path = local_path
        self.shadow_memory_path = shadow_memory_path
        self.repo_name = self._extract_repo_name(repo_url)
        self.last_commit_hash = None
        self.file_hashes = {}
        self.code_analyzer = CodeAnalyzer()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize tracking
        self._initialize_tracking()

    def _extract_repo_name(self, repo_url: str) -> str:
        return repo_url.split('/')[-1].replace('.git', '')

    def _initialize_tracking(self):
        """Initialize tracking by cloning or updating repository"""
        try:
            if not os.path.exists(self.local_path):
                self.logger.info(f"Cloning repository: {self.repo_url}")
                subprocess.run([
                    'git', 'clone', self.repo_url, self.local_path
                ], check=True)
            else:
                self.logger.info(f"Updating repository: {self.local_path}")
                subprocess.run([
                    'git', '-C', self.local_path, 'pull'
                ], check=True)

            # Get latest commit hash
            result = subprocess.run([
                'git', '-C', self.local_path, 'rev-parse', 'HEAD'
            ], capture_output=True, text=True, check=True)

            self.last_commit_hash = result.stdout.strip()
            self.logger.info(f"Latest commit: {self.last_commit_hash}")

            # Build file hash database and analyze codebase
            self._build_file_hashes()
            self._analyze_codebase()

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git operation failed: {e}")
            raise

    def _build_file_hashes(self):
        """Build database of current file hashes"""
        for root, dirs, files in os.walk(self.local_path):
            dirs[:] = [d for d in dirs if d != '.git']

            for file in files:
                if file.endswith(('.py', '.md', '.json', '.yaml', '.yml')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.local_path)

                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            file_hash = hashlib.md5(content).hexdigest()
                            self.file_hashes[relative_path] = file_hash
                    except Exception as e:
                        self.logger.warning(f"Could not hash {relative_path}: {e}")

    def _analyze_codebase(self):
        """Analyze the entire codebase for Shadow understanding"""
        self.logger.info("Analyzing codebase for Shadow understanding...")

        codebase_analysis = {
            'timestamp': datetime.now().isoformat(),
            'repository': self.repo_name,
            'files_analyzed': 0,
            'codebase_summary': {},
            'architecture_overview': '',
            'all_files': []
        }

        for root, dirs, files in os.walk(self.local_path):
            dirs[:] = [d for d in dirs if d != '.git' and d != '__pycache__']

            for file in files:
                # Incluir todos los archivos relevantes del proyecto
                if file.endswith(('.py', '.json', '.md', '.txt', '.yaml', '.yml', '.txt', '.cfg')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.local_path)
                    
                    # Añadir a lista de todos los archivos
                    codebase_analysis['all_files'].append(relative_path)

                    # Solo analizar archivos Python con AST
                    if file.endswith('.py'):
                        analysis = self.code_analyzer.analyze_file(file_path)
                        codebase_analysis['codebase_summary'][relative_path] = analysis
                        codebase_analysis['files_analyzed'] += 1

        # Generate architecture overview
        codebase_analysis['architecture_overview'] = self._generate_architecture_overview(codebase_analysis)

        # Register codebase understanding in Shadow memory
        self._register_codebase_understanding(codebase_analysis)

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

        overview = f"Codebase contains {total_files} Python files with {classes} classes and {functions} functions. "

        if main_components:
            overview += f"Main components: {', '.join(main_components[:5])}"
            if len(main_components) > 5:
                overview += f" and {len(main_components) - 5} more"

        return overview

    def check_for_changes(self) -> List[Dict[str, Any]]:
        """Check for repository changes and analyze them"""
        try:
            # Update repository
            subprocess.run([
                'git', '-C', self.local_path, 'fetch'
            ], check=True)

            # Get current commit
            result = subprocess.run([
                'git', '-C', self.local_path, 'rev-parse', 'HEAD'
            ], capture_output=True, text=True, check=True)

            current_commit_hash = result.stdout.strip()

            if current_commit_hash == self.last_commit_hash:
                self.logger.info("No new commits found")
                return []

            # Get changed files
            result = subprocess.run([
                'git', '-C', self.local_path, 'diff', '--name-only',
                self.last_commit_hash, current_commit_hash
            ], capture_output=True, text=True, check=True)

            changed_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            changed_files = [f for f in changed_files if f]

            # Get commit details
            result = subprocess.run([
                'git', '-C', self.local_path, 'log', '--oneline', '-1', current_commit_hash
            ], capture_output=True, text=True, check=True)

            commit_message = result.stdout.strip()

            # Analyze changes with code understanding
            changes_analysis = self.code_analyzer.analyze_changes(changed_files, self.local_path)

            # Register comprehensive understanding in Shadow memory
            self._register_change_understanding(changes_analysis, commit_message, current_commit_hash)

            # Update tracking
            self.last_commit_hash = current_commit_hash
            self._build_file_hashes()

            return [changes_analysis]

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git operation failed: {e}")
            return []

    def _register_codebase_understanding(self, analysis: Dict[str, Any]):
        """Register comprehensive codebase understanding in Shadow memory"""
        shadow_entry = {
            "timestamp": analysis['timestamp'],
            "action": f"Shadow Code Understanding: Complete codebase analysis for {self.repo_name}",
            "agent": "EnhancedGitHubMonitor",
            "component": "code_understanding",
            "status": "success",
            "details": {
                "repository": self.repo_name,
                "analysis_type": "FULL_CODEBASE_ANALYSIS",
                "files_analyzed": analysis['files_analyzed'],
                "architecture_overview": analysis['architecture_overview'],
                "codebase_summary": analysis['codebase_summary'],
                "all_files": analysis.get('all_files', []),
                "understanding_status": "COMPLETE"
            }
        }

        self._add_to_shadow_memory(shadow_entry)
        self.logger.info(f"Registered complete codebase understanding: {analysis['files_analyzed']} files analyzed")

    def _register_change_understanding(self, changes_analysis: Dict[str, Any], commit_message: str, commit_hash: str):
        """Register detailed change understanding in Shadow memory"""
        shadow_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": f"Shadow Code Understanding: Change analysis - {commit_message[:50]}...",
            "agent": "EnhancedGitHubMonitor",
            "component": "change_understanding",
            "status": "success",
            "details": {
                "repository": self.repo_name,
                "commit_hash": commit_hash,
                "commit_message": commit_message,
                "analysis_type": "CHANGE_ANALYSIS",
                "changed_files": changes_analysis['changed_files'],
                "code_understanding": changes_analysis['code_understanding'],
                "functionality_changes": changes_analysis['functionality_changes'],
                "impact_assessment": changes_analysis['impact_assessment'],
                "understanding_status": "DETAILED_ANALYSIS_COMPLETE"
            }
        }

        self._add_to_shadow_memory(shadow_entry)
        self.logger.info(f"Registered change understanding: {len(changes_analysis['changed_files'])} files, {len(changes_analysis['functionality_changes'])} functionality changes")

    def _add_to_shadow_memory(self, entry: Dict[str, Any]):
        """Add entry to Shadow memory system"""
        memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')

        try:
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []

            history.append(entry)

            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Failed to update Shadow memory: {e}")

    def get_repository_status(self) -> Dict[str, Any]:
        """Get current repository status with understanding metrics"""
        try:
            result = subprocess.run([
                'git', '-C', self.local_path, 'log', '--oneline', '-5'
            ], capture_output=True, text=True, check=True)

            recent_commits = result.stdout.strip().split('\n')

            return {
                "repository_name": self.repo_name,
                "repository_url": self.repo_url,
                "local_path": self.local_path,
                "last_commit_hash": self.last_commit_hash,
                "recent_commits": recent_commits,
                "tracked_files": len(self.file_hashes),
                "codebase_analyzed": True,
                "last_check": datetime.now().isoformat(),
                "shadow_understanding": "ACTIVE"
            }

        except Exception as e:
            return {
                "error": f"Failed to get repository status: {e}",
                "repository_name": self.repo_name,
                "last_check": datetime.now().isoformat()
            }


def main():
    """Main function to run the enhanced GitHub monitor"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced GitHub Repository Monitor with Code Understanding for Shadow')
    parser.add_argument('--repo-url', required=True, help='GitHub repository URL')
    parser.add_argument('--local-path', required=True, help='Local path for repository')
    parser.add_argument('--memory-path', required=True, help='Path to Shadow memory storage')
    parser.add_argument('--check-interval', type=int, default=300, help='Check interval in seconds')

    args = parser.parse_args()

    # Create enhanced monitor
    monitor = EnhancedGitHubRepositoryMonitor(
        repo_url=args.repo_url,
        local_path=args.local_path,
        shadow_memory_path=args.memory_path
    )

    # Check for changes
    changes = monitor.check_for_changes()

    if changes:
        print(f"Analyzed {len(changes)} change sets:")
        for change_set in changes:
            print(f"  - Impact: {change_set.get('impact_assessment', 'Unknown')}")
            print(f"  - Files changed: {len(change_set.get('changed_files', []))}")
            print(f"  - Functionality changes: {len(change_set.get('functionality_changes', []))}")
    else:
        print("No changes detected")

    # Print status
    status = monitor.get_repository_status()
    print("\nEnhanced Repository Status:")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()