# github_monitor.py - Monitor for Shadow system to track GitHub repository changes
import os
import sys
import subprocess
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class GitHubRepositoryMonitor:
    """
    Monitors GitHub repository for changes and automatically registers them in Shadow memory
    """
    
    def __init__(self, repo_url: str, local_path: str, shadow_memory_path: str):
        self.repo_url = repo_url
        self.local_path = local_path
        self.shadow_memory_path = shadow_memory_path
        self.repo_name = self._extract_repo_name(repo_url)
        self.last_commit_hash = None
        self.file_hashes = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking
        self._initialize_tracking()
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL"""
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
            
            # Build file hash database
            self._build_file_hashes()
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git operation failed: {e}")
            raise
    
    def _build_file_hashes(self):
        """Build database of current file hashes"""
        for root, dirs, files in os.walk(self.local_path):
            # Skip .git directory
            dirs[:] = [d for d in dirs if d != '.git']
            
            for file in files:
                if file.endswith('.py') or file.endswith('.md') or file.endswith('.json') or file.endswith('.yaml'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.local_path)
                    
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            file_hash = hashlib.md5(content).hexdigest()
                            self.file_hashes[relative_path] = file_hash
                    except Exception as e:
                        self.logger.warning(f"Could not hash {relative_path}: {e}")
    
    def check_for_changes(self) -> List[Dict[str, Any]]:
        """Check for repository changes since last check"""
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
            changed_files = [f for f in changed_files if f]  # Remove empty strings
            
            # Get commit details
            result = subprocess.run([
                'git', '-C', self.local_path, 'log', '--oneline', '-1', current_commit_hash
            ], capture_output=True, text=True, check=True)
            
            commit_message = result.stdout.strip()
            
            # Analyze changes
            changes = self._analyze_changes(changed_files, commit_message, current_commit_hash)
            
            # Update tracking
            self.last_commit_hash = current_commit_hash
            self._build_file_hashes()
            
            return changes
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git operation failed: {e}")
            return []
    
    def _analyze_changes(self, changed_files: List[str], commit_message: str, commit_hash: str) -> List[Dict[str, Any]]:
        """Analyze changed files and categorize changes"""
        changes = []
        
        # Analyze each changed file
        for file_path in changed_files:
            if not file_path:
                continue
                
            change_info = {
                'file_path': file_path,
                'change_type': self._determine_change_type(file_path),
                'commit_hash': commit_hash,
                'commit_message': commit_message,
                'timestamp': datetime.now().isoformat()
            }
            
            changes.append(change_info)
        
        # Register changes in Shadow memory
        self._register_shadow_changes(changes, commit_message, commit_hash)
        
        return changes
    
    def _determine_change_type(self, file_path: str) -> str:
        """Determine the type of change based on file path"""
        if file_path.endswith('.py'):
            return 'CODE_CHANGE'
        elif file_path.endswith(('.md', '.txt')):
            return 'DOCUMENTATION_CHANGE'
        elif file_path.endswith(('.yaml', '.yml', '.json')):
            return 'CONFIG_CHANGE'
        elif file_path.endswith('.sh'):
            return 'SCRIPT_CHANGE'
        else:
            return 'OTHER_CHANGE'
    
    def _register_shadow_changes(self, changes: List[Dict[str, Any]], commit_message: str, commit_hash: str):
        """Register changes in Shadow memory system"""
        try:
            # Prepare change entry
            shadow_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": f"GitHub Repository: Automatic change detection from {self.repo_name}",
                "agent": "GitHubMonitor",
                "component": "github_monitor",
                "status": "success",
                "details": {
                    "source_component": "GitHub Repository",
                    "entry_category": "AUTO_DETECTED_CHANGE",
                    "commit_message": commit_message,
                    "commit_hash": commit_hash,
                    "repository": self.repo_url,
                    "changes_detected": len(changes),
                    "changed_files": [change['file_path'] for change in changes],
                    "change_details": changes
                }
            }
            
            # Add to Shadow memory
            self._add_to_shadow_memory(shadow_entry)
            
            self.logger.info(f"Registered {len(changes)} changes in Shadow memory")
            
        except Exception as e:
            self.logger.error(f"Failed to register changes in Shadow memory: {e}")
    
    def _add_to_shadow_memory(self, entry: Dict[str, Any]):
        """Add entry to Shadow memory system"""
        memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')
        
        try:
            # Read current history
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add new entry
            history.append(entry)
            
            # Write back
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to update Shadow memory: {e}")
    
    def get_repository_status(self) -> Dict[str, Any]:
        """Get current repository status"""
        try:
            # Get latest commit info
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
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get repository status: {e}",
                "repository_name": self.repo_name,
                "last_check": datetime.now().isoformat()
            }


def main():
    """Main function to run the GitHub monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GitHub Repository Monitor for Shadow')
    parser.add_argument('--repo-url', required=True, help='GitHub repository URL')
    parser.add_argument('--local-path', required=True, help='Local path for repository')
    parser.add_argument('--memory-path', required=True, help='Path to Shadow memory storage')
    parser.add_argument('--check-interval', type=int, default=300, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = GitHubRepositoryMonitor(
        repo_url=args.repo_url,
        local_path=args.local_path,
        shadow_memory_path=args.memory_path
    )
    
    # Check for changes
    changes = monitor.check_for_changes()
    
    if changes:
        print(f"Detected {len(changes)} changes:")
        for change in changes:
            print(f"  - {change['file_path']} ({change['change_type']})")
    else:
        print("No changes detected")
    
    # Print status
    status = monitor.get_repository_status()
    print("\nRepository Status:")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()