"""Analyzer aggiornato con tracking degli smell."""

import os
import subprocess
import tempfile
import datetime
import json
import logging
import shutil
import threading
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import asdict
from git import Repo, Commit, GitCommandError
import concurrent.futures
import multiprocessing
import psutil

import pandas as pd

from ..models import CommitMetrics, RepositoryInfo
from ..utils import CodeAnalysisUtils, GitRepositoryManager, create_single_repo_report
from .smell_tracker import SmellEvolutionTracker, SmellDetection, SmellEventType

logger = logging.getLogger('repo_analyzer.repo_analyzer')

class RepoAnalyzer:
    """Optimized Git repository analyzer with smell tracking"""
    
    def __init__(self, repo_info: RepositoryInfo, resume: bool = False):
        """
        Initialize the repository analyzer.
        
        Args:
            repo_info: Repository information
            resume: Flag to resume analysis from an interrupted commit
        """
        self.repo_info = repo_info
        self.resume = resume
        self.sequential_mode = True
        
        # Set up directories
        self._setup_directories()
        
        # Initialize Git repository
        self._initialize_repository()
        
        # Initialize smell tracker
        self.smell_tracker = SmellEvolutionTracker()
        
        # Shared data structures with proper thread safety
        self.lock = threading.Lock()
        self.author_stats = {}
        self.last_commit_by_author = {}
        self.commit_metrics = []
        self.analyzed_commits = set()
        self.skipped_commits = set()
        self.bugged_files_freq = {}
        self.edited_files_freq = {}
        self.edited_files_complexity = {}

        # Progress display integration
        self._progress = None
        
        # Load previous state if resuming
        if resume:
            self.load_checkpoint()
    
    def _setup_directories(self):
        """Set up the output and temporary directories."""
        # Set output directory
        if not self.repo_info.output_dir:
            self.repo_info.output_dir = self.repo_info.name
        
        os.makedirs(self.repo_info.output_dir, exist_ok=True)
        
        # Set temporary directory
        if not self.repo_info.temp_dir:
            self.repo_info.temp_dir = tempfile.mkdtemp(prefix=f"repo_analyzer_{self.repo_info.name}_")
        else:
            os.makedirs(self.repo_info.temp_dir, exist_ok=True)
        
        # Set local path for the main clone
        if not self.repo_info.local_path:
            self.repo_info.local_path = os.path.join(self.repo_info.temp_dir, "repo")
        
        # Output and checkpoint files
        self.output_file = os.path.join(self.repo_info.output_dir, "commit_metrics.csv")
        self.checkpoint_file = os.path.join(self.repo_info.output_dir, "checkpoint.json")
        self.smell_evolution_file = os.path.join(self.repo_info.output_dir, "smell_evolution.json")
        
        logger.info(f"Analysis of {self.repo_info.url}")
        logger.info(f"Using temporary directory: {self.repo_info.temp_dir}")
        logger.info(f"Results will be saved to: {self.repo_info.output_dir}")
    
    def _initialize_repository(self):
        """Initialize the Git repository and create worker clones if needed."""
        try:
            self.repo = GitRepositoryManager.clone_repository(self.repo_info.url, self.repo_info.local_path)

            # Check if GitPython is working correctly
            self.use_gitpython = self._check_git_readiness()
            if not self.use_gitpython:
                logger.warning("Will use subprocess for git operations instead of GitPython")

            self.worker_dirs = []
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary directories."""
        if os.path.exists(self.repo_info.temp_dir):
            shutil.rmtree(self.repo_info.temp_dir)
            logger.info(f"Temporary directory cleaned: {self.repo_info.temp_dir}")

    def _safe_git_operation(self, operation_name: str, func, *args, **kwargs):
        """
        Safely execute a Git operation with better error handling for GitPython errors.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                if "read of closed file" in str(e):
                    logger.warning(f"GitPython file handle error in {operation_name} (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(1 + attempt)
                    
                    if attempt == max_retries - 1:
                        logger.error(f"All attempts failed for {operation_name} due to file handle issues")
                        return None
                else:
                    raise
            except (GitCommandError, IOError, OSError) as e:
                logger.warning(f"Git error in {operation_name} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1 + attempt)
                
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for {operation_name}: {e}")
                    return None
        
        return None

    def _analyze_commit_common(self, commit: Commit, checkout_path: str, author_name: str, commit_date: datetime.datetime) -> Optional[CommitMetrics]:
        """
        Common commit analysis logic with smell tracking.
        """
        # Verify commit object integrity before proceeding
        try:
            _ = commit.tree
            for parent in commit.parents:
                _ = parent.tree
            _ = commit.stats
        except (ValueError, KeyError, GitCommandError, AttributeError) as e:
            error_msg = str(e)
            if any(err_text in error_msg for err_text in ["Odd-length string", "Non-hexadecimal", "could not be resolved", "does not exist", "not a valid"]):
                logger.error(f"Invalid Git object reference in commit {commit.hexsha}: {error_msg}")
                return None
            else:
                logger.warning(f"Unexpected error accessing commit {commit.hexsha}: {error_msg}")

        try:
            # Calculate LOC and changed files
            loc_added, loc_deleted, files_changed, changed_files = self._calculate_diff_stats(commit, checkout_path)

            if not changed_files and files_changed == 0:
                logger.warning(f"No valid changes detected in commit {commit.hexsha}, skipping")
                return None

            # Update edited files frequency counter
            self._update_edited_files_counter(changed_files)

            # Calculate cyclomatic complexity for Python files
            commit_complexity, project_complexity, files_complexity = self._calculate_complexity(checkout_path, changed_files)

            # Update edited files complexity
            self._update_edited_files_complexity(files_complexity)

            # Calculate code quality metrics with detailed smell information
            total_smells, total_warnings, smells_by_file, smells_by_type = self._calculate_code_quality_with_smells(checkout_path, changed_files)

            # Process smell evolution
            smell_events = self._process_smell_evolution(commit.hexsha, commit_date, changed_files, smells_by_file)

            # Calculate smell density
            total_loc, smell_density = self._calculate_smell_density(checkout_path, total_smells)

            # Author metrics
            author_experience, time_since_last_commit = self._update_author_metrics(author_name, commit_date)

            # Check if commit is from a Pull Request
            is_pr = False
            try:
                is_pr = CodeAnalysisUtils.is_from_pull_request(commit.message)
            except Exception as pr_err:
                logger.warning(f"Error checking if commit is from PR: {pr_err}")

            # Check if commit is a bug fix
            is_bug_fix = False
            try:
                is_bug_fix = CodeAnalysisUtils.is_bug_fix(commit.message)
            except Exception as bf_err:
                logger.warning(f"Error checking if commit is bug fix: {bf_err}")

            # Update bug frequency if this is a bug fix
            if is_bug_fix:
                self._update_bug_frequency(changed_files)

            # Classify commit message
            commit_goals = {}
            try:
                commit_goals = CodeAnalysisUtils._classify_commit_message(commit.message)
            except Exception as classify_err:
                logger.warning(f"Error classifying commit message: {classify_err}")
                commit_goals = {
                    "new_feature": 0,
                    "bug_fixing": 1 if is_bug_fix else 0,
                    "enhancement": 0,
                    "refactoring": 0
                }

            # Extract smell statistics from events
            smells_introduced = sum(1 for event in smell_events if event.get('event_type') == 'introduced')
            smells_removed = sum(1 for event in smell_events if event.get('event_type') == 'removed')
            smells_modified = sum(1 for event in smell_events if event.get('event_type') == 'modified')
            smells_persisted = sum(1 for event in smell_events if event.get('event_type') == 'persisted')

            # Get files with smell information
            files_with_smells = list(smells_by_file.keys())
            files_smell_introduced = [event.get('filename') for event in smell_events 
                                    if event.get('event_type') == 'introduced']
            files_smell_removed = [event.get('filename') for event in smell_events 
                                 if event.get('event_type') == 'removed']

            # Create and return metrics object
            metrics = CommitMetrics(
                commit_hash=commit.hexsha,
                author=author_name,
                date=commit_date,
                message=commit.message.strip() if commit.message else "",
                LOC_added=loc_added,
                LOC_deleted=loc_deleted,
                files_changed=files_changed,
                commit_cyclomatic_complexity=commit_complexity,
                project_cyclomatic_complexity=project_complexity,
                author_experience=author_experience,
                time_since_last_commit=time_since_last_commit,
                total_smells_found=total_smells,
                smell_density=smell_density,
                num_warnings=total_warnings,
                is_pr=is_pr,
                changed_files=changed_files,
                changed_files_complexity=files_complexity,
                is_bug_fix=is_bug_fix, 
                repo_name=self.repo_info.name,
                new_feature=commit_goals.get('new_feature', 0),
                bug_fixing=commit_goals.get('bug_fixing', 0),
                enhancement=commit_goals.get('enhancement', 0),
                refactoring=commit_goals.get('refactoring', 0),
                # Smell tracking fields
                smells_by_file=smells_by_file,
                smells_by_type=smells_by_type,
                smell_events=smell_events,
                smells_introduced=smells_introduced,
                smells_removed=smells_removed,
                smells_modified=smells_modified,
                smells_persisted=smells_persisted,
                files_with_smells=files_with_smells,
                files_smell_introduced=files_smell_introduced,
                files_smell_removed=files_smell_removed
            )

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing commit {commit.hexsha} content: {e}")
            return None
    
    def _calculate_code_quality_with_smells(self, checkout_path: str, changed_files: List[str]) -> Tuple[int, int, Dict, Dict]:
        """Calculate code quality metrics with detailed smell information."""
        total_smells = 0
        total_warnings = 0
        smells_by_file = {}
        smells_by_type = {}
        
        for file_path in changed_files:
            try:
                full_path = os.path.join(checkout_path, file_path)
                if os.path.exists(full_path) and CodeAnalysisUtils.is_code_file(file_path):
                    try:
                        # Get detailed smell information
                        result_cs = CodeAnalysisUtils.detect_code_smells(full_path)
                        if result_cs and isinstance(result_cs, dict):
                            file_smells = result_cs.get('total_smells', 0)
                            total_smells += file_smells
                            
                            if file_smells > 0:
                                smells_by_file[file_path] = result_cs
                                
                                # Count smells by type
                                for smell_type, count in result_cs.get('smells_by_type', {}).items():
                                    smells_by_type[smell_type] = smells_by_type.get(smell_type, 0) + count
                                    
                    except Exception as sm_err:
                        logger.warning(f"Error detecting code smells for {file_path}: {sm_err}")
                    
                    try:
                        total_warnings += CodeAnalysisUtils.detect_warnings(full_path)
                    except Exception as w_err:
                        logger.warning(f"Error detecting warnings for {file_path}: {w_err}")
            except Exception as e:
                logger.warning(f"Error analyzing file {file_path}: {e}")
        
        return total_smells, total_warnings, smells_by_file, smells_by_type
    
    def _process_smell_evolution(self, commit_hash: str, commit_date: datetime.datetime, 
                               changed_files: List[str], smells_by_file: Dict) -> List[Dict]:
        """Process smell evolution and return events."""
        try:
            # Process smells with the tracker
            self.smell_tracker.process_commit_smells(commit_hash, commit_date, changed_files, smells_by_file)
            
            # Extract events for this commit
            events = []
            for filename in changed_files:
                if filename in self.smell_tracker.file_trackers:
                    file_tracker = self.smell_tracker.file_trackers[filename]
                    
                    # Get events for this commit from active smells
                    for smell_history in file_tracker.active_smells.values():
                        for event in smell_history.events:
                            if event.commit_hash == commit_hash:
                                events.append({
                                    'filename': filename,
                                    'smell_id': smell_history.smell_id,
                                    'smell_name': smell_history.smell_name,
                                    'function_name': smell_history.function_name,
                                    'event_type': event.event_type.value,
                                    'line': event.current_line,
                                    'previous_line': event.previous_line,
                                    'notes': event.notes
                                })
                    
                    # Get events from inactive smells too
                    for smell_history in file_tracker.inactive_smells.values():
                        for event in smell_history.events:
                            if event.commit_hash == commit_hash:
                                events.append({
                                    'filename': filename,
                                    'smell_id': smell_history.smell_id,
                                    'smell_name': smell_history.smell_name,
                                    'function_name': smell_history.function_name,
                                    'event_type': event.event_type.value,
                                    'line': event.current_line,
                                    'previous_line': event.previous_line,
                                    'notes': event.notes
                                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error processing smell evolution: {e}")
            return []
    
    def _calculate_diff_stats(self, commit: Commit, checkout_path: str) -> Tuple[int, int, int, List[str]]:
        """Calculate diff statistics for a commit."""
        loc_added = 0
        loc_deleted = 0
        files_changed = 0
        changed_files = []
        
        try:
            # Determine changed files
            if commit.parents:
                # For commits with parents, use standard Git methods
                for parent in commit.parents:
                    try:
                        for diff in parent.diff(commit):
                            # Check if file is code file
                            if hasattr(diff, 'b_path') and diff.b_path and CodeAnalysisUtils.is_code_file(diff.b_path):
                                changed_files.append(diff.b_path)
                                files_changed += 1
                            elif hasattr(diff, 'a_path') and diff.a_path and CodeAnalysisUtils.is_code_file(diff.a_path):
                                changed_files.append(diff.a_path)
                                files_changed += 1
                    except Exception as e:
                        logger.warning(f"Error getting diff for commit {commit.hexsha}: {e}")
            else:
                # For initial commit, find all Python files
                for root, _, files in os.walk(checkout_path):
                    for file in files:
                        if CodeAnalysisUtils.is_code_file(file):
                            rel_path = os.path.relpath(os.path.join(root, file), checkout_path)
                            changed_files.append(rel_path)
                            files_changed += 1
            
            # Try to get line statistics
            if changed_files:
                try:
                    stats = commit.stats.total
                    loc_added = stats.get('insertions', 0)
                    loc_deleted = stats.get('deletions', 0)
                except Exception as stats_err:
                    logger.warning(f"Error getting commit statistics: {stats_err}")
                    # Manual line counting fallback
                    for file_path in changed_files:
                        try:
                            full_path = os.path.join(checkout_path, file_path)
                            if os.path.exists(full_path) and CodeAnalysisUtils.is_code_file(file_path):
                                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    lines = f.readlines()
                                    loc_added += len([l for l in lines if l.strip() and not l.strip().startswith(('#', '//', '/*', '*', '*/')) and '/**/' not in l])
                        except Exception as read_err:
                            logger.warning(f"Error reading file {file_path}: {read_err}")
            else:
                # Fallback to find changed Python files
                try:
                    stats = commit.stats.files
                    for file_path in stats:
                        if CodeAnalysisUtils.is_code_file(file_path):
                            changed_files.append(file_path)
                            files_changed += 1
                            loc_added += stats[file_path].get('insertions', 0)
                            loc_deleted += stats[file_path].get('deletions', 0)
                except Exception as e:
                    logger.warning(f"Fallback for changed files failed: {e}")
        
        except Exception as e:
            logger.error(f"General error in diff analysis: {e}")
        
        return loc_added, loc_deleted, files_changed, changed_files
    
    # Include all other necessary methods from the original analyzer
    def _update_edited_files_complexity(self, files_complexity):
        """Update the complexity of edited files."""
        self.edited_files_complexity.update(files_complexity)
    
    def _update_edited_files_counter(self, changed_files: List[str]):
        """Update the edited files frequency counter."""
        for file_path in changed_files:
            if CodeAnalysisUtils.is_code_file(file_path):
                self.edited_files_freq[file_path] = self.edited_files_freq.get(file_path, 0) + 1
    
    def _update_bug_frequency(self, changed_files: List[str]):
        """Update the bug frequency counter for files in a bug fix."""
        for file_path in changed_files:
            if CodeAnalysisUtils.is_code_file(file_path):
                self.bugged_files_freq[file_path] = self.bugged_files_freq.get(file_path, 0) + 1
    
    def _calculate_complexity(self, checkout_path: str, changed_files: List[str]) -> Tuple[int, int]:
        """Calculate cyclomatic complexity for changed files and the entire project."""
        commit_complexity = 0
        project_complexity = 0
        files_complexity = {}
        
        # Changed files complexity
        for file_path in changed_files:
            try:
                full_path = os.path.join(checkout_path, file_path)
                if os.path.exists(full_path) and CodeAnalysisUtils.is_code_file(file_path):
                    try:
                        file_complexity = CodeAnalysisUtils.calculate_cyclomatic_complexity(full_path)
                        commit_complexity += file_complexity.get('complexity_total', 0.0)
                        files_complexity[file_path] = file_complexity
                    except Exception as cc_err:
                        logger.warning(f"Error calculating complexity for {file_path}: {cc_err}")
            except Exception as e:
                logger.warning(f"Error analyzing file {file_path}: {e}")
        
        # Project-wide complexity
        try:
            for root, _, files in os.walk(checkout_path):
                for file in files:
                    if CodeAnalysisUtils.is_code_file(file):
                        full_path = os.path.join(root, file)
                        try:
                            file_complexity = CodeAnalysisUtils.calculate_cyclomatic_complexity(full_path)
                            if file_complexity:
                                project_complexity += file_complexity.get('complexity_total', 0.0)
                        except Exception as e:
                            logger.warning(f"Error calculating complexity for {full_path}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing project files: {e}")
        
        return commit_complexity, project_complexity, files_complexity
    
    def _calculate_smell_density(self, checkout_path: str, total_smells: int) -> Tuple[int, float]:
        """Calculate smell density (smells per LOC)."""
        total_loc = 0
        try:
            for root, _, files in os.walk(checkout_path):
                for file in files:
                    if CodeAnalysisUtils.is_code_file(file):
                        full_path = os.path.join(root, file)
                        try:
                            total_loc += CodeAnalysisUtils.count_lines_of_code(full_path)
                        except Exception as e:
                            logger.warning(f"Error counting LOC for {full_path}: {e}")
        except Exception as e:
            logger.error(f"Error in total LOC count: {e}")
        
        smell_density = total_smells / total_loc if total_loc > 0 else 0
        return total_loc, smell_density
    
    def _update_author_metrics(self, author_name: str, commit_date: datetime.datetime) -> Tuple[int, float]:
        """Update author metrics and return experience and time since last commit."""
        return self._update_author_metrics_internal(author_name, commit_date)
    
    def _update_author_metrics_internal(self, author_name: str, commit_date: datetime.datetime) -> Tuple[int, float]:
        """Internal function to update author metrics without lock management."""
        # Author experience (number of previous commits)
        if author_name not in self.author_stats:
            self.author_stats[author_name] = 0
        author_experience = self.author_stats[author_name]
        self.author_stats[author_name] += 1
        
        # Time since last commit
        time_since_last_commit = 0.0
        if author_name in self.last_commit_by_author:
            last_commit_date = self.last_commit_by_author[author_name]
            time_delta = commit_date - last_commit_date
            time_since_last_commit = time_delta.total_seconds() / 3600  # In hours
        self.last_commit_by_author[author_name] = commit_date
        
        return author_experience, time_since_last_commit
    
    def save_results(self):
        """Save analysis results to CSV and JSON files, including smell evolution data."""
        metrics_dicts = [asdict(m) for m in self.commit_metrics]
        
        # Convert dates to strings and handle complex data types
        for m in metrics_dicts:
            m['date'] = m['date'].isoformat()
            # Remove changed files list (could be too long)
            m.pop('changed_files', None)
            # Convert complex fields to JSON strings for CSV compatibility
            m['smells_by_file'] = json.dumps(m.get('smells_by_file', {}))
            m['smells_by_type'] = json.dumps(m.get('smells_by_type', {}))
            m['smell_events'] = json.dumps(m.get('smell_events', []))
            m['changed_files_complexity'] = json.dumps(m.get('changed_files_complexity', {}))
            m['files_with_smells'] = json.dumps(m.get('files_with_smells', []))
            m['files_smell_introduced'] = json.dumps(m.get('files_smell_introduced', []))
            m['files_smell_removed'] = json.dumps(m.get('files_smell_removed', []))
        
        df = pd.DataFrame(metrics_dicts)
        
        # Save as CSV
        df.to_csv(self.output_file, index=False)
        logger.info(f"Results saved to {self.output_file}")
        
        # Also save as JSON (with original complex data types)
        json_file = os.path.splitext(self.output_file)[0] + '.json'
        original_metrics = [asdict(m) for m in self.commit_metrics]
        for m in original_metrics:
            m['date'] = m['date'].isoformat()
            m.pop('changed_files', None)  # Remove for size
        
        with open(json_file, 'w') as f:
            json.dump(original_metrics, f, indent=2)
        logger.info(f"Results also saved to {json_file}")
        
        # Save smell evolution data
        smell_evolution_data = self.smell_tracker.get_smell_evolution_report()
        with open(self.smell_evolution_file, 'w') as f:
            json.dump(smell_evolution_data, f, indent=2)
        logger.info(f"Smell evolution data saved to {self.smell_evolution_file}")
        
        # Save bug and edit frequency data
        freq_data = {
            'bugged_files': self.bugged_files_freq,
            'edited_files': self.edited_files_freq,
            'edited_files_complexity': self.edited_files_complexity
        }
        
        freq_file = os.path.join(self.repo_info.output_dir, "file_frequencies.json")
        with open(freq_file, 'w') as f:
            json.dump(freq_data, f, indent=2)
        logger.info(f"File frequencies saved to {freq_file}")
        
        # Create summary report
        create_single_repo_report(self.repo_info.output_dir, df, self.repo_info.name)
    
    # Include essential methods from original class
    def analyze_commit_sequential(self, commit: Commit) -> Optional[CommitMetrics]:
        """Analyze a single commit and extract metrics (sequential version)."""
        # Extract commit info early to avoid GitPython file handle issues
        commit_info = {}
        try:
            commit_info = {
                'hexsha': commit.hexsha,
                'message': commit.message if hasattr(commit, 'message') else '',
                'author_name': commit.author.name if hasattr(commit, 'author') and hasattr(commit.author, 'name') else "Unknown",
                'commit_date': commit.committed_datetime if hasattr(commit, 'committed_datetime') else datetime.datetime.now()
            }
        except ValueError as e:
            if "read of closed file" in str(e):
                logger.error(f"GitPython file handle error extracting commit info: {e}")
                try:
                    self.skipped_commits.add(commit.hexsha if hasattr(commit, 'hexsha') else "unknown")
                except ValueError:
                    self.skipped_commits.add("unknown")
                return None
            else:
                raise
        except Exception as extract_err:
            logger.error(f"Error extracting commit info: {extract_err}")
            try:
                self.skipped_commits.add(commit.hexsha if hasattr(commit, 'hexsha') else "unknown")
            except ValueError:
                self.skipped_commits.add("unknown")
            return None

        try:
            # Validate commit hash format
            if not self._is_valid_commit_hash(commit_info['hexsha']):
                logger.error(f"Invalid commit hash: {commit_info['hexsha']}, skipping")
                self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Check if commit has already been analyzed or skipped
            if commit_info['hexsha'] in self.skipped_commits:
                logger.info(f"Commit {commit_info['hexsha'][:8]} previously skipped, skipping again")
                return None

            logger.info(f"Analyzing commit: {commit_info['hexsha'][:8]} - {commit_info['message'].splitlines()[0] if commit_info['message'] else 'No message'}")

            # Use the main repository directory directly
            checkout_path = os.path.join(self.repo_info.temp_dir, "checkout")

            # Create a clean checkout directory
            if not self._prepare_checkout_directory(checkout_path):
                self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Checkout the commit
            checkout_success = self._safe_checkout_commit(commit_info['hexsha'])
            if not checkout_success:
                logger.error(f"Unable to checkout commit {commit_info['hexsha']}, skipping")
                self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Use secure copy method
            if not self._secure_copy_directory(self.repo_info.local_path, checkout_path):
                logger.error(f"Unable to copy repository to checkout directory, skipping")
                self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Proceed with the actual analysis
            try:
                return self._analyze_commit_common(commit, checkout_path, commit_info['author_name'], commit_info['commit_date'])
            except ValueError as ve:
                if "read of closed file" in str(ve):
                    logger.error(f"GitPython file handle error during analysis: {ve}")
                    self.skipped_commits.add(commit_info['hexsha'])
                    return None
                else:
                    raise
                
        except Exception as e:
            logger.error(f"Error analyzing commit {commit_info.get('hexsha', 'unknown')}: {e}")
            self.skipped_commits.add(commit_info.get('hexsha', 'unknown'))
            return None

    def _is_valid_commit_hash(self, commit_hash: str) -> bool:
        """Check if a given string is a valid commit hash."""
        if not commit_hash or not all(c in '0123456789abcdefABCDEF' for c in commit_hash):
            return False
        if len(commit_hash) != 40:
            return False
        return True
    
    def _prepare_checkout_directory(self, checkout_path: str) -> bool:
        """Prepare a clean checkout directory with proper permissions."""
        if os.path.exists(checkout_path):
            try:
                subprocess.run(['chmod', '-R', '+w', checkout_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                shutil.rmtree(checkout_path)
            except Exception as e:
                logger.error(f"Error cleaning checkout directory: {e}")
                try:
                    subprocess.run(['rm', '-rf', checkout_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                except Exception as rm_err:
                    logger.error(f"Force remove failed: {rm_err}")
                    return False

        try:
            os.makedirs(checkout_path, exist_ok=True)
            os.chmod(checkout_path, 0o755)
            return True
        except Exception as e:
            logger.error(f"Error creating checkout directory: {e}")
            return False
    
    def _safe_checkout_commit(self, commit_hash: str) -> bool:
        """Safely checkout a commit."""
        try:
            process = subprocess.run(
                ['git', 'checkout', commit_hash],
                cwd=self.repo_info.local_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60
            )
            return process.returncode == 0
        except Exception as e:
            logger.warning(f"Checkout failed: {e}")
            return False
    
    def _secure_copy_directory(self, src: str, dst: str) -> bool:
        """A more reliable directory copy method."""
        try:
            # Try rsync if available
            try:
                subprocess.run(['rsync', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(
                    ['rsync', '-a', '--delete', '--exclude=.git/objects/pack/', f"{src}/", f"{dst}/"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300
                )
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            # Manual copy fallback
            for root, dirs, files in os.walk(src):
                rel_path = os.path.relpath(root, src)
                dst_path = os.path.join(dst, rel_path) if rel_path != '.' else dst
                os.makedirs(dst_path, exist_ok=True)
                os.chmod(dst_path, 0o755)

                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dst_path, file)
                    if os.path.exists(dst_file) and not os.access(dst_file, os.W_OK):
                        continue
                    with open(dst_file, 'wb') as f:
                        pass
                    os.chmod(dst_file, 0o644)
                    with open(src_file, 'rb') as src_f, open(dst_file, 'wb') as dst_f:
                        dst_f.write(src_f.read())
            return True
        except Exception as e:
            logger.error(f"Error in secure copy: {e}")
            return False
    
    def _check_git_readiness(self):
        """Verify that GitPython is working correctly."""
        try:
            logger.info("Checking GitPython status...")
            try:
                _ = self.repo.head.commit
            except Exception as e:
                if "read of closed file" in str(e):
                    logger.warning("GitPython file handle issue detected.")
                    return False
            try:
                _ = list(self.repo.iter_commits(max_count=5))
            except Exception as e:
                if "read of closed file" in str(e):
                    logger.warning("GitPython file handle issue detected in iter_commits.")
                    return False
            return True
        except Exception as e:
            logger.warning(f"Git readiness check failed: {e}")
            return False
    
    # Add remaining essential methods for full functionality
    def analyze_repository(self, max_commits: Optional[int] = None):
        """Analyze the repository"""

        if self._progress:
            self._progress.log("Starting sequential analysis...")

        all_commits = self._get_ordered_commits()
        
        if max_commits:
            all_commits = all_commits[:max_commits]

        if self._progress:
            self._progress.set_total_commits(len(all_commits))
            self._progress.log(f"Found {len(all_commits)} commits to analyze")
        
        start_index = self._find_resume_point(all_commits)
        
        if start_index >= len(all_commits):
            logger.info("All commits have already been analyzed.")
            return
        
        total_remaining = len(all_commits) - start_index
        logger.info(f"Sequential analysis of {total_remaining} commits...")
        self._analyze_sequential(all_commits[start_index:], total_remaining)
        self._create_analysis_summary(all_commits)
    
    def _get_ordered_commits(self) -> List[Commit]:
        """Get all commits ordered by date (oldest first)."""
        try:
            try:
                output = subprocess.check_output(
                    ['git', 'log', '--pretty=format:%H', '--reverse'],
                    cwd=self.repo_info.local_path,
                    text=True,
                    timeout=300
                )
                commit_hashes = [h for h in output.strip().split('\n') if h]

                commits = []
                for hash_chunk in [commit_hashes[i:i+100] for i in range(0, len(commit_hashes), 100)]:
                    for commit_hash in hash_chunk:
                        try:
                            commit = self._safe_git_operation(
                                f"get_commit_{commit_hash[:8]}", 
                                self.repo.commit,
                                commit_hash
                            )
                            if commit:
                                commits.append(commit)
                            else:
                                logger.warning(f"Failed to get commit object for {commit_hash[:8]}, skipping")
                        except Exception as commit_err:
                            logger.warning(f"Error getting commit {commit_hash[:8]}: {commit_err}")

                    import gc
                    gc.collect()

                return commits

            except subprocess.SubprocessError as e:
                logger.warning(f"Subprocess error getting commits: {e}, falling back to GitPython")
                all_commits = self._safe_git_operation("iter_commits", lambda: list(self.repo.iter_commits()))
                if all_commits:
                    all_commits.reverse()
                    return all_commits
                raise Exception("Failed to get commits using both methods")

        except Exception as e:
            logger.error(f"Error retrieving commits: {e}")
            raise
    
    def _find_resume_point(self, all_commits: List[Commit]) -> int:
        """Find the index to resume analysis from."""
        start_index = 0
        if self.resume:
            last_commit_hash = self.load_checkpoint()
            if last_commit_hash:
                for i, commit in enumerate(all_commits):
                    if commit.hexsha == last_commit_hash:
                        start_index = i + 1
                        break
                logger.info(f"Resuming analysis from commit {start_index} of {len(all_commits)}")
        return start_index
    
    def _create_analysis_summary(self, all_commits: List[Commit]):
        """Create a summary of the analysis results."""
        analyzed_count = len(self.analyzed_commits)
        skipped_count = len(self.skipped_commits)
        total_count = len(all_commits)
        
        logger.info("=== ANALYSIS COMPLETED ===")
        logger.info(f"Commits analyzed: {analyzed_count}/{total_count} ({analyzed_count/total_count*100:.1f}%)")
        logger.info(f"Commits skipped: {skipped_count}/{total_count} ({skipped_count/total_count*100:.1f}%)")
        
        # Generate reports including smell evolution summary
        error_report_file = os.path.join(self.repo_info.output_dir, "error_report.txt")
        with open(error_report_file, 'w') as f:
            f.write(f"Error report for {self.repo_info.name}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total commits: {total_count}\n")
            f.write(f"Successfully analyzed commits: {analyzed_count}\n")
            f.write(f"Commits skipped due to errors: {skipped_count}\n\n")
            
            if skipped_count > 0:
                f.write("List of skipped commit hashes:\n")
                for commit_hash in sorted(self.skipped_commits):
                    f.write(f"- {commit_hash}\n")
            
            # Add smell evolution summary
            f.write("\n=== SMELL EVOLUTION SUMMARY ===\n")
            smell_summary = self.smell_tracker.get_smell_evolution_report()
            f.write(f"Total smells tracked: {smell_summary['summary']['total_smells_tracked']}\n")
            f.write(f"Active smells: {smell_summary['summary']['active_smells']}\n")
            f.write(f"Removed smells: {smell_summary['summary']['removed_smells']}\n")
            f.write(f"Files with smells: {smell_summary['summary']['files_with_smells']}\n")
            
            f.write("\nSmells by type:\n")
            for smell_type, counts in smell_summary['smells_by_type'].items():
                f.write(f"- {smell_type}: {counts['total']} total ({counts['active']} active, {counts['removed']} removed)\n")
        
        logger.info(f"Error report saved to {error_report_file}")
        
        # Remove checkpoint file if all commits have been processed
        if os.path.exists(self.checkpoint_file) and (analyzed_count + skipped_count >= total_count):
            os.remove(self.checkpoint_file)
            logger.info("Analysis complete, checkpoint removed.")
    
    def _analyze_sequential(self, commits: List[Commit], total_remaining: int):
        """Analyze commits sequentially."""
        try:
            for i, commit in enumerate(commits):
                if commit.hexsha in self.analyzed_commits:
                    logger.info(f"Commit {commit.hexsha[:8]} already analyzed, skipping")
                    continue
                
                metrics = self.analyze_commit_sequential(commit)
                if metrics:
                    self.commit_metrics.append(metrics)
                    self.analyzed_commits.add(metrics.commit_hash)

                    # AGGIORNA PROGRESS
                    if self._progress:
                        self._progress.increment_analyzed(metrics.commit_hash)
                        self._progress.log(f"Analyzed commit {metrics.commit_hash[:8]}")

                    # Save results and checkpoint after EACH commit
                    self.save_results()
                    self.save_checkpoint(metrics.commit_hash)
                else:
                    # AGGIORNA PROGRESS
                    if self._progress:
                        self._progress.increment_skipped()
                        self._progress.log(f"Skipped commit {commit.hexsha[:8]}")

            # Final save
            self.save_results()

        except KeyboardInterrupt:
            if self._progress:
                self._progress.log("Manual interruption detected. State saved.")
    
    def save_checkpoint(self, last_commit_hash: str):
        """Save a checkpoint to resume analysis in case of error."""
        checkpoint_data = {
            'last_commit_hash': last_commit_hash,
            'author_stats': self.author_stats,
            'last_commit_by_author': {author: date.isoformat() for author, date in self.last_commit_by_author.items()},
            'metrics_count': len(self.commit_metrics),
            'skipped_commits': list(self.skipped_commits),
            'bugged_files_freq': self.bugged_files_freq,
            'edited_files_freq': self.edited_files_freq,
            'edited_files_complexity': self.edited_files_complexity,
            'smell_tracker_state': self._serialize_smell_tracker()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved to {self.checkpoint_file}")
    
    def load_checkpoint(self):
        """Load a previously saved checkpoint."""
        if not os.path.exists(self.checkpoint_file):
            logger.info("No checkpoint found, starting analysis from the beginning.")
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore author statistics
            self.author_stats = checkpoint_data.get('author_stats', {})
            
            # Restore last commit dates by author
            last_commit_dates = checkpoint_data.get('last_commit_by_author', {})
            for author, date in last_commit_dates.items():
                self.last_commit_by_author[author] = datetime.datetime.fromisoformat(date)
            
            # Restore skipped commits
            self.skipped_commits = set(checkpoint_data.get('skipped_commits', []))
            
            # Restore frequency counters
            self.bugged_files_freq = checkpoint_data.get('bugged_files_freq', {})
            self.edited_files_freq = checkpoint_data.get('edited_files_freq', {})
            self.edited_files_complexity = checkpoint_data.get('edited_files_complexity', {})
            
            # Restore smell tracker state
            if 'smell_tracker_state' in checkpoint_data:
                self._deserialize_smell_tracker(checkpoint_data['smell_tracker_state'])
            
            # Load data from previously analyzed commits
            if os.path.exists(self.output_file):
                try:
                    df = pd.read_csv(self.output_file)
                    for _, row in df.iterrows():
                        # Parse JSON fields back to objects
                        smells_by_file = json.loads(row.get('smells_by_file', '{}')) if pd.notna(row.get('smells_by_file')) else {}
                        smells_by_type = json.loads(row.get('smells_by_type', '{}')) if pd.notna(row.get('smells_by_type')) else {}
                        smell_events = json.loads(row.get('smell_events', '[]')) if pd.notna(row.get('smell_events')) else []
                        changed_files_complexity = json.loads(row.get('changed_files_complexity', '{}')) if pd.notna(row.get('changed_files_complexity')) else {}
                        files_with_smells = json.loads(row.get('files_with_smells', '[]')) if pd.notna(row.get('files_with_smells')) else []
                        files_smell_introduced = json.loads(row.get('files_smell_introduced', '[]')) if pd.notna(row.get('files_smell_introduced')) else []
                        files_smell_removed = json.loads(row.get('files_smell_removed', '[]')) if pd.notna(row.get('files_smell_removed')) else []
                        
                        metrics = CommitMetrics(
                            commit_hash=row.get('commit_hash', ''),
                            author=row.get('author', ''),
                            date=datetime.datetime.fromisoformat(row.get('date', '')),
                            message=row.get('message', ''),
                            LOC_added=row.get('LOC_added', 0),
                            LOC_deleted=row.get('LOC_deleted', 0),
                            files_changed=row.get('files_changed', 0),
                            commit_cyclomatic_complexity=row.get('commit_cyclomatic_complexity', 0),
                            project_cyclomatic_complexity=row.get('project_cyclomatic_complexity', 0),
                            author_experience=row.get('author_experience', 0),
                            time_since_last_commit=row.get('time_since_last_commit', 0.0),
                            total_smells_found=row.get('total_smells_found', 0),
                            smell_density=row.get('smell_density', 0.0),
                            num_warnings=row.get('num_warnings', 0),
                            is_pr=row.get('is_pr', False),
                            changed_files_complexity=changed_files_complexity,
                            is_bug_fix=row.get('is_bug_fix', False),
                            repo_name=row.get('repo_name', self.repo_info.name),
                            new_feature=row.get('new_feature', 0),
                            bug_fixing=row.get('bug_fixing', 0),
                            enhancement=row.get('enhancement', 0),
                            refactoring=row.get('refactoring', 0),
                            # Smell tracking fields
                            smells_by_file=smells_by_file,
                            smells_by_type=smells_by_type,
                            smell_events=smell_events,
                            smells_introduced=row.get('smells_introduced', 0),
                            smells_removed=row.get('smells_removed', 0),
                            smells_modified=row.get('smells_modified', 0),
                            smells_persisted=row.get('smells_persisted', 0),
                            files_with_smells=files_with_smells,
                            files_smell_introduced=files_smell_introduced,
                            files_smell_removed=files_smell_removed
                        )
                        self.commit_metrics.append(metrics)
                        self.analyzed_commits.add(row.get('commit_hash', ''))
                except Exception as e:
                    logger.error(f"Error loading CSV data: {e}")
            
            logger.info(f"Checkpoint loaded: {len(self.commit_metrics)} commits already analyzed, {len(self.skipped_commits)} commits skipped.")
            return checkpoint_data.get('last_commit_hash')
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def _serialize_smell_tracker(self) -> dict:
        """Serialize smell tracker state for checkpointing."""
        try:
            # Create a simplified representation of the smell tracker state
            serialized_state = {
                'file_trackers': {}
            }
            
            for filename, file_tracker in self.smell_tracker.file_trackers.items():
                serialized_state['file_trackers'][filename] = {
                    'filename': file_tracker.filename,
                    'total_modifications': file_tracker.total_modifications,
                    'commits_with_changes': list(file_tracker.commits_with_changes),
                    'active_smells': {
                        smell_id: {
                            'smell_id': history.smell_id,
                            'filename': history.filename,
                            'function_name': history.function_name,
                            'smell_name': history.smell_name,
                            'description': history.description,
                            'first_introduced_commit': history.first_introduced_commit,
                            'first_introduced_date': history.first_introduced_date.isoformat() if history.first_introduced_date else None,
                            'last_seen_commit': history.last_seen_commit,
                            'last_seen_date': history.last_seen_date.isoformat() if history.last_seen_date else None,
                            'is_active': history.is_active,
                            'current_line': history.current_line,
                            'times_modified': history.times_modified,
                            'times_file_changed': history.times_file_changed,
                            'commits_with_smell': list(history.commits_with_smell)
                        }
                        for smell_id, history in file_tracker.active_smells.items()
                    },
                    'inactive_smells': {
                        smell_id: {
                            'smell_id': history.smell_id,
                            'filename': history.filename,
                            'function_name': history.function_name,
                            'smell_name': history.smell_name,
                            'description': history.description,
                            'first_introduced_commit': history.first_introduced_commit,
                            'first_introduced_date': history.first_introduced_date.isoformat() if history.first_introduced_date else None,
                            'last_seen_commit': history.last_seen_commit,
                            'last_seen_date': history.last_seen_date.isoformat() if history.last_seen_date else None,
                            'is_active': history.is_active,
                            'current_line': history.current_line,
                            'times_modified': history.times_modified,
                            'times_file_changed': history.times_file_changed,
                            'commits_with_smell': list(history.commits_with_smell)
                        }
                        for smell_id, history in file_tracker.inactive_smells.items()
                    }
                }
            
            return serialized_state
        except Exception as e:
            logger.error(f"Error serializing smell tracker: {e}")
            return {}
    
    def _deserialize_smell_tracker(self, serialized_state: dict):
        """Deserialize smell tracker state from checkpoint."""
        try:
            from .smell_tracker import FileSmellTracker, SmellHistory
            
            # Restore file trackers
            for filename, file_data in serialized_state.get('file_trackers', {}).items():
                file_tracker = FileSmellTracker(filename)
                file_tracker.total_modifications = file_data.get('total_modifications', 0)
                file_tracker.commits_with_changes = set(file_data.get('commits_with_changes', []))
                
                # Restore active smells
                for smell_id, smell_data in file_data.get('active_smells', {}).items():
                    history = SmellHistory(
                        smell_id=smell_data['smell_id'],
                        filename=smell_data['filename'],
                        function_name=smell_data['function_name'],
                        smell_name=smell_data['smell_name'],
                        description=smell_data['description']
                    )
                    history.first_introduced_commit = smell_data['first_introduced_commit']
                    history.first_introduced_date = datetime.datetime.fromisoformat(smell_data['first_introduced_date']) if smell_data['first_introduced_date'] else None
                    history.last_seen_commit = smell_data['last_seen_commit']
                    history.last_seen_date = datetime.datetime.fromisoformat(smell_data['last_seen_date']) if smell_data['last_seen_date'] else None
                    history.is_active = smell_data['is_active']
                    history.current_line = smell_data['current_line']
                    history.times_modified = smell_data['times_modified']
                    history.times_file_changed = smell_data['times_file_changed']
                    history.commits_with_smell = set(smell_data['commits_with_smell'])
                    file_tracker.active_smells[smell_id] = history
                
                # Restore inactive smells
                for smell_id, smell_data in file_data.get('inactive_smells', {}).items():
                    history = SmellHistory(
                        smell_id=smell_data['smell_id'],
                        filename=smell_data['filename'],
                        function_name=smell_data['function_name'],
                        smell_name=smell_data['smell_name'],
                        description=smell_data['description']
                    )
                    history.first_introduced_commit = smell_data['first_introduced_commit']
                    history.first_introduced_date = datetime.datetime.fromisoformat(smell_data['first_introduced_date']) if smell_data['first_introduced_date'] else None
                    history.last_seen_commit = smell_data['last_seen_commit']
                    history.last_seen_date = datetime.datetime.fromisoformat(smell_data['last_seen_date']) if smell_data['last_seen_date'] else None
                    history.is_active = smell_data['is_active']
                    history.current_line = smell_data['current_line']
                    history.times_modified = smell_data['times_modified']
                    history.times_file_changed = smell_data['times_file_changed']
                    history.commits_with_smell = set(smell_data['commits_with_smell'])
                    file_tracker.inactive_smells[smell_id] = history
                
                self.smell_tracker.file_trackers[filename] = file_tracker
                
        except Exception as e:
            logger.error(f"Error deserializing smell tracker: {e}")
            # Reset smell tracker if deserialization fails
            self.smell_tracker = SmellEvolutionTracker()