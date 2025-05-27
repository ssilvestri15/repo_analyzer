"""Improved and optimized analyzer for Git repositories supporting both parallel and sequential analysis."""

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

logger = logging.getLogger('repo_analyzer.repo_analyzer')

class RepoAnalyzer:
    """Optimized Git repository analyzer with support for both parallel and sequential analysis."""
    
    def __init__(self, repo_info: RepositoryInfo, resume: bool = False, max_workers: Optional[int] = None):
        """
        Initialize the repository analyzer.
        
        Args:
            repo_info: Repository information
            resume: Flag to resume analysis from an interrupted commit
            max_workers: Maximum number of parallel workers (default: auto-calculated)
                         If max_workers=1, sequential mode is used
        """
        self.repo_info = repo_info
        self.resume = resume
        
        # Determine optimal number of workers based on system specs
        self.max_workers = self._calculate_optimal_workers(max_workers)
        # IMPORTANT: For GitPython, reduce max workers to avoid 'read of closed file' errors
        # These errors occur when too many Git processes are spawned simultaneously
        if self.max_workers > 8:
            logger.warning(f"Reducing worker count from {self.max_workers} to 8 to avoid GitPython file handle errors")
            self.max_workers = 8
        self.sequential_mode = (self.max_workers == 1)
        
        # Set up directories
        self._setup_directories()
        
        # Initialize Git repository
        self._initialize_repository()
        
        # Shared data structures with proper thread safety
        self.lock = threading.Lock()
        self.author_stats = {}
        self.last_commit_by_author = {}
        self.commit_metrics = []
        self.analyzed_commits = set()
        self.skipped_commits = set()
        self.bugged_files_freq = {}
        self.edited_files_freq = {}
        
        # Load previous state if resuming
        if resume:
            self.load_checkpoint()
    
    def _calculate_optimal_workers(self, max_workers: Optional[int]) -> int:
        """Calculate the optimal number of workers based on system resources."""
        if max_workers is not None:
            return max_workers
            
        # Use 75% of available cores, at least 1
        cpu_based_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        
        # Also consider available memory to avoid swapping
        mem = psutil.virtual_memory()
        mem_per_worker = 500 * 1024 * 1024  # Estimate 500MB per worker
        mem_based_workers = max(1, int(mem.available * 0.7 / mem_per_worker))
        
        # Use the smaller of the two limits
        optimal_workers = min(cpu_based_workers, mem_based_workers)
        
        logger.info(f"Using {'sequential' if optimal_workers == 1 else 'parallel'} mode with {optimal_workers} worker(s)")
        return optimal_workers
    
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

            # For parallel mode, create separate work directories for each worker
            self.worker_dirs = []
            if not self.sequential_mode:
                self._create_worker_repositories()
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise
        
    
    def _create_worker_repositories(self):
        """Create worker repositories for parallel processing."""
        for i in range(self.max_workers):
            worker_dir = os.path.join(self.repo_info.temp_dir, f"worker_{i}")
            os.makedirs(worker_dir, exist_ok=True)
            self.worker_dirs.append(worker_dir)
            
            # Clone the repository into each worker directory
            worker_repo_path = os.path.join(worker_dir, "repo")
            self._clone_to_worker(worker_repo_path, i)
    
    def _clone_to_worker(self, worker_repo_path: str, worker_id: int, max_retries: int = 2):
        """Clone the repository to a worker directory with retry logic and added delay."""
        # Add delay between worker creations to avoid git lock conflicts
        time.sleep(worker_id * 0.5)  # Stagger worker initialization

        for attempt in range(max_retries):
            try:
                if os.path.exists(worker_repo_path):
                    # Try to remove lock files first if they exist
                    index_lock = os.path.join(worker_repo_path, '.git', 'index.lock')
                    if os.path.exists(index_lock):
                        try:
                            os.remove(index_lock)
                            logger.info(f"Removed stale lock file at {index_lock}")
                        except Exception as e:
                            logger.warning(f"Failed to remove lock file {index_lock}: {e}")

                    # Now remove the directory
                    shutil.rmtree(worker_repo_path)

                # Clone with a timeout to prevent hanging
                subprocess.run(
                    ['git', 'clone', self.repo_info.local_path, worker_repo_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=120
                )
                return  # Success
            except subprocess.CalledProcessError as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for worker {worker_id} repository cloning: {e}")
                # Wait longer between retries
                time.sleep(2 + attempt * 2)
                if attempt == max_retries - 1:
                    logger.error(f"Failed to clone repository for worker {worker_id} after {max_retries} attempts")
                    raise
            except Exception as e:
                logger.warning(f"Unexpected error in worker {worker_id} cloning: {e}")
                time.sleep(2 + attempt * 2)
                if attempt == max_retries - 1:
                    logger.error(f"Failed to clone repository for worker {worker_id} after {max_retries} attempts")
                    raise
    
    def cleanup(self):
        """Clean up temporary directories."""
        if os.path.exists(self.repo_info.temp_dir):
            shutil.rmtree(self.repo_info.temp_dir)
            logger.info(f"Temporary directory cleaned: {self.repo_info.temp_dir}")

    def _safe_git_operation(self, operation_name: str, func, *args, **kwargs):
        """
        Safely execute a Git operation with better error handling for GitPython errors.
        
        Args:
            operation_name: Name of the operation for logging
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Result of the function or None if failed
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                if "read of closed file" in str(e):
                    logger.warning(f"GitPython file handle error in {operation_name} (attempt {attempt+1}/{max_retries}): {e}")
                    # When this error occurs, we need to wait for file handles to be properly closed
                    time.sleep(1 + attempt)
                    
                    # On the last attempt, give up but don't crash
                    if attempt == max_retries - 1:
                        logger.error(f"All attempts failed for {operation_name} due to file handle issues")
                        return None
                else:
                    # For other value errors, just raise
                    raise
            except (GitCommandError, IOError, OSError) as e:
                logger.warning(f"Git error in {operation_name} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1 + attempt)
                
                # On the last attempt, give up but don't crash
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for {operation_name}: {e}")
                    return None
        
        return None  # Should never reach here
    
    def safe_checkout(self, commit_hash: str, repo_path: str, max_retries: int = 3) -> bool:
        """
        Safely checkout a commit with multiple attempts and enhanced error handling.

        Args:
            commit_hash: Commit hash to checkout
            repo_path: Path to the repository
            max_retries: Maximum number of retry attempts

        Returns:
            True if checkout succeeds, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # First, check for and remove any lock files
                index_lock = os.path.join(repo_path, '.git', 'index.lock')
                if os.path.exists(index_lock):
                    try:
                        os.remove(index_lock)
                        logger.info(f"Removed stale lock file before checkout at {index_lock}")
                    except Exception as e:
                        logger.warning(f"Failed to remove lock file {index_lock}: {e}")
                        # If we can't remove the lock file, wait longer to let other processes finish
                        time.sleep(3 + attempt * 2)

                # Before checkout, let's validate the commit hash format to avoid non-hex errors
                if not all(c in '0123456789abcdefABCDEF' for c in commit_hash):
                    logger.error(f"Invalid commit hash format: {commit_hash}")
                    return False

                # Try checkout with better error detection
                process = subprocess.run(
                    ['git', 'checkout', commit_hash],
                    cwd=repo_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=60  # 60-second timeout
                )

                if process.returncode == 0:
                    return True  # Success

                # Process specific errors
                error_msg = process.stderr.decode('utf-8', errors='ignore')

                # Handle lock file conflicts
                if "index.lock" in error_msg:
                    logger.warning(f"Lock file conflict during checkout, retrying after delay")
                    time.sleep(2 + attempt * 2)
                    continue

                if "tree" in error_msg or "parent" in error_msg:
                    logger.warning(f"Git object reference error: {error_msg}")

                    # Try to recover the object by fetching and unpacking
                    try:
                        # Try to get the remote name
                        remote_process = subprocess.run(
                            ['git', 'remote'],
                            cwd=repo_path,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True
                        )

                        if remote_process.returncode == 0 and remote_process.stdout.strip():
                            remote = remote_process.stdout.strip().split('\n')[0]

                            # Fetch objects from remote
                            subprocess.run(
                                ['git', 'fetch', remote, '--tags', '--force', '--prune'],
                                cwd=repo_path,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=300
                            )

                            # Force unpack objects
                            subprocess.run(
                                ['git', 'fsck', '--full'],
                                cwd=repo_path,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=300
                            )

                            # Try checkout again after fetch
                            retry_process = subprocess.run(
                                ['git', 'checkout', commit_hash],
                                cwd=repo_path,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )

                            if retry_process.returncode == 0:
                                return True
                    except Exception as fetch_err:
                        logger.warning(f"Fetch recovery attempt failed: {fetch_err}")

                    # If we couldn't fix by fetching, try cloning at the specific commit
                    try:
                        # Create a temporary directory for the single commit clone
                        temp_clone_dir = os.path.join(os.path.dirname(repo_path), f"single_commit_{commit_hash[:8]}")
                        if os.path.exists(temp_clone_dir):
                            shutil.rmtree(temp_clone_dir)

                        os.makedirs(temp_clone_dir, exist_ok=True)

                        # Get the repository URL
                        url_process = subprocess.run(
                            ['git', 'config', '--get', 'remote.origin.url'],
                            cwd=repo_path,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True
                        )

                        if url_process.returncode == 0 and url_process.stdout.strip():
                            repo_url = url_process.stdout.strip()

                            # Clone just this specific commit
                            clone_process = subprocess.run(
                                ['git', 'clone', '--depth', '1', '--branch', commit_hash, repo_url, temp_clone_dir],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=300
                            )

                            # If specific commit clone worked, copy the objects to our repo
                            if clone_process.returncode == 0 and os.path.exists(os.path.join(temp_clone_dir, '.git')):
                                subprocess.run(
                                    ['cp', '-r', f"{temp_clone_dir}/.git/objects/*", f"{repo_path}/.git/objects/"],
                                    shell=True,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )

                                # Try checkout one more time
                                final_checkout = subprocess.run(
                                    ['git', 'checkout', commit_hash],
                                    cwd=repo_path,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )

                                if final_checkout.returncode == 0:
                                    return True

                            # Clean up
                            if os.path.exists(temp_clone_dir):
                                shutil.rmtree(temp_clone_dir)
                    except Exception as clone_err:
                        logger.warning(f"Single commit clone recovery failed: {clone_err}")

                # Handle permission errors by re-initializing the work tree
                if "Permission denied" in error_msg:
                    logger.warning("Permission error detected, reinitializing work tree")
                    try:
                        subprocess.run(
                            ['chmod', '-R', '+w', repo_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=60
                        )
                    except Exception as chmod_err:
                        logger.warning(f"Chmod failed: {chmod_err}")

                # Clean up repository for next attempt
                try:
                    # Try a more thorough cleanup
                    subprocess.run(['git', 'reset', '--hard'], cwd=repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['git', 'clean', '-fdx'], cwd=repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['git', 'gc', '--prune=now'], cwd=repo_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as clean_err:
                    logger.warning(f"Repository cleanup failed: {clean_err}")

                # Wait before next attempt
                time.sleep(2 + attempt)  # Increasing wait time with each attempt

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for checkout of commit {commit_hash}: {e}")
                time.sleep(2 + attempt)

        return False

    def analyze_commit_parallel(self, commit: Commit, worker_id: int) -> Optional[CommitMetrics]:
        """
        Analyze a single commit and extract metrics (parallel version).

        Args:
            commit: Git commit object to analyze
            worker_id: ID of the worker performing the analysis

        Returns:
            CommitMetrics object with calculated metrics or None if an error occurs
        """
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
                logger.error(f"Worker {worker_id}: GitPython file handle error extracting commit info: {e}")
                with self.lock:
                    try:
                        self.skipped_commits.add(commit.hexsha if hasattr(commit, 'hexsha') else "unknown")
                    except ValueError:
                        self.skipped_commits.add("unknown")
                return None
            else:
                raise
        except Exception as extract_err:
            logger.error(f"Worker {worker_id}: Error extracting commit info: {extract_err}")
            with self.lock:
                try:
                    self.skipped_commits.add(commit.hexsha if hasattr(commit, 'hexsha') else "unknown")
                except ValueError:
                    self.skipped_commits.add("unknown")
            return None

        try:
            # Validate commit hash format
            if not self._is_valid_commit_hash(commit_info['hexsha']):
                logger.error(f"Worker {worker_id}: Invalid commit hash: {commit_info['hexsha']}, skipping")
                with self.lock:
                    self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Check if commit has already been analyzed or skipped
            if commit_info['hexsha'] in self.skipped_commits:
                logger.info(f"Worker {worker_id}: Commit {commit_info['hexsha'][:8]} previously skipped, skipping again")
                return None

            logger.info(f"Worker {worker_id}: Analyzing commit: {commit_info['hexsha'][:8]} - {commit_info['message'].splitlines()[0] if commit_info['message'] else 'No message'}")

            # Worker directory setup
            worker_dir = self.worker_dirs[worker_id]
            checkout_path = os.path.join(worker_dir, "checkout")
            repo_path = os.path.join(worker_dir, "repo")

            # Create a clean checkout directory
            if not self._prepare_checkout_directory(checkout_path):
                with self.lock:
                    self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Checkout the commit in the worker's repository using subprocess to avoid GitPython issues
            checkout_success = False
            try:
                # First try with subprocess directly
                process = subprocess.run(
                    ['git', 'checkout', commit_info['hexsha']],
                    cwd=repo_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=60
                )
                checkout_success = (process.returncode == 0)

                if not checkout_success:
                    error_msg = process.stderr.decode('utf-8', errors='ignore')
                    logger.warning(f"Worker {worker_id}: Checkout failed: {error_msg}")

                    # Try with safe_checkout as fallback
                    checkout_success = self.safe_checkout(commit_info['hexsha'], repo_path)
            except Exception as checkout_err:
                logger.error(f"Worker {worker_id}: Checkout error: {checkout_err}")
                checkout_success = self.safe_checkout(commit_info['hexsha'], repo_path)

            if not checkout_success:
                logger.error(f"Worker {worker_id}: Unable to checkout commit {commit_info['hexsha']}, skipping")
                with self.lock:
                    self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Use secure copy method instead of shutil.copytree
            if not self._secure_copy_directory(repo_path, checkout_path):
                logger.error(f"Worker {worker_id}: Unable to copy repository to checkout directory, skipping")
                with self.lock:
                    self.skipped_commits.add(commit_info['hexsha'])
                return None
            
            try:
                shutil.rmtree(os.path.join(checkout_path, 'old'), ignore_errors=True)
            except Exception as e:
                logger.warning(f"Worker {worker_id}: Could not remove 'old/' directory: {e}")

            # Proceed with the actual analysis
            # Pass original commit object but also the extracted info
            try:
                return self._analyze_commit_common(commit, checkout_path, commit_info['author_name'], commit_info['commit_date'])
            except ValueError as ve:
                if "read of closed file" in str(ve):
                    logger.error(f"Worker {worker_id}: GitPython file handle error during analysis: {ve}")
                    with self.lock:
                        self.skipped_commits.add(commit_info['hexsha'])
                    return None
                else:
                    raise
                
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error analyzing commit {commit_info.get('hexsha', 'unknown')}: {e}")
            # Add commit to skipped list
            with self.lock:
                self.skipped_commits.add(commit_info.get('hexsha', 'unknown'))
            return None
    
    def analyze_commit_sequential(self, commit: Commit) -> Optional[CommitMetrics]:
        """
        Analyze a single commit and extract metrics (sequential version).

        Args:
            commit: Git commit object to analyze

        Returns:
            CommitMetrics object with calculated metrics or None if an error occurs
        """
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

            # Checkout the commit in the main repository using subprocess to avoid GitPython issues
            checkout_success = False
            try:
                # First try with subprocess directly
                process = subprocess.run(
                    ['git', 'checkout', commit_info['hexsha']],
                    cwd=self.repo_info.local_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=60
                )
                checkout_success = (process.returncode == 0)

                if not checkout_success:
                    error_msg = process.stderr.decode('utf-8', errors='ignore')
                    logger.warning(f"Checkout failed: {error_msg}")

                    # Try with safe_checkout as fallback
                    checkout_success = self.safe_checkout(commit_info['hexsha'], self.repo_info.local_path)
            except Exception as checkout_err:
                logger.error(f"Checkout error: {checkout_err}")
                checkout_success = self.safe_checkout(commit_info['hexsha'], self.repo_info.local_path)

            if not checkout_success:
                logger.error(f"Unable to checkout commit {commit_info['hexsha']}, skipping")
                self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Use secure copy method instead of shutil.copytree
            if not self._secure_copy_directory(self.repo_info.local_path, checkout_path):
                logger.error(f"Unable to copy repository to checkout directory, skipping")
                self.skipped_commits.add(commit_info['hexsha'])
                return None

            # Proceed with the actual analysis
            # Pass original commit object but also the extracted info
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
            # Add commit to skipped list
            self.skipped_commits.add(commit_info.get('hexsha', 'unknown'))
            return None
        
    def _is_valid_commit_hash(self, commit_hash: str) -> bool:
        """
        Check if a given string is a valid commit hash.

        Args:
            commit_hash: The commit hash to validate

        Returns:
            True if valid, False otherwise
        """
        # Check if it's a valid hexadecimal string
        if not commit_hash or not all(c in '0123456789abcdefABCDEF' for c in commit_hash):
            return False

        # Check length (full SHA-1 is 40 chars)
        if len(commit_hash) != 40:
            return False

        return True
    
    def _prepare_checkout_directory(self, checkout_path: str) -> bool:
        """Prepare a clean checkout directory with proper permissions."""
        if os.path.exists(checkout_path):
            try:
                # First, fix any permission issues
                subprocess.run(
                    ['chmod', '-R', '+w', checkout_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60
                )

                # Then remove the directory
                shutil.rmtree(checkout_path)
            except Exception as e:
                logger.error(f"Error cleaning checkout directory: {e}")
                # Try an alternative approach with system commands
                try:
                    subprocess.run(
                        ['rm', '-rf', checkout_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=120
                    )
                except Exception as rm_err:
                    logger.error(f"Force remove failed: {rm_err}")
                    return False

        try:
            os.makedirs(checkout_path, exist_ok=True)
            # Set proper permissions
            os.chmod(checkout_path, 0o755)  # rwxr-xr-x permissions
            return True
        except Exception as e:
            logger.error(f"Error creating checkout directory: {e}")
            return False
    
    def _analyze_commit_common(self, commit: Commit, checkout_path: str, author_name: str, commit_date: datetime.datetime) -> Optional[CommitMetrics]:
        """
        Common commit analysis logic used by both sequential and parallel modes.

        Args:
            commit: Git commit object
            checkout_path: Path to checkout directory
            author_name: Name of the commit author
            commit_date: Commit date and time

        Returns:
            CommitMetrics object with calculated metrics or None if an error occurs
        """
        # Verify commit object integrity before proceeding
        try:
            # Validate crucial commit properties - these will fail for corrupt commits
            _ = commit.tree  # Check tree access
            for parent in commit.parents:
                _ = parent.tree  # Check parent trees

            # Access commit stats to verify they're available
            _ = commit.stats

        except (ValueError, KeyError, GitCommandError, AttributeError) as e:
            error_msg = str(e)
            if any(err_text in error_msg for err_text in ["Odd-length string", "Non-hexadecimal", "could not be resolved", "does not exist", "not a valid"]):
                logger.error(f"Invalid Git object reference in commit {commit.hexsha}: {error_msg}")
                return None
            else:
                logger.warning(f"Unexpected error accessing commit {commit.hexsha}: {error_msg}")
                # Continue execution despite warning - it might still work

        try:
            # Calculate LOC and changed files
            loc_added, loc_deleted, files_changed, changed_files = self._calculate_diff_stats(commit, checkout_path)

            # If no files were changed or we couldn't determine changes, skip the commit
            if not changed_files and files_changed == 0:
                logger.warning(f"No valid changes detected in commit {commit.hexsha}, skipping")
                return None

            # Update edited files frequency counter
            self._update_edited_files_counter(changed_files)

            # Calculate cyclomatic complexity for Python files
            commit_complexity, project_complexity = self._calculate_complexity(checkout_path, changed_files)

            # Calculate code quality metrics
            total_smells, total_warnings = self._calculate_code_quality(checkout_path, changed_files)

            # Calculate smell density
            total_loc, smell_density = self._calculate_smell_density(checkout_path, total_smells)

            # Author metrics - requires synchronization in parallel mode
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
                # Use CodeAnalysisUtils static method instead of class method
                commit_goals = CodeAnalysisUtils._classify_commit_message(commit.message)
            except Exception as classify_err:
                logger.warning(f"Error classifying commit message: {classify_err}")
                # Use default values if classification fails
                commit_goals = {
                    "new_feature": 0,
                    "bug_fixing": 1 if is_bug_fix else 0,  # Use is_bug_fix as fallback
                    "enhancement": 0,
                    "refactoring": 0
                }

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
                smell_density=smell_density,
                num_warnings=total_warnings,
                is_pr=is_pr,
                changed_files=changed_files,
                is_bug_fix=is_bug_fix, 
                repo_name=self.repo_info.name,
                new_feature=commit_goals.get('new_feature', 0),
                bug_fixing=commit_goals.get('bug_fixing', 0),
                enhancement=commit_goals.get('enhancement', 0),
                refactoring=commit_goals.get('refactoring', 0)
            )

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing commit {commit.hexsha} content: {e}")
            # Return None to indicate failure
            return None
    
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
                    # Manual line counting
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
    
    def _update_edited_files_counter(self, changed_files: List[str]):
        """Update the edited files frequency counter."""
        for file_path in changed_files:
            if CodeAnalysisUtils.is_code_file(file_path):
                if not self.sequential_mode:
                    with self.lock:
                        self.edited_files_freq[file_path] = self.edited_files_freq.get(file_path, 0) + 1
                else:
                    self.edited_files_freq[file_path] = self.edited_files_freq.get(file_path, 0) + 1
    
    def _update_bug_frequency(self, changed_files: List[str]):
        """Update the bug frequency counter for files in a bug fix."""
        for file_path in changed_files:
            if CodeAnalysisUtils.is_code_file(file_path):
                if not self.sequential_mode:
                    with self.lock:
                        self.bugged_files_freq[file_path] = self.bugged_files_freq.get(file_path, 0) + 1
                else:
                    self.bugged_files_freq[file_path] = self.bugged_files_freq.get(file_path, 0) + 1
    
    def _calculate_complexity(self, checkout_path: str, changed_files: List[str]) -> Tuple[int, int]:
        """Calculate cyclomatic complexity for changed files and the entire project."""
        commit_complexity = 0
        project_complexity = 0
        
        # Changed files complexity
        for file_path in changed_files:
            try:
                full_path = os.path.join(checkout_path, file_path)
                if os.path.exists(full_path) and CodeAnalysisUtils.is_code_file(file_path):
                    try:
                        commit_complexity += CodeAnalysisUtils.calculate_cyclomatic_complexity(full_path)
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
                            project_complexity += CodeAnalysisUtils.calculate_cyclomatic_complexity(full_path)
                        except Exception as e:
                            logger.warning(f"Error calculating complexity for {full_path}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing project files: {e}")
        
        return commit_complexity, project_complexity
    
    def _calculate_code_quality(self, checkout_path: str, changed_files: List[str]) -> Tuple[int, int]:
        """Calculate code quality metrics for changed files."""
        total_smells = 0
        total_warnings = 0
        
        for file_path in changed_files:
            try:
                full_path = os.path.join(checkout_path, file_path)
                if os.path.exists(full_path) and CodeAnalysisUtils.is_code_file(file_path):
                    try:
                        total_smells += CodeAnalysisUtils.detect_code_smells(full_path)
                    except Exception as sm_err:
                        logger.warning(f"Error detecting code smells for {file_path}: {sm_err}")
                    
                    try:
                        total_warnings += CodeAnalysisUtils.detect_warnings(full_path)
                    except Exception as w_err:
                        logger.warning(f"Error detecting warnings for {file_path}: {w_err}")
            except Exception as e:
                logger.warning(f"Error analyzing file {file_path}: {e}")
        
        return total_smells, total_warnings
    
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
        if not self.sequential_mode:
            with self.lock:
                return self._update_author_metrics_internal(author_name, commit_date)
        else:
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
    
    def _secure_copy_directory(self, src: str, dst: str) -> bool:
        """
        A more reliable directory copy method that handles permission issues.

        Args:
            src: Source directory
            dst: Destination directory

        Returns:
            True if copy successful, False otherwise
        """
        try:
            # First approach: try rsync if available (handles permissions better)
            try:
                # Check if rsync is available
                subprocess.run(['rsync', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Use rsync for copying
                subprocess.run(
                    ['rsync', '-a', '--delete', '--exclude=.git/objects/pack/', f"{src}/", f"{dst}/"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=300
                )

                # Copy the pack files separately with proper permissions
                pack_src = os.path.join(src, '.git/objects/pack')
                pack_dst = os.path.join(dst, '.git/objects/pack')

                if os.path.exists(pack_src):
                    os.makedirs(pack_dst, exist_ok=True)
                    # Set permissions on destination pack directory
                    os.chmod(pack_dst, 0o755)

                    # Copy pack files with correct permissions
                    for file in os.listdir(pack_src):
                        src_file = os.path.join(pack_src, file)
                        dst_file = os.path.join(pack_dst, file)

                        # Create the file with proper permissions first
                        with open(dst_file, 'wb') as f:
                            pass
                        os.chmod(dst_file, 0o644)  # rw-r--r--

                        # Now copy the content
                        shutil.copyfile(src_file, dst_file)

                return True

            except (subprocess.SubprocessError, FileNotFoundError):
                # Rsync not available or failed, fall back to manual copy
                pass

            # Second approach: manual copy with explicit permission management
            for root, dirs, files in os.walk(src):
                # Create relative path
                rel_path = os.path.relpath(root, src)
                dst_path = os.path.join(dst, rel_path) if rel_path != '.' else dst

                # Create directories with proper permissions
                os.makedirs(dst_path, exist_ok=True)
                os.chmod(dst_path, 0o755)  # rwxr-xr-x

                # Copy files with proper permissions
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dst_path, file)

                    # Skip if destination exists and is read-only (like in .git/objects)
                    if os.path.exists(dst_file) and not os.access(dst_file, os.W_OK):
                        continue
                    
                    # Create the file with proper permissions first
                    with open(dst_file, 'wb') as f:
                        pass
                    os.chmod(dst_file, 0o644)  # rw-r--r--

                    # Now copy the content
                    with open(src_file, 'rb') as src_f, open(dst_file, 'wb') as dst_f:
                        dst_f.write(src_f.read())

            return True

        except Exception as e:
            logger.error(f"Error in secure copy: {e}")
            return False
    
    def analyze_repository(self, max_commits: Optional[int] = None):
        """
        Analyze the repository, choosing between parallel or sequential mode.
        
        Args:
            max_commits: Maximum number of commits to analyze (None for all)
        """
        # Get all commits ordered by date
        all_commits = self._get_ordered_commits()
        
        if max_commits:
            all_commits = all_commits[:max_commits]
        
        # Find starting point for analysis
        start_index = self._find_resume_point(all_commits)
        
        # Check if we've already analyzed all commits
        if start_index >= len(all_commits):
            logger.info("All commits have already been analyzed.")
            return
        
        total_remaining = len(all_commits) - start_index
        
        # Choose analysis mode based on configuration
        if self.sequential_mode:
            logger.info(f"Sequential analysis of {total_remaining} commits...")
            self._analyze_sequential(all_commits[start_index:], total_remaining)
        else:
            logger.info(f"Parallel analysis of {total_remaining} commits with {self.max_workers} workers...")
            self._analyze_parallel(all_commits[start_index:], total_remaining)
        
        # Create a report on the analysis status
        self._create_analysis_summary(all_commits)
    
    def _get_ordered_commits(self) -> List[Commit]:
        """Get all commits ordered by date (oldest first)."""
        try:
            # Use subprocess directly to get commit hashes, avoiding GitPython file handle issues
            try:
                output = subprocess.check_output(
                    ['git', 'log', '--pretty=format:%H', '--reverse'],
                    cwd=self.repo_info.local_path,
                    text=True,
                    timeout=300
                )
                # Split the output and filter out any empty lines
                commit_hashes = [h for h in output.strip().split('\n') if h]

                # Now use GitPython to get commit objects, but with better protection
                commits = []
                for hash_chunk in [commit_hashes[i:i+100] for i in range(0, len(commit_hashes), 100)]:
                    # Process commits in chunks to avoid too many open files
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

                    # Force garbage collection after each chunk to release file handles
                    import gc
                    gc.collect()

                return commits

            except subprocess.SubprocessError as e:
                logger.warning(f"Subprocess error getting commits: {e}, falling back to GitPython")
                # Fall back to GitPython but in a safer way
                all_commits = self._safe_git_operation("iter_commits", lambda: list(self.repo.iter_commits()))
                if all_commits:
                    all_commits.reverse()  # Oldest to newest
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
                # Find the index of the commit to resume from
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
        
        # Generate an error report
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
        
        logger.info(f"Error report saved to {error_report_file}")
        
        # Remove checkpoint file if all commits have been processed
        if os.path.exists(self.checkpoint_file) and (analyzed_count + skipped_count >= total_count):
            os.remove(self.checkpoint_file)
            logger.info("Analysis complete, checkpoint removed.")
    
    def _analyze_sequential(self, commits: List[Commit], total_remaining: int):
        """
        Analyze commits sequentially.

        Args:
            commits: List of commits to analyze
            total_remaining: Total number of remaining commits
        """
        try:
            for i, commit in enumerate(commits):
                # Skip already analyzed commits
                if commit.hexsha in self.analyzed_commits:
                    logger.info(f"Commit {commit.hexsha[:8]} already analyzed, skipping")
                    continue
                
                # Analyze the commit
                metrics = self.analyze_commit_sequential(commit)
                if metrics:
                    self.commit_metrics.append(metrics)
                    self.analyzed_commits.add(metrics.commit_hash)

                    # Save results and checkpoint after EACH commit
                    self.save_results()
                    self.save_checkpoint(metrics.commit_hash)

                    # Print progress
                    progress = (i + 1) / total_remaining * 100
                    logger.info(f"Progress: {i+1}/{total_remaining} ({progress:.1f}%)")

            # Final save is redundant now but kept for safety
            self.save_results()

        except KeyboardInterrupt:
            logger.warning("\nManual interruption detected. State already saved.")
            logger.info("You can resume analysis with the --resume option")
            
    
    def _analyze_parallel(self, commits: List[Commit], total_remaining: int):
        """
        Analyze commits in parallel with better resource management.

        Args:
            commits: List of commits to analyze
            total_remaining: Total number of remaining commits
        """
        try:
            # Use a thread pool with a fixed worker count, to avoid too many file handles
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Use smaller batches to reduce memory pressure
                batch_size = min(100, len(commits))  # Smaller than before (reduced from 200)

                # Process in batches
                for batch_start in range(0, len(commits), batch_size):
                    batch_end = min(batch_start + batch_size, len(commits))
                    batch = commits[batch_start:batch_end]

                    logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(commits) + batch_size - 1)//batch_size} ({batch_end - batch_start} commits)")

                    # Submit only commits not yet analyzed
                    futures = {}
                    for i, commit in enumerate(batch):
                        if hasattr(commit, 'hexsha') and commit.hexsha not in self.analyzed_commits and commit.hexsha not in self.skipped_commits:
                            # Small delay between submissions to avoid lock conflicts
                            time.sleep(0.1)
                            futures[executor.submit(self.analyze_commit_parallel, commit, i % self.max_workers)] = commit

                    # If no commits to analyze, move to next batch
                    if not futures:
                        logger.info(f"No commits to analyze in this batch, moving to next")
                        continue
                    
                    # Process results as they arrive
                    completed_count = 0

                    # Add regular garbage collection during processing
                    gc_counter = 0

                    for future in concurrent.futures.as_completed(futures):
                        commit = futures[future]
                        try:
                            metrics = future.result()
                            if metrics:
                                with self.lock:
                                    self.commit_metrics.append(metrics)
                                    self.analyzed_commits.add(metrics.commit_hash)

                                    # Save after each commit
                                    self.save_results()
                                    self.save_checkpoint(metrics.commit_hash)

                                completed_count += 1

                                # Print progress
                                progress = (batch_start + completed_count) / len(commits) * 100
                                logger.info(f"Progress: {batch_start + completed_count}/{len(commits)} ({progress:.1f}%)")

                                # Periodic garbage collection to avoid memory leaks
                                gc_counter += 1
                                if gc_counter >= 5:  # More frequent garbage collection
                                    import gc
                                    gc.collect()
                                    gc_counter = 0

                        except Exception as e:
                            logger.error(f"Error processing commit: {e}")
                            with self.lock:
                                if hasattr(commit, 'hexsha'):
                                    self.skipped_commits.add(commit.hexsha)

                    # Force garbage collection after each batch
                    import gc
                    gc.collect()

                    # Wait a bit between batches to let git processes finish
                    time.sleep(2)

                    # Print batch summary
                    logger.info(f"Batch completed: {completed_count} commits analyzed, {len(batch) - completed_count} commits skipped")

                    # Clear variables to free memory
                    futures.clear()
                    del batch

                # Final save is redundant now but kept for safety
                self.save_results()

        except KeyboardInterrupt:
            logger.warning("\nManual interruption detected. State already saved.")
            logger.info("You can resume analysis with the --resume option")
    
    def save_checkpoint(self, last_commit_hash: str):
        """
        Save a checkpoint to resume analysis in case of error.
        
        Args:
            last_commit_hash: Hash of the last successfully analyzed commit
        """
        checkpoint_data = {
            'last_commit_hash': last_commit_hash,
            'author_stats': self.author_stats,
            'last_commit_by_author': {author: date.isoformat() for author, date in self.last_commit_by_author.items()},
            'metrics_count': len(self.commit_metrics),
            'skipped_commits': list(self.skipped_commits),
            'bugged_files_freq': self.bugged_files_freq,
            'edited_files_freq': self.edited_files_freq
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved to {self.checkpoint_file}")
    
    def load_checkpoint(self):
        """
        Load a previously saved checkpoint.
        
        Returns:
            Hash of the last analyzed commit or None
        """
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
            
            # Restore bug frequency and edit frequency counters
            self.bugged_files_freq = checkpoint_data.get('bugged_files_freq', {})
            self.edited_files_freq = checkpoint_data.get('edited_files_freq', {})
            
            # Load data from previously analyzed commits
            if os.path.exists(self.output_file):
                try:
                    df = pd.read_csv(self.output_file)
                    # Convert DataFrame rows to CommitMetrics objects
                    for _, row in df.iterrows():
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
                            smell_density=row.get('smell_density', 0.0),
                            num_warnings=row.get('num_warnings', 0),
                            is_pr=row.get('is_pr', False),
                            is_bug_fix=row.get('is_bug_fix', False),
                            repo_name=row.get('repo_name', self.repo_info.name),
                            new_feature=row.get('new_feature', 0),
                            bug_fixing=row.get('bug_fixing', 0),
                            enhancement=row.get('enhancement', 0),
                            refactoring=row.get('refactoring', 0)
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
        
    def _check_git_readiness(self):
        """Verify that GitPython is working correctly before starting analysis."""
        try:
            # Try basic git operations that might fail with "read of closed file"
            logger.info("Checking GitPython status...")

            # Check if we can access the repo head
            try:
                _ = self.repo.head.commit
            except Exception as e:
                if "read of closed file" in str(e):
                    logger.warning("GitPython file handle issue detected. Switching to subprocess for git operations.")
                    return False

            # Try to get a small number of commits
            try:
                _ = list(self.repo.iter_commits(max_count=5))
            except Exception as e:
                if "read of closed file" in str(e):
                    logger.warning("GitPython file handle issue detected in iter_commits. Switching to subprocess for git operations.")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Git readiness check failed: {e}")
            return False
    
    def save_results(self):
        """Save analysis results to CSV and JSON files."""
        if self.sequential_mode:
            # In sequential mode, no lock needed
            metrics_dicts = [asdict(m) for m in self.commit_metrics]
        else:
            # In parallel mode, use lock
            with self.lock:
                metrics_dicts = [asdict(m) for m in self.commit_metrics]
        
        # Convert dates to strings
        for m in metrics_dicts:
            m['date'] = m['date'].isoformat()
            # Remove changed files list (could be too long)
            m.pop('changed_files', None)
        
        df = pd.DataFrame(metrics_dicts)
        
        # Save as CSV
        df.to_csv(self.output_file, index=False)
        logger.info(f"Results saved to {self.output_file}")
        
        # Also save as JSON
        json_file = os.path.splitext(self.output_file)[0] + '.json'
        with open(json_file, 'w') as f:
            json.dump(metrics_dicts, f, indent=2)
        logger.info(f"Results also saved to {json_file}")
        
        # Save bug and edit frequency data
        freq_data = {
            'bugged_files': self.bugged_files_freq,
            'edited_files': self.edited_files_freq
        }
        
        freq_file = os.path.join(self.repo_info.output_dir, "file_frequencies.json")
        with open(freq_file, 'w') as f:
            json.dump(freq_data, f, indent=2)
        logger.info(f"File frequencies saved to {freq_file}")
        
        # Create summary report
        create_single_repo_report(self.repo_info.output_dir, df, self.repo_info.name)