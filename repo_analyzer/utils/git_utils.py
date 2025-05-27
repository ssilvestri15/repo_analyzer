"""Utilities for Git repository management with enhanced error handling and recovery."""

import os
import shutil
import subprocess
import time
import logging
from typing import Dict, Any, Optional
from git import Repo, GitCommandError

logger = logging.getLogger('repo_analyzer.git_utils')

class GitRepositoryManager:
    """Class to manage Git repositories with robust error handling."""
    
    @staticmethod
    def clone_repository(url: str, local_path: str, options: Optional[Dict[str, Any]] = None) -> Repo:
        """
        Clone a Git repository from URL with enhanced error handling and recovery.
        
        Args:
            url: Repository URL
            local_path: Local path where to clone the repository
            options: Dictionary with cloning options:
                     - depth: Clone depth (e.g., 100)
                     - branch: Branch to clone (e.g., 'master')
                     - mirror: Whether to create a mirror (True/False)
        
        Returns:
            Repo object
        
        Raises:
            RuntimeError: If repository cannot be cloned after several attempts
        """
        options = options or {}
        logger.info(f"Cloning repository {url} to {local_path}")
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
        # Try subprocess-based cloning first as it's more reliable with large repos
        try:
            # Clean existing directory if needed
            if os.path.exists(local_path):
                try:
                    # Try to fix permission issues before removal
                    subprocess.run(
                        ['chmod', '-R', '+w', local_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=60
                    )
                    
                    # Then remove the directory
                    shutil.rmtree(local_path)
                except Exception as e:
                    logger.warning(f"Could not clean directory: {e}, trying system commands")
                    try:
                        subprocess.run(
                            ['rm', '-rf', local_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=120
                        )
                    except Exception as rm_err:
                        logger.warning(f"Force remove failed: {rm_err}")
                        # Use alternative path
                        local_path = f"{local_path}_{int(time.time())}"
            
            # Create directory
            os.makedirs(local_path, exist_ok=True)
            
            # Build git clone command with options
            cmd = ['git', 'clone']
            
            if options.get('depth'):
                cmd.extend(['--depth', str(options['depth'])])
            
            if options.get('branch'):
                cmd.extend(['--branch', options['branch']])
            
            if options.get('mirror'):
                cmd.append('--mirror')
            
            # Add options to reduce file handle issues
            cmd.append('--config=core.compression=0')  # Disable compression to save CPU
            
            cmd.extend([url, local_path])
            
            # Try cloning with subprocess
            result = subprocess.run(cmd, 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            timeout=600)  # 10-minute timeout for large repos
            
            if result.returncode == 0:
                # Success - create and return the Repo object
                return Repo(local_path)
            else:
                # Clone failed, log error and try GitPython fallback
                stderr = result.stderr.decode('utf-8', errors='ignore')
                logger.warning(f"Subprocess clone failed: {stderr}")
            
        except Exception as e:
            logger.warning(f"Subprocess-based clone failed: {e}")
        
        # If we reach here, the subprocess approach failed, try GitPython
        logger.info("Trying GitPython clone as fallback")
        
        # Try different cloning approaches
        methods = [
            # Method 1: GitPython's clone_from with minimal options
            lambda: GitRepositoryManager._clone_with_gitpython(url, local_path, options),
            
            # Method 2: Shallow clone first, then fetch
            lambda: GitRepositoryManager._clone_shallow_then_deepen(url, local_path, options),
            
            # Method 3: Clone without checkout, then fetch and checkout
            lambda: GitRepositoryManager._clone_without_checkout(url, local_path, options),
        ]
        
        # Try each method in order
        last_error = None
        for i, method in enumerate(methods):
            try:
                logger.info(f"Trying GitPython clone method {i+1}/{len(methods)}")
                repo = method()
                logger.info(f"Repository cloned successfully with GitPython method {i+1}")
                return repo
            except Exception as e:
                logger.warning(f"GitPython clone method {i+1} failed: {e}")
                last_error = e
                
                # Clean up for next attempt
                try:
                    if os.path.exists(local_path):
                        # Fix permissions
                        subprocess.run(
                            ['chmod', '-R', '+w', local_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=60
                        )
                        # Remove
                        shutil.rmtree(local_path)
                except Exception:
                    try:
                        subprocess.run(
                            ['rm', '-rf', local_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=120
                        )
                    except Exception:
                        pass
                    
                # Create directory again
                os.makedirs(local_path, exist_ok=True)
        
        # If all methods failed
        raise RuntimeError(f"Failed to clone repository after trying all methods: {last_error}")
    
    @staticmethod
    def _clone_with_gitpython(url: str, local_path: str, options: Dict[str, Any]) -> Repo:
        """Clone repository using GitPython."""
        clone_kwargs = {}
        
        if options.get('depth'):
            clone_kwargs['depth'] = options['depth']
        
        if options.get('branch'):
            clone_kwargs['branch'] = options['branch']
        
        if options.get('mirror'):
            clone_kwargs['mirror'] = True
        
        return Repo.clone_from(url, local_path, **clone_kwargs)
    
    @staticmethod
    def _clone_with_subprocess(url: str, local_path: str, options: Dict[str, Any]) -> Repo:
        """Clone repository using direct Git subprocess."""
        cmd = ['git', 'clone']
        
        if options.get('depth'):
            cmd.extend(['--depth', str(options['depth'])])
        
        if options.get('branch'):
            cmd.extend(['--branch', options['branch']])
        
        if options.get('mirror'):
            cmd.append('--mirror')
        
        cmd.extend([url, local_path])
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
        return Repo(local_path)
    
    @staticmethod
    def _clone_shallow_then_deepen(url: str, local_path: str, options: Dict[str, Any]) -> Repo:
        """Clone shallow first, then deepen the repository."""
        # Start with very shallow clone
        shallow_cmd = ['git', 'clone', '--depth', '1']
        
        if options.get('branch'):
            shallow_cmd.extend(['--branch', options['branch']])
        
        shallow_cmd.extend([url, local_path])
        
        subprocess.run(shallow_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
        
        # Now unshallow if needed
        if not options.get('depth') or options.get('depth') > 1:
            deepen_cmd = ['git', 'fetch', '--unshallow']
            subprocess.run(deepen_cmd, cwd=local_path, check=True, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
        
        return Repo(local_path)
    
    @staticmethod
    def _clone_without_checkout(url: str, local_path: str, options: Dict[str, Any]) -> Repo:
        """Clone without checkout, then fetch and checkout."""
        # Clone without checkout
        no_checkout_cmd = ['git', 'clone', '--no-checkout']
        
        if options.get('depth'):
            no_checkout_cmd.extend(['--depth', str(options['depth'])])
        
        no_checkout_cmd.extend([url, local_path])
        
        subprocess.run(no_checkout_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
        
        # Fetch all
        fetch_cmd = ['git', 'fetch', '--all']
        subprocess.run(fetch_cmd, cwd=local_path, check=True, 
                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
        
        # Checkout default branch
        checkout_cmd = ['git', 'checkout']
        if options.get('branch'):
            checkout_cmd.append(options['branch'])
        else:
            checkout_cmd.append('HEAD')
            
        subprocess.run(checkout_cmd, cwd=local_path, check=True, 
                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
        
        return Repo(local_path)
    
    @staticmethod
    def _clone_mirror_then_convert(url: str, local_path: str, options: Dict[str, Any]) -> Repo:
        """Clone as mirror then convert to normal repository."""
        # First create a temp directory for the mirror
        mirror_path = f"{local_path}_mirror"
        
        try:
            # Clone as mirror
            mirror_cmd = ['git', 'clone', '--mirror', url, mirror_path]
            subprocess.run(mirror_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
            
            # Now clone from the mirror to the actual path
            local_clone_cmd = ['git', 'clone']
            
            if options.get('depth'):
                local_clone_cmd.extend(['--depth', str(options['depth'])])
            
            if options.get('branch'):
                local_clone_cmd.extend(['--branch', options['branch']])
                
            local_clone_cmd.extend([mirror_path, local_path])
            
            subprocess.run(local_clone_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
            
            # Fix remote URL
            subprocess.run(['git', 'remote', 'set-url', 'origin', url], 
                         cwd=local_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            return Repo(local_path)
            
        finally:
            # Clean up mirror
            if os.path.exists(mirror_path):
                try:
                    shutil.rmtree(mirror_path)
                except Exception:
                    try:
                        subprocess.run(['rm', '-rf', mirror_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        pass
    
    @staticmethod
    def checkout_commit(repo: Repo, commit_hash: str, max_retries: int = 3) -> bool:
        """
        Checkout a specific commit with retry logic.
        
        Args:
            repo: Repo object
            commit_hash: Commit hash
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if checkout successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                repo.git.checkout(commit_hash)
                return True
            except GitCommandError as e:
                error_msg = str(e)
                logger.warning(f"Checkout attempt {attempt+1}/{max_retries} failed: {error_msg}")
                
                # Handle specific errors
                if "tree" in error_msg or "object not found" in error_msg:
                    # Try to recover missing objects
                    try:
                        # Fetch all objects
                        repo.git.fetch('--all', '--tags', '--force')
                        
                        # Run garbage collection
                        repo.git.gc('--aggressive', '--prune=now')
                        
                        # Try again after recovery
                        continue
                    except Exception as recovery_err:
                        logger.warning(f"Recovery failed: {recovery_err}")
                
                # Try system git command as fallback
                try:
                    subprocess.run(
                        ['git', 'checkout', commit_hash],
                        cwd=repo.working_dir,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                        timeout=60
                    )
                    return True
                except Exception:
                    pass
                
                # Clean and retry
                try:
                    repo.git.reset('--hard')
                    repo.git.clean('-fdx')
                except Exception:
                    pass
                
                # Wait before retrying
                time.sleep(2)
        
        return False
    
    @staticmethod
    def validate_repository(repo_path: str) -> bool:
        """
        Validate and attempt to repair a Git repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            True if repository is valid or was repaired, False otherwise
        """
        try:
            # Check if it's a valid Git repository
            repo = Repo(repo_path)
            _ = repo.git_dir
            
            # Run basic validation
            try:
                repo.git.fsck('--full')
                return True
            except GitCommandError as fsck_err:
                logger.warning(f"Repository validation failed: {fsck_err}")
                
                # Try to repair
                try:
                    # Fetch all objects
                    repo.git.fetch('--all', '--tags', '--force')
                    
                    # Run garbage collection
                    repo.git.gc('--aggressive', '--prune=now')
                    
                    # Check if repair was successful
                    repo.git.fsck('--full')
                    logger.info("Repository repair successful")
                    return True
                except Exception as repair_err:
                    logger.error(f"Repository repair failed: {repair_err}")
                    return False
        except Exception as e:
            logger.error(f"Invalid repository: {e}")
            return False