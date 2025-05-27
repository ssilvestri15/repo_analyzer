"""Optimized parallel analyzer for multiple Git repositories."""

import os
import logging
import concurrent.futures
import json
import time
from typing import List, Optional, Dict, Set
from pathlib import Path

import pandas as pd

from ..models import RepositoryInfo
from ..analyzers.repo_analyzer import RepoAnalyzer
from ..utils import create_comparative_report

logger = logging.getLogger('repo_analyzer.multi_repo_analyzer')

class MultiRepoAnalyzer:
    """Optimized analyzer for multiple Git repositories with parallel processing capabilities."""
    
    def __init__(self, output_dir: str = "repo_analysis_results", max_workers: Optional[int] = None):
        """
        Initialize the multi-repository analyzer.
        
        Args:
            output_dir: Main directory for storing results
            max_workers: Maximum number of parallel worker processes (default: auto-calculated)
        """
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # If max_workers is None, it will use the default from ProcessPoolExecutor (os.cpu_count())
        self.max_workers = max_workers
        
        # Track analysis results and status
        self.all_results = []
        self.failed_repos = []
        self.analysis_stats = {
            'total_repos': 0,
            'successful_repos': 0,
            'failed_repos': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Set up checkpoint file
        self.checkpoint_file = os.path.join(self.output_dir, "multi_repo_checkpoint.json")
        
        logger.info(f"Multi-repository analyzer initialized. Results will be saved to: {self.output_dir}")
    
    def analyze_from_url_list(self, url_list: List[str], max_commits: Optional[int] = None, resume: bool = False):
        """
        Analyze multiple repositories from a list of URLs.
        
        Args:
            url_list: List of repository URLs to analyze
            max_commits: Maximum number of commits to analyze per repository
            resume: Flag to resume previously interrupted analysis
        """
        self.analysis_stats['total_repos'] = len(url_list)
        self.analysis_stats['start_time'] = time.time()
        
        # If resume is requested, load previous state
        completed_repos = set()
        if resume and os.path.exists(self.checkpoint_file):
            completed_repos = self._load_checkpoint()
            logger.info(f"Resuming analysis. {len(completed_repos)} repositories already completed.")
        
        # Filter out completed repositories if resuming
        if resume:
            pending_urls = [url for url in url_list if self._get_repo_name(url) not in completed_repos]
            logger.info(f"{len(pending_urls)}/{len(url_list)} repositories still need analysis.")
        else:
            pending_urls = url_list
        
        if not pending_urls:
            logger.info("All repositories have already been analyzed. Nothing to do.")
            return
        
        logger.info(f"Starting analysis of {len(pending_urls)} repositories in parallel")
        
        # Configure worker count based on repository count
        effective_workers = min(len(pending_urls), self.max_workers if self.max_workers else os.cpu_count())
        logger.info(f"Using {effective_workers} worker processes")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            # Prepare parameters for each repository
            futures = []
            for url in pending_urls:
                repo_name = self._get_repo_name(url)
                repo_output_dir = os.path.join(self.output_dir, repo_name)
                
                repo_info = RepositoryInfo(
                    url=url,
                    name=repo_name,
                    output_dir=repo_output_dir
                )
                
                # Submit the task
                future = executor.submit(self._analyze_single_repo, repo_info, max_commits, resume)
                futures.append((future, repo_name, url))
            
            # Process results as they complete
            for i, (future, repo_name, url) in enumerate(
                zip(concurrent.futures.as_completed([f for f, _, _ in futures]), 
                    [r for _, r, _ in futures],
                    [u for _, _, u in futures])
            ):
                try:
                    result = future.result()
                    if result:
                        self.all_results.append(result)
                        self.analysis_stats['successful_repos'] += 1
                        
                        # Update and save checkpoint
                        completed_repos.add(repo_name)
                        self._save_checkpoint(completed_repos)
                        
                        # Log progress
                        progress = (i + 1) / len(pending_urls) * 100
                        logger.info(f"Progress: {i+1}/{len(pending_urls)} repositories ({progress:.1f}%) - Successfully analyzed: {repo_name}")
                    else:
                        self.failed_repos.append((repo_name, url))
                        self.analysis_stats['failed_repos'] += 1
                        logger.warning(f"Repository analysis failed: {repo_name}")
                except Exception as e:
                    self.failed_repos.append((repo_name, url))
                    self.analysis_stats['failed_repos'] += 1
                    logger.error(f"Error analyzing repository {repo_name}: {e}")
        
        self.analysis_stats['end_time'] = time.time()
        
        # Combine results and generate comparative reports
        if self.all_results:
            self._combine_results()
            self._generate_analysis_summary()
    
    def analyze_from_file(self, file_path: str, max_commits: Optional[int] = None, resume: bool = False):
        """
        Analyze multiple repositories by reading URLs from a file.
        
        Args:
            file_path: Path to file containing repository URLs (one per line)
            max_commits: Maximum number of commits to analyze per repository
            resume: Flag to resume previously interrupted analysis
        """
        urls = []
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        except Exception as e:
            logger.error(f"Error reading URL file {file_path}: {e}")
            return
        
        logger.info(f"Found {len(urls)} URLs in file {file_path}")
        self.analyze_from_url_list(urls, max_commits, resume)
    
    def analyze_from_config(self, config_file: str, max_commits: Optional[int] = None, resume: bool = False):
        """
        Analyze repositories based on a JSON configuration file.
        
        Args:
            config_file: Path to JSON configuration file
            max_commits: Maximum number of commits to analyze per repository (overrides config)
            resume: Flag to resume previously interrupted analysis
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Extract repository URLs from config
            urls = []
            for repo_config in config.get('repositories', []):
                if 'url' in repo_config:
                    urls.append(repo_config['url'])
            
            # Get max_commits from config if not provided
            if max_commits is None and 'max_commits' in config:
                max_commits = config.get('max_commits')
            
            logger.info(f"Found {len(urls)} repositories in config file {config_file}")
            self.analyze_from_url_list(urls, max_commits, resume)
            
        except Exception as e:
            logger.error(f"Error processing config file {config_file}: {e}")
    
    def _get_repo_name(self, url: str) -> str:
        """Extract repository name from URL."""
        repo_name = url.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        return repo_name
    
    def _analyze_single_repo(self, repo_info: RepositoryInfo, max_commits: Optional[int], resume: bool) -> str:
        """
        Analyze a single repository with enhanced error handling.
        
        Args:
            repo_info: Repository information
            max_commits: Maximum number of commits to analyze
            resume: Flag to resume interrupted analysis
        
        Returns:
            Name of the analyzed repository or None if analysis failed
        """
        logger.info(f"Starting analysis of repository: {repo_info.name}")
        
        # Ensure output directory exists with proper permissions
        try:
            os.makedirs(repo_info.output_dir, exist_ok=True)
            os.chmod(repo_info.output_dir, 0o755)  # rwxr-xr-x
        except Exception as dir_err:
            logger.error(f"Error creating output directory for {repo_info.name}: {dir_err}")
            return None
        
        # Try different cloning strategies if needed
        strategies = [
            {"depth": 100, "branch": "master"},  # Start with a shallow clone of master
            {"depth": 100},                      # Try shallow clone without branch
            {"depth": 1000},                     # Try deeper clone
            {"mirror": True},                    # Try full mirror clone
            {}                                  # Finally try default clone
        ]
        
        for strategy_index, clone_options in enumerate(strategies):
            try:
                # Set up temporary directory with a unique name for this attempt
                temp_dir = os.path.join(repo_info.output_dir, f"temp_{strategy_index}_{int(time.time())}")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Set clone options
                repo_info.clone_options = clone_options
                repo_info.temp_dir = temp_dir
                repo_info.local_path = os.path.join(temp_dir, "repo")
                
                # Define worker count
                worker_count = max(1, (os.cpu_count() // 4) if self.max_workers is None else (self.max_workers // 4))
                
                # Log cloning strategy
                strategy_desc = ', '.join(f"{k}={v}" for k, v in clone_options.items()) if clone_options else "default"
                logger.info(f"Trying repository analysis with strategy {strategy_index+1}/{len(strategies)} ({strategy_desc}) for {repo_info.name}")
                
                # Create analyzer and run analysis
                analyzer = RepoAnalyzer(repo_info, resume, max_workers=worker_count)
                analyzer.analyze_repository(max_commits)
                analyzer.cleanup()
                
                logger.info(f"Successfully completed analysis of {repo_info.name} with strategy {strategy_index+1}")
                return repo_info.name
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Strategy {strategy_index+1} failed for {repo_info.name}: {error_msg}")
                
                # Clean up this attempt's temp directory
                try:
                    if os.path.exists(temp_dir):
                        subprocess.run(['chmod', '-R', '+w', temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        shutil.rmtree(temp_dir)
                except Exception as clean_err:
                    logger.warning(f"Error cleaning temp directory: {clean_err}")
                
                # If we've tried all strategies, give up
                if strategy_index == len(strategies) - 1:
                    logger.error(f"All cloning strategies failed for {repo_info.url}")
                    return None
                
                # Wait before trying next strategy
                time.sleep(2)
        
        return None  # Should not reach here, but just in case
    
    def _save_checkpoint(self, completed_repos: Set[str]):
        """Save the current analysis state to a checkpoint file."""
        checkpoint_data = {
            'completed_repos': list(completed_repos),
            'timestamp': time.time()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _load_checkpoint(self) -> Set[str]:
        """Load analysis state from checkpoint file."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            return set(checkpoint_data.get('completed_repos', []))
        except Exception as e:
            logger.error(f"Error loading checkpoint file: {e}")
            return set()
    
    def _combine_results(self):
        """Combine results from all repositories into a unified report."""
        all_metrics = []
        all_bugged_files = {}
        all_edited_files = {}

        # Collect data from individual repositories
        for repo_name in self.all_results:
            repo_output_dir = os.path.join(self.output_dir, repo_name)

            # Process commit metrics
            csv_file = os.path.join(repo_output_dir, "commit_metrics.csv")
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    # Add repository name column if not present
                    if 'repo_name' not in df.columns:
                        df['repo_name'] = repo_name
                    all_metrics.append(df)
                except Exception as e:
                    logger.error(f"Error reading metrics file {csv_file}: {e}")

            # Process file frequency data
            freq_file = os.path.join(repo_output_dir, "file_frequencies.json")
            if os.path.exists(freq_file):
                try:
                    with open(freq_file, 'r') as f:
                        freq_data = json.load(f)

                    # Add repository prefix to file paths
                    bugged_files = {f"{repo_name}:{file}": count 
                                   for file, count in freq_data.get('bugged_files', {}).items()}
                    edited_files = {f"{repo_name}:{file}": count 
                                   for file, count in freq_data.get('edited_files', {}).items()}

                    all_bugged_files.update(bugged_files)
                    all_edited_files.update(edited_files)
                except Exception as e:
                    logger.error(f"Error reading frequency file {freq_file}: {e}")

        if not all_metrics:
            logger.warning("No results to combine")
            return

        # Create a unified DataFrame with all data
        combined_df = pd.concat(all_metrics, ignore_index=True)

        # Save the combined CSV
        combined_file = os.path.join(self.output_dir, "all_repos_metrics.csv")
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"Combined metrics saved to {combined_file}")

        # Generate commit classification summary per repository
        classification_summary = self._generate_classification_summary(combined_df)
        classification_file = os.path.join(self.output_dir, "commit_classification_summary.json")
        with open(classification_file, 'w') as f:
            json.dump(classification_summary, f, indent=2)
        logger.info(f"Commit classification summary saved to {classification_file}")

        # Save combined frequency data
        combined_freq = {
            'bugged_files': all_bugged_files,
            'edited_files': all_edited_files
        }
        combined_freq_file = os.path.join(self.output_dir, "all_repos_frequencies.json")
        with open(combined_freq_file, 'w') as f:
            json.dump(combined_freq, f, indent=2)
        logger.info(f"Combined frequency data saved to {combined_freq_file}")

        # Create comparative report
        create_comparative_report(self.output_dir, combined_df, self.all_results)
    
    def _generate_classification_summary(self, combined_df: pd.DataFrame) -> Dict:
        """
        Generate a summary of commit classifications by repository.

        Args:
            combined_df: DataFrame with combined commit metrics

        Returns:
            Dictionary with classification summaries
        """
        # Initialize summary structure
        summary = {
            'overall': {
                'total_commits': len(combined_df),
                'new_feature': int(combined_df['new_feature'].sum()),
                'bug_fixing': int(combined_df['bug_fixing'].sum()),
                'enhancement': int(combined_df['enhancement'].sum()),
                'refactoring': int(combined_df['refactoring'].sum()),
                'percentages': {}
            },
            'by_repository': {}
        }

        # Calculate overall percentages
        total = summary['overall']['total_commits']
        if total > 0:
            summary['overall']['percentages'] = {
                'new_feature': round(summary['overall']['new_feature'] / total * 100, 2),
                'bug_fixing': round(summary['overall']['bug_fixing'] / total * 100, 2),
                'enhancement': round(summary['overall']['enhancement'] / total * 100, 2),
                'refactoring': round(summary['overall']['refactoring'] / total * 100, 2)
            }

        # Generate per-repository summaries
        for repo_name in combined_df['repo_name'].unique():
            repo_df = combined_df[combined_df['repo_name'] == repo_name]
            repo_total = len(repo_df)

            repo_summary = {
                'total_commits': repo_total,
                'new_feature': int(repo_df['new_feature'].sum()),
                'bug_fixing': int(repo_df['bug_fixing'].sum()),
                'enhancement': int(repo_df['enhancement'].sum()),
                'refactoring': int(repo_df['refactoring'].sum()),
                'percentages': {}
            }

            # Calculate repository percentages
            if repo_total > 0:
                repo_summary['percentages'] = {
                    'new_feature': round(repo_summary['new_feature'] / repo_total * 100, 2),
                    'bug_fixing': round(repo_summary['bug_fixing'] / repo_total * 100, 2),
                    'enhancement': round(repo_summary['enhancement'] / repo_total * 100, 2),
                    'refactoring': round(repo_summary['refactoring'] / repo_total * 100, 2)
                }

            summary['by_repository'][repo_name] = repo_summary

        return summary