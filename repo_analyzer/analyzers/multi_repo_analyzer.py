"""Optimized parallel analyzer for multiple Git repositories with parallel commit processing."""

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
from ..analyzers.parallel_repo_analyzer import RepoAnalyzerFactory
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
    
    def analyze_from_url_list(self, url_list: List[str], max_commits: Optional[int] = None, 
                             resume: bool = False, parallel_commits: bool = True, 
                             commit_workers: Optional[int] = None):
        """
        Analyze multiple repositories from a list of URLs with support for parallel commit processing.
        
        Args:
            url_list: List of repository URLs to analyze
            max_commits: Maximum number of commits to analyze per repository
            resume: Flag to resume previously interrupted analysis
            parallel_commits: Whether to use parallel commit processing for each repository
            commit_workers: Number of workers for parallel commit processing
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
        
        mode_text = "parallel" if parallel_commits else "sequential"
        worker_text = f" with {commit_workers} workers" if commit_workers and parallel_commits else ""
        logger.info(f"Starting analysis of {len(pending_urls)} repositories using {mode_text} commit processing{worker_text}")
        
        # Configure worker count based on repository count
        effective_workers = min(len(pending_urls), self.max_workers if self.max_workers else os.cpu_count())
        logger.info(f"Using {effective_workers} repository worker processes")
        
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
                
                # Submit the task with parallel/sequential mode selection
                future = executor.submit(
                    self._analyze_single_repo_with_mode, 
                    repo_info, 
                    max_commits, 
                    resume, 
                    parallel_commits, 
                    commit_workers
                )
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
    
    def analyze_from_file(self, file_path: str, max_commits: Optional[int] = None, 
                         resume: bool = False, parallel_commits: bool = True, 
                         commit_workers: Optional[int] = None):
        """
        Analyze multiple repositories by reading URLs from a file.
        
        Args:
            file_path: Path to file containing repository URLs (one per line)
            max_commits: Maximum number of commits to analyze per repository
            resume: Flag to resume previously interrupted analysis
            parallel_commits: Whether to use parallel commit processing
            commit_workers: Number of workers for parallel commit processing
        """
        urls = []
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        except Exception as e:
            logger.error(f"Error reading URL file {file_path}: {e}")
            return
        
        logger.info(f"Found {len(urls)} URLs in file {file_path}")
        self.analyze_from_url_list(urls, max_commits, resume, parallel_commits, commit_workers)
    
    def analyze_from_config(self, config_file: str, max_commits: Optional[int] = None, 
                           resume: bool = False, parallel_commits: bool = True, 
                           commit_workers: Optional[int] = None):
        """
        Analyze repositories based on a JSON configuration file.
        
        Args:
            config_file: Path to JSON configuration file
            max_commits: Maximum number of commits to analyze per repository (overrides config)
            resume: Flag to resume previously interrupted analysis
            parallel_commits: Whether to use parallel commit processing
            commit_workers: Number of workers for parallel commit processing
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
            
            # Get parallel settings from config if not provided
            if 'parallel_commits' in config and parallel_commits is None:
                parallel_commits = config.get('parallel_commits', True)
            
            if 'commit_workers' in config and commit_workers is None:
                commit_workers = config.get('commit_workers')
            
            logger.info(f"Found {len(urls)} repositories in config file {config_file}")
            self.analyze_from_url_list(urls, max_commits, resume, parallel_commits, commit_workers)
            
        except Exception as e:
            logger.error(f"Error processing config file {config_file}: {e}")
    
    def _get_repo_name(self, url: str) -> str:
        """Extract repository name from URL."""
        repo_name = url.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        return repo_name
    
    def _analyze_single_repo_with_mode(self, repo_info: RepositoryInfo, max_commits: Optional[int], 
                                      resume: bool, parallel_commits: bool, 
                                      commit_workers: Optional[int]) -> str:
        """
        Analyze a single repository with the specified processing mode.
        
        Args:
            repo_info: Repository information
            max_commits: Maximum number of commits to analyze
            resume: Flag to resume interrupted analysis
            parallel_commits: Whether to use parallel commit processing
            commit_workers: Number of workers for parallel processing
        
        Returns:
            Name of the analyzed repository or None if analysis failed
        """
        logger.info(f"Starting {'parallel' if parallel_commits else 'sequential'} analysis of repository: {repo_info.name}")
        
        # Ensure output directory exists with proper permissions
        try:
            os.makedirs(repo_info.output_dir, exist_ok=True)
            os.chmod(repo_info.output_dir, 0o755)  # rwxr-xr-x
        except Exception as dir_err:
            logger.error(f"Error creating output directory for {repo_info.name}: {dir_err}")
            return None
        
        # Try different strategies if analysis fails
        strategies = [
            {"parallel": parallel_commits, "workers": commit_workers},
            {"parallel": False, "workers": 1},  # Fallback to sequential
        ]
        
        # If we're already using sequential, don't try parallel as fallback
        if not parallel_commits:
            strategies = [{"parallel": False, "workers": 1}]
        
        for strategy_index, strategy in enumerate(strategies):
            try:
                logger.info(f"Trying analysis strategy {strategy_index+1}/{len(strategies)} "
                           f"({'parallel' if strategy['parallel'] else 'sequential'}) for {repo_info.name}")
                
                # Create analyzer using factory
                analyzer = RepoAnalyzerFactory.create_analyzer(
                    repo_info=repo_info,
                    resume=resume,
                    parallel=strategy["parallel"],
                    max_workers=strategy["workers"]
                )
                
                # Run analysis
                analyzer.analyze_repository(max_commits)
                analyzer.cleanup()
                
                logger.info(f"Successfully completed analysis of {repo_info.name} with strategy {strategy_index+1}")
                return repo_info.name
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Strategy {strategy_index+1} failed for {repo_info.name}: {error_msg}")
                
                # If we've tried all strategies, give up
                if strategy_index == len(strategies) - 1:
                    logger.error(f"All analysis strategies failed for {repo_info.url}")
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
    
    def _generate_analysis_summary(self):
        """Generate a final summary of the multi-repository analysis."""
        total_time = self.analysis_stats['end_time'] - self.analysis_stats['start_time']
        
        logger.info("=== MULTI-REPOSITORY ANALYSIS COMPLETED ===")
        logger.info(f"Total repositories: {self.analysis_stats['total_repos']}")
        logger.info(f"Successfully analyzed: {self.analysis_stats['successful_repos']}")
        logger.info(f"Failed repositories: {self.analysis_stats['failed_repos']}")
        logger.info(f"Success rate: {(self.analysis_stats['successful_repos'] / self.analysis_stats['total_repos'] * 100):.1f}%")
        logger.info(f"Total analysis time: {total_time:.2f} seconds")
        logger.info(f"Average time per repository: {(total_time / max(self.analysis_stats['total_repos'], 1)):.2f} seconds")
        
        if self.failed_repos:
            logger.warning("Failed repositories:")
            for repo_name, repo_url in self.failed_repos:
                logger.warning(f"  - {repo_name} ({repo_url})")
        
        # Generate performance statistics
        summary_file = os.path.join(self.output_dir, "multi_repo_summary.json")
        summary_data = {
            'analysis_stats': self.analysis_stats,
            'failed_repos': self.failed_repos,
            'successful_repos': self.all_results,
            'performance': {
                'total_time_seconds': total_time,
                'average_time_per_repo': total_time / max(self.analysis_stats['total_repos'], 1),
                'success_rate_percent': (self.analysis_stats['successful_repos'] / self.analysis_stats['total_repos'] * 100) if self.analysis_stats['total_repos'] > 0 else 0
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Multi-repository summary saved to {summary_file}")


class EnhancedMultiRepoAnalyzer(MultiRepoAnalyzer):
    """Enhanced version with advanced parallel processing and optimization features."""
    
    def __init__(self, output_dir: str = "repo_analysis_results", max_workers: Optional[int] = None,
                 enable_smart_scheduling: bool = True, enable_resource_monitoring: bool = True):
        """
        Initialize the enhanced multi-repository analyzer.
        
        Args:
            output_dir: Main directory for storing results
            max_workers: Maximum number of parallel worker processes
            enable_smart_scheduling: Enable intelligent scheduling based on repository size
            enable_resource_monitoring: Enable real-time resource monitoring
        """
        super().__init__(output_dir, max_workers)
        self.enable_smart_scheduling = enable_smart_scheduling
        self.enable_resource_monitoring = enable_resource_monitoring
        self.repo_priorities = {}  # Repository priority queue for smart scheduling
        
        if enable_resource_monitoring:
            self._setup_resource_monitoring()
    
    def _setup_resource_monitoring(self):
        """Setup real-time resource monitoring."""
        import psutil
        self.initial_memory = psutil.virtual_memory().percent
        self.initial_cpu = psutil.cpu_percent()
        logger.info(f"Resource monitoring enabled. Initial: Memory {self.initial_memory}%, CPU {self.initial_cpu}%")
    
    def analyze_from_url_list_enhanced(self, url_list: List[str], max_commits: Optional[int] = None,
                                     resume: bool = False, parallel_commits: bool = True,
                                     commit_workers: Optional[int] = None, 
                                     adaptive_workers: bool = True):
        """
        Enhanced analysis with adaptive worker management and smart scheduling.
        
        Args:
            url_list: List of repository URLs to analyze
            max_commits: Maximum number of commits to analyze per repository
            resume: Flag to resume previously interrupted analysis
            parallel_commits: Whether to use parallel commit processing
            commit_workers: Number of workers for parallel commit processing
            adaptive_workers: Enable adaptive worker count based on system load
        """
        if self.enable_smart_scheduling:
            # Pre-analyze repositories to determine optimal processing order
            url_list = self._smart_schedule_repositories(url_list)
        
        # Monitor resources and adjust workers if needed
        if adaptive_workers and self.enable_resource_monitoring:
            optimal_workers = self._calculate_adaptive_workers()
            self.max_workers = optimal_workers
            logger.info(f"Adaptive workers: Using {optimal_workers} workers based on system resources")
        
        # Call the parent method with enhancements
        self.analyze_from_url_list(url_list, max_commits, resume, parallel_commits, commit_workers)
        
        # Generate enhanced reports
        self._generate_enhanced_reports()
    
    def _smart_schedule_repositories(self, url_list: List[str]) -> List[str]:
        """
        Intelligently schedule repositories based on size and complexity estimates.
        
        Args:
            url_list: Original list of repository URLs
            
        Returns:
            Reordered list optimized for parallel processing
        """
        logger.info("Analyzing repositories for smart scheduling...")
        
        repo_info_list = []
        for url in url_list:
            try:
                # Quick analysis to estimate repository complexity
                repo_name = self._get_repo_name(url)
                priority_score = self._estimate_repo_priority(url)
                repo_info_list.append((url, repo_name, priority_score))
            except Exception as e:
                logger.warning(f"Could not analyze {url} for scheduling: {e}")
                repo_info_list.append((url, self._get_repo_name(url), 1.0))  # Default priority
        
        # Sort by priority (higher priority = more complex, should start first)
        repo_info_list.sort(key=lambda x: x[2], reverse=True)
        
        optimized_order = [info[0] for info in repo_info_list]
        
        logger.info("Smart scheduling completed:")
        for i, (url, repo_name, priority) in enumerate(repo_info_list[:5]):  # Show top 5
            logger.info(f"  {i+1}. {repo_name} (priority: {priority:.2f})")
        
        return optimized_order
    
    def _estimate_repo_priority(self, url: str) -> float:
        """
        Estimate repository processing priority based on available information.
        Higher priority = should be processed first.
        
        Args:
            url: Repository URL
            
        Returns:
            Priority score (higher = more priority)
        """
        try:
            # If we can access the repo quickly, estimate based on basic metrics
            import subprocess
            import tempfile
            
            # Quick shallow clone to get basic info
            with tempfile.TemporaryDirectory() as temp_dir:
                clone_path = os.path.join(temp_dir, "quick_clone")
                
                # Very shallow clone just to get basic info
                result = subprocess.run(
                    ['git', 'clone', '--depth', '10', url, clone_path],
                    capture_output=True, text=True, timeout=60
                )
                
                if result.returncode != 0:
                    return 1.0  # Default priority if can't clone
                
                # Count Python files
                python_files = 0
                total_size = 0
                for root, dirs, files in os.walk(clone_path):
                    for file in files:
                        if file.endswith('.py'):
                            python_files += 1
                            try:
                                file_path = os.path.join(root, file)
                                total_size += os.path.getsize(file_path)
                            except:
                                pass
                
                # Get commit count (limited by shallow clone)
                try:
                    commit_result = subprocess.run(
                        ['git', 'rev-list', '--count', 'HEAD'],
                        cwd=clone_path, capture_output=True, text=True, timeout=10
                    )
                    commit_count = int(commit_result.stdout.strip()) if commit_result.returncode == 0 else 10
                except:
                    commit_count = 10
                
                # Calculate priority score
                # Factors: number of Python files, total size, estimated commits
                file_factor = min(python_files / 100, 2.0)  # Normalized
                size_factor = min(total_size / (1024 * 1024), 2.0)  # Size in MB, normalized
                commit_factor = min(commit_count / 1000, 2.0)  # Normalized
                
                priority = (file_factor + size_factor + commit_factor) / 3
                return max(0.1, priority)  # Minimum priority
                
        except Exception as e:
            logger.debug(f"Could not estimate priority for {url}: {e}")
            return 1.0  # Default priority
    
    def _calculate_adaptive_workers(self) -> int:
        """
        Calculate optimal number of workers based on current system resources.
        
        Returns:
            Optimal worker count
        """
        import psutil
        
        # Get current system state
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Base calculation
        base_workers = self.max_workers or max(2, cpu_count // 2)
        
        # Adjust based on memory usage
        if memory_percent > 80:
            memory_factor = 0.5  # Reduce workers significantly
        elif memory_percent > 60:
            memory_factor = 0.7  # Reduce workers moderately
        else:
            memory_factor = 1.0  # No reduction
        
        # Adjust based on CPU usage
        if cpu_percent > 80:
            cpu_factor = 0.6  # Reduce workers
        elif cpu_percent > 60:
            cpu_factor = 0.8  # Reduce workers slightly
        else:
            cpu_factor = 1.0  # No reduction
        
        # Calculate final worker count
        adaptive_workers = int(base_workers * memory_factor * cpu_factor)
        adaptive_workers = max(1, min(adaptive_workers, cpu_count))  # Bounds check
        
        logger.debug(f"Adaptive workers calculation: Memory {memory_percent}%, CPU {cpu_percent}%, "
                    f"Base {base_workers} -> Final {adaptive_workers}")
        
        return adaptive_workers
    
    def _generate_enhanced_reports(self):
        """Generate enhanced reports with performance analytics."""
        if not self.enable_resource_monitoring:
            return
        
        try:
            import psutil
            
            # Performance metrics
            final_memory = psutil.virtual_memory().percent
            final_cpu = psutil.cpu_percent()
            
            performance_report = {
                'resource_usage': {
                    'initial_memory_percent': self.initial_memory,
                    'final_memory_percent': final_memory,
                    'memory_delta': final_memory - self.initial_memory,
                    'initial_cpu_percent': self.initial_cpu,
                    'final_cpu_percent': final_cpu,
                    'cpu_delta': final_cpu - self.initial_cpu
                },
                'optimization_features': {
                    'smart_scheduling_enabled': self.enable_smart_scheduling,
                    'resource_monitoring_enabled': self.enable_resource_monitoring,
                    'final_worker_count': self.max_workers
                },
                'recommendations': self._generate_performance_recommendations()
            }
            
            perf_file = os.path.join(self.output_dir, "performance_report.json")
            with open(perf_file, 'w') as f:
                json.dump(performance_report, f, indent=2)
            
            logger.info(f"Enhanced performance report saved to {perf_file}")
            
        except Exception as e:
            logger.error(f"Error generating enhanced reports: {e}")
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations based on analysis results."""
        recommendations = []
        
        if self.analysis_stats['failed_repos'] > 0:
            failure_rate = self.analysis_stats['failed_repos'] / self.analysis_stats['total_repos']
            if failure_rate > 0.2:
                recommendations.append("High failure rate detected. Consider using sequential mode for problematic repositories.")
        
        if self.enable_resource_monitoring:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 80:
                recommendations.append("High memory usage detected. Consider reducing max_workers or enabling adaptive workers.")
        
        total_time = self.analysis_stats.get('end_time', 0) - self.analysis_stats.get('start_time', 0)
        if total_time > 3600:  # More than 1 hour
            recommendations.append("Long analysis time. Consider using more parallel workers or analyzing fewer commits per repository.")
        
        if len(recommendations) == 0:
            recommendations.append("Analysis completed efficiently with current settings.")
        
        return recommendations