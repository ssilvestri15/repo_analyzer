"""Parallel analyzer per Git repositories con tracking degli smell ottimizzato."""

import os
import subprocess
import tempfile
import datetime
import json
import logging
import shutil
import threading
import time
import psutil
import multiprocessing
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import asdict
from git import Repo, Commit, GitCommandError
import concurrent.futures
from pathlib import Path
from collections import defaultdict

import pandas as pd

from ..models import CommitMetrics, RepositoryInfo
from ..utils import CodeAnalysisUtils, GitRepositoryManager, create_single_repo_report
from .smell_tracker import SmellEvolutionTracker, SmellDetection, SmellEventType

logger = logging.getLogger('repo_analyzer.parallel_repo_analyzer')

class ParallelRepoAnalyzer:
    """Analyzer parallelo per repository Git con isolamento completo per commit."""
    
    def __init__(self, repo_info: RepositoryInfo, resume: bool = False, max_workers: Optional[int] = None):
        """
        Initialize the parallel repository analyzer.
        
        Args:
            repo_info: Repository information
            resume: Flag to resume analysis from an interrupted commit
            max_workers: Maximum number of parallel workers
        """
        self.repo_info = repo_info
        self.resume = resume
        
        # Calcola il numero ottimale di worker
        self.max_workers = self._calculate_optimal_workers(max_workers)
        # Progress display integration
        self._progress = None

        # Set up directories
        self._setup_directories()
        
        # Initialize Git repository
        self._initialize_repository()
        
        # Thread-safe data structures
        self.lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        # Dati condivisi thread-safe
        self.commit_results = {}  # commit_hash -> result_data
        self.analyzed_commits = set()
        self.skipped_commits = set()
        self.processing_commits = set()
        
        # Carica lo stato precedente se resuming
        if resume:
            self._load_previous_state()
    
    def _calculate_optimal_workers(self, max_workers: Optional[int]) -> int:
        """Calcola il numero ottimale di worker basandosi sulle specifiche della macchina."""
        if max_workers:
            return max_workers
        
        # Ottieni le specifiche del sistema
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Algoritmo per calcolare i worker ottimali
        # Considera CPU, memoria e I/O disk
        
        # Base: usa 75% dei core disponibili
        base_workers = max(1, int(cpu_count * 0.75))
        
        # Limita in base alla memoria (assumendo ~1GB per worker)
        memory_limited = max(1, int(memory_gb * 0.8))
        
        # Per I/O intensivo, non superare 2x i core fisici
        io_limited = cpu_count * 2
        
        # Prendi il minimo tra i limiti
        optimal = min(base_workers, memory_limited, io_limited)
        
        # Assicurati che sia ragionevole (tra 1 e 16)
        optimal = max(1, min(optimal, 16))
        
        logger.info(f"Sistema rilevato: {cpu_count} CPU, {memory_gb:.1f}GB RAM")
        logger.info(f"Worker ottimali calcolati: {optimal}")
        
        return optimal
    
    def _setup_directories(self):
        """Set up the output and temporary directories."""
        # Set output directory
        if not self.repo_info.output_dir:
            self.repo_info.output_dir = self.repo_info.name
        
        os.makedirs(self.repo_info.output_dir, exist_ok=True)
        
        # Set temporary directory
        if not self.repo_info.temp_dir:
            self.repo_info.temp_dir = tempfile.mkdtemp(prefix=f"parallel_analyzer_{self.repo_info.name}_")
        else:
            os.makedirs(self.repo_info.temp_dir, exist_ok=True)
        
        # Directory per i risultati paralleli
        self.parallel_results_dir = os.path.join(self.repo_info.output_dir, "parallel_results")
        os.makedirs(self.parallel_results_dir, exist_ok=True)
        
        # Directory per i workspace dei commit
        self.commit_workspaces_dir = os.path.join(self.repo_info.temp_dir, "commit_workspaces")
        os.makedirs(self.commit_workspaces_dir, exist_ok=True)
        
        # Set local path for the main clone
        if not self.repo_info.local_path:
            self.repo_info.local_path = os.path.join(self.repo_info.temp_dir, "repo")
        
        # File di output e checkpoint
        self.output_file = os.path.join(self.repo_info.output_dir, "commit_metrics.csv")
        self.checkpoint_file = os.path.join(self.repo_info.output_dir, "parallel_checkpoint.json")
        self.smell_evolution_file = os.path.join(self.repo_info.output_dir, "smell_evolution.json")
        self.parallel_state_file = os.path.join(self.parallel_results_dir, "parallel_state.json")
        
        logger.info(f"Parallel analysis of {self.repo_info.url}")
        logger.info(f"Using {self.max_workers} parallel workers")
        logger.info(f"Workspace directory: {self.commit_workspaces_dir}")
        logger.info(f"Results will be saved to: {self.repo_info.output_dir}")
    
    def _initialize_repository(self):
        """Initialize the Git repository."""
        try:
            # Clone o usa repository esistente
            if not os.path.exists(self.repo_info.local_path):
                self.repo = GitRepositoryManager.clone_repository(
                    self.repo_info.url, 
                    self.repo_info.local_path,
                    options={"depth": None}  # Clone completo per accesso parallelo
                )
            else:
                self.repo = Repo(self.repo_info.local_path)
                # Aggiorna il repository
                try:
                    self.repo.git.fetch('--all')
                except Exception as e:
                    logger.warning(f"Could not fetch updates: {e}")
            
        except Exception as e:
            logger.error(f"Error initializing repository: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary directories."""
        if os.path.exists(self.repo_info.temp_dir):
            # Cleanup in modo sicuro per evitare problemi di permessi
            try:
                subprocess.run(['chmod', '-R', '+w', self.repo_info.temp_dir], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                shutil.rmtree(self.repo_info.temp_dir)
                logger.info(f"Temporary directory cleaned: {self.repo_info.temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory: {e}")
    
    def analyze_repository(self, max_commits: Optional[int] = None):
        """Analyze the repository using parallel processing."""
        logger.info("Starting parallel repository analysis...")
        
        # Update progress display if available
        if hasattr(self, '_progress') and self._progress:
            self._progress.log("Initializing parallel analysis...")
    
        # Ottieni tutti i commit ordinati
        all_commits = self._get_ordered_commits()
        
        if max_commits:
            all_commits = all_commits[:max_commits]
        
        # Update total commits in progress display
        if self._progress:
            self._progress.set_total_commits(len(all_commits))
            self._progress.log(f"Found {len(all_commits)} commits to analyze")
        
        # Trova il punto di ripresa
        start_index = self._find_resume_point(all_commits)
        
        if start_index >= len(all_commits):
            logger.info("All commits have already been analyzed.")
            self._finalize_analysis()
            return
        
        remaining_commits = all_commits[start_index:]
        total_remaining = len(remaining_commits)
        
        logger.info(f"Parallel analysis of {total_remaining} commits using {self.max_workers} workers...")
        
        # Analisi parallela
        self._analyze_parallel(remaining_commits)
        
        # Finalizza l'analisi
        self._finalize_analysis()
    
    def _analyze_parallel(self, commits: List[Commit]):
        """Esegue l'analisi parallela dei commit."""
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Sottometti tutti i task
            future_to_commit = {}
            for i, commit in enumerate(commits):
                if commit.hexsha not in self.analyzed_commits and commit.hexsha not in self.processing_commits:
                    future = executor.submit(self._analyze_single_commit, commit, i, len(commits))
                    future_to_commit[future] = commit
                    
                    with self.lock:
                        self.processing_commits.add(commit.hexsha)
            
            # Processa i risultati man mano che completano
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_commit):
                commit = future_to_commit[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    if result:
                        with self.results_lock:
                            self.commit_results[commit.hexsha] = result
                            self.analyzed_commits.add(commit.hexsha)

                        # AGGIORNA PROGRESS
                        if self._progress:
                            self._progress.increment_analyzed(commit.hexsha)
                            self._progress.log(f"Analyzed commit {commit.hexsha[:8]}")
                        
                        # Salva il risultato immediatamente
                        self._save_commit_result(commit.hexsha, result)
                    else:
                        with self.lock:
                            self.skipped_commits.add(commit.hexsha)
                        # AGGIORNA PROGRESS
                        if self._progress:
                            self._progress.increment_skipped()
                            self._progress.log(f"Skipped commit {commit.hexsha[:8]}")
                    
                    # Salva checkpoint periodicamente
                    if completed_count % 10 == 0:
                        self._save_parallel_state()
                        
                except Exception as e:
                    with self.lock:
                        self.skipped_commits.add(commit.hexsha)
                        self.processing_commits.discard(commit.hexsha)
                    # AGGIORNA PROGRESS
                    if self._progress:
                        self._progress.increment_skipped()
                        self._progress.log(f"Error in commit {commit.hexsha[:8]}: {str(e)[:40]}...")
                finally:
                    with self.lock:
                        self.processing_commits.discard(commit.hexsha)
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel analysis completed in {elapsed:.2f} seconds")
        
        # Salva stato finale
        self._save_parallel_state()
    
    def _analyze_single_commit(self, commit: Commit, index: int, total: int) -> Optional[Dict]:
        """Analizza un singolo commit in modo isolato."""
        commit_hash = commit.hexsha
        
        try:
            # Crea workspace isolato per questo commit
            workspace_dir = os.path.join(self.commit_workspaces_dir, f"commit_{commit_hash[:12]}")
            os.makedirs(workspace_dir, exist_ok=True)
            
            # Estrai informazioni base del commit
            commit_info = self._extract_commit_info(commit)
            if not commit_info:
                return None
            
            # Ottieni i file modificati
            changed_files = self._get_changed_files(commit)
            if not changed_files:
                logger.warning(f"No valid Python files changed in commit {commit_hash[:8]}")
                return None
            
            # Scarica solo i file modificati
            files_content = self._download_changed_files(commit, changed_files, workspace_dir)
            if not files_content:
                logger.warning(f"Could not download files for commit {commit_hash[:8]}")
                return None
            
            # Analizza i file scaricati
            metrics = self._analyze_commit_files(commit_info, changed_files, files_content, workspace_dir)
            
            # Cleanup workspace
            self._cleanup_workspace(workspace_dir)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in parallel analysis of commit {commit_hash[:8]}: {e}")
            # Cleanup in caso di errore
            workspace_dir = os.path.join(self.commit_workspaces_dir, f"commit_{commit_hash[:12]}")
            self._cleanup_workspace(workspace_dir)
            return None
    
    def _extract_commit_info(self, commit: Commit) -> Optional[Dict]:
        """Estrae le informazioni base del commit in modo thread-safe."""
        try:
            return {
                'commit_hash': commit.hexsha,
                'author': commit.author.name if hasattr(commit, 'author') and hasattr(commit.author, 'name') else "Unknown",
                'date': commit.committed_datetime if hasattr(commit, 'committed_datetime') else datetime.datetime.now(),
                'message': commit.message.strip() if hasattr(commit, 'message') and commit.message else "",
                'parents': [p.hexsha for p in commit.parents] if hasattr(commit, 'parents') else []
            }
        except Exception as e:
            logger.error(f"Error extracting commit info: {e}")
            return None
    
    def _get_changed_files(self, commit: Commit) -> List[str]:
        """Ottiene la lista dei file Python modificati nel commit."""
        changed_files = []
        
        try:
            if commit.parents:
                # Commit con genitori
                for parent in commit.parents:
                    try:
                        diffs = parent.diff(commit)
                        for diff in diffs:
                            if diff.b_path and CodeAnalysisUtils.is_code_file(diff.b_path):
                                changed_files.append(diff.b_path)
                            elif diff.a_path and CodeAnalysisUtils.is_code_file(diff.a_path):
                                changed_files.append(diff.a_path)
                    except Exception as e:
                        logger.warning(f"Error getting diff: {e}")
            else:
                # Commit iniziale - usa git show per ottenere i file
                try:
                    result = subprocess.run(
                        ['git', 'show', '--name-only', '--pretty=format:', commit.hexsha],
                        cwd=self.repo_info.local_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                        changed_files = [f for f in files if CodeAnalysisUtils.is_code_file(f)]
                except Exception as e:
                    logger.warning(f"Error getting initial commit files: {e}")
        
        except Exception as e:
            logger.error(f"Error getting changed files: {e}")
        
        return list(set(changed_files))  # Rimuovi duplicati
    
    def _download_changed_files(self, commit: Commit, changed_files: List[str], workspace_dir: str) -> Dict[str, str]:
        """Scarica solo i file modificati per il commit specifico."""
        files_content = {}
        
        try:
            for file_path in changed_files:
                try:
                    # Usa git show per ottenere il contenuto del file a questo commit
                    result = subprocess.run(
                        ['git', 'show', f"{commit.hexsha}:{file_path}"],
                        cwd=self.repo_info.local_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        # Crea la struttura di directory
                        full_file_path = os.path.join(workspace_dir, file_path)
                        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
                        
                        # Salva il file
                        with open(full_file_path, 'w', encoding='utf-8') as f:
                            f.write(result.stdout)
                        
                        files_content[file_path] = full_file_path
                    else:
                        logger.warning(f"Could not download {file_path} for commit {commit.hexsha[:8]}")
                        
                except Exception as e:
                    logger.warning(f"Error downloading file {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error in download process: {e}")
        
        return files_content
    
    def _analyze_commit_files(self, commit_info: Dict, changed_files: List[str], 
                            files_content: Dict[str, str], workspace_dir: str) -> Dict:
        """Analizza i file del commit e calcola le metriche."""
        try:
            # Calcola statistiche diff (approssimate, basate sui file correnti)
            loc_added, loc_deleted = self._calculate_loc_stats(files_content)
            
            # Calcola complessità ciclomatica
            commit_complexity, files_complexity = self._calculate_files_complexity(files_content)
            
            # Calcola metriche di qualità del codice con smell tracking
            total_smells, total_warnings, smells_by_file, smells_by_type = self._analyze_code_quality(files_content)
            
            # Calcola densità smell
            total_loc = sum(CodeAnalysisUtils.count_lines_of_code(path) for path in files_content.values())
            smell_density = total_smells / total_loc if total_loc > 0 else 0
            
            # Classifica il commit
            is_pr = CodeAnalysisUtils.is_from_pull_request(commit_info['message'])
            is_bug_fix = CodeAnalysisUtils.is_bug_fix(commit_info['message'])
            commit_goals = CodeAnalysisUtils._classify_commit_message(commit_info['message'])
            
            # Crea l'oggetto risultato
            result = {
                'commit_hash': commit_info['commit_hash'],
                'author': commit_info['author'],
                'date': commit_info['date'].isoformat(),
                'message': commit_info['message'],
                'LOC_added': loc_added,
                'LOC_deleted': loc_deleted,
                'files_changed': len(changed_files),
                'commit_cyclomatic_complexity': commit_complexity,
                'changed_files': changed_files,
                'changed_files_complexity': files_complexity,
                'total_smells_found': total_smells,
                'smell_density': smell_density,
                'num_warnings': total_warnings,
                'is_pr': is_pr,
                'is_bug_fix': is_bug_fix,
                'repo_name': self.repo_info.name,
                'new_feature': commit_goals.get('new_feature', 0),
                'bug_fixing': commit_goals.get('bug_fixing', 0),
                'enhancement': commit_goals.get('enhancement', 0),
                'refactoring': commit_goals.get('refactoring', 0),
                'smells_by_file': smells_by_file,
                'smells_by_type': smells_by_type,
                'analysis_timestamp': datetime.datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing commit files: {e}")
            return None
    
    def _calculate_loc_stats(self, files_content: Dict[str, str]) -> Tuple[int, int]:
        """Calcola statistiche LOC approssimate."""
        total_loc = 0
        for file_path in files_content.values():
            total_loc += CodeAnalysisUtils.count_lines_of_code(file_path)
        
        # Per ora restituiamo una stima approssimativa
        # In un'implementazione più sofisticata, potresti confrontare con il parent commit
        return total_loc, 0
    
    def _calculate_files_complexity(self, files_content: Dict[str, str]) -> Tuple[int, Dict]:
        """Calcola la complessità ciclomatica dei file."""
        total_complexity = 0
        files_complexity = {}
        
        for rel_path, full_path in files_content.items():
            try:
                complexity_data = CodeAnalysisUtils.calculate_cyclomatic_complexity(full_path)
                file_complexity = complexity_data.get('complexity_total', 0)
                total_complexity += file_complexity
                files_complexity[rel_path] = complexity_data
            except Exception as e:
                logger.warning(f"Error calculating complexity for {rel_path}: {e}")
                files_complexity[rel_path] = {'complexity_media': 0, 'complexity_total': 0}
        
        return total_complexity, files_complexity
    
    def _analyze_code_quality(self, files_content: Dict[str, str]) -> Tuple[int, int, Dict, Dict]:
        """Analizza la qualità del codice per i file."""
        total_smells = 0
        total_warnings = 0
        smells_by_file = {}
        smells_by_type = {}
        
        for rel_path, full_path in files_content.items():
            try:
                # Analisi smell
                smell_result = CodeAnalysisUtils.detect_code_smells(full_path)
                if smell_result and isinstance(smell_result, dict):
                    file_smells = smell_result.get('total_smells', 0)
                    total_smells += file_smells
                    
                    if file_smells > 0:
                        smells_by_file[rel_path] = smell_result
                        
                        # Conta smell per tipo
                        for smell_type, count in smell_result.get('smells_by_type', {}).items():
                            smells_by_type[smell_type] = smells_by_type.get(smell_type, 0) + count
                
                # Analisi warning
                try:
                    warnings = CodeAnalysisUtils.detect_warnings(full_path)
                    total_warnings += warnings
                except Exception as e:
                    logger.warning(f"Error detecting warnings for {rel_path}: {e}")
                    
            except Exception as e:
                logger.warning(f"Error analyzing quality for {rel_path}: {e}")
        
        return total_smells, total_warnings, smells_by_file, smells_by_type
    
    def _cleanup_workspace(self, workspace_dir: str):
        """Pulisce il workspace del commit."""
        if os.path.exists(workspace_dir):
            try:
                subprocess.run(['chmod', '-R', '+w', workspace_dir], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                shutil.rmtree(workspace_dir)
            except Exception as e:
                logger.warning(f"Error cleaning workspace {workspace_dir}: {e}")
    
    def _save_commit_result(self, commit_hash: str, result: Dict):
        """Salva il risultato di un commit in un file JSON separato."""
        try:
            result_file = os.path.join(self.parallel_results_dir, f"{commit_hash}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving commit result {commit_hash}: {e}")
    
    def _save_parallel_state(self):
        """Salva lo stato dell'analisi parallela."""
        try:
            state = {
                'analyzed_commits': list(self.analyzed_commits),
                'skipped_commits': list(self.skipped_commits),
                'processing_commits': list(self.processing_commits),
                'timestamp': datetime.datetime.now().isoformat(),
                'max_workers': self.max_workers
            }
            
            with open(self.parallel_state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving parallel state: {e}")
    
    def _load_previous_state(self):
        """Carica lo stato precedente dell'analisi."""
        try:
            if os.path.exists(self.parallel_state_file):
                with open(self.parallel_state_file, 'r') as f:
                    state = json.load(f)
                
                self.analyzed_commits = set(state.get('analyzed_commits', []))
                self.skipped_commits = set(state.get('skipped_commits', []))
                
                logger.info(f"Loaded previous state: {len(self.analyzed_commits)} analyzed, "
                           f"{len(self.skipped_commits)} skipped")
                
                # Carica i risultati dei commit già analizzati
                self._load_commit_results()
                
        except Exception as e:
            logger.error(f"Error loading previous state: {e}")
    
    def _load_commit_results(self):
        """Carica i risultati dei commit già analizzati."""
        try:
            for commit_hash in self.analyzed_commits:
                result_file = os.path.join(self.parallel_results_dir, f"{commit_hash}.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                        self.commit_results[commit_hash] = result
                        
        except Exception as e:
            logger.error(f"Error loading commit results: {e}")
    
    def _get_ordered_commits(self) -> List[Commit]:
        """Get all commits ordered by date (oldest first)."""
        try:
            # Usa subprocess per ottenere gli hash dei commit ordinati
            result = subprocess.run(
                ['git', 'log', '--pretty=format:%H', '--reverse'],
                cwd=self.repo_info.local_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                raise Exception(f"Git log failed: {result.stderr}")
            
            commit_hashes = [h.strip() for h in result.stdout.strip().split('\n') if h.strip()]
            
            # Converte in oggetti Commit
            commits = []
            for commit_hash in commit_hashes:
                try:
                    commit = self.repo.commit(commit_hash)
                    commits.append(commit)
                except Exception as e:
                    logger.warning(f"Could not get commit object for {commit_hash}: {e}")
            
            return commits
            
        except Exception as e:
            logger.error(f"Error getting ordered commits: {e}")
            raise
    
    def _find_resume_point(self, all_commits: List[Commit]) -> int:
        """Find the index to resume analysis from."""
        if not self.resume or not self.analyzed_commits:
            return 0
        
        # Trova l'ultimo commit analizzato
        for i, commit in enumerate(all_commits):
            if commit.hexsha in self.analyzed_commits:
                continue
            return i
        
        return len(all_commits)
    
    def _finalize_analysis(self):
        """Finalizza l'analisi combinando tutti i risultati."""
        logger.info("Finalizing analysis and merging results...")
        
        try:
            # Carica tutti i risultati
            all_results = []
            commit_order = []
            
            # Leggi tutti i file di risultato
            for result_file in os.listdir(self.parallel_results_dir):
                if result_file.endswith('.json') and result_file != 'parallel_state.json':
                    try:
                        with open(os.path.join(self.parallel_results_dir, result_file), 'r') as f:
                            result = json.load(f)
                            all_results.append(result)
                            commit_order.append((result['date'], result))
                    except Exception as e:
                        logger.error(f"Error loading result file {result_file}: {e}")
            
            # Ordina per data
            commit_order.sort(key=lambda x: x[0])
            ordered_results = [result for _, result in commit_order]
            
            # Calcola metriche aggregate con smell tracking
            self._calculate_aggregate_metrics(ordered_results)
            
            # Processa smell evolution
            smell_tracker = self._process_smell_evolution(ordered_results)
            
            # Converte in DataFrame e salva
            self._save_final_results(ordered_results, smell_tracker)
            
            # Genera report
            self._generate_reports(ordered_results)
            
            logger.info(f"Analysis finalized. Processed {len(ordered_results)} commits.")
            
        except Exception as e:
            logger.error(f"Error finalizing analysis: {e}")
            raise
    
    def _calculate_aggregate_metrics(self, ordered_results: List[Dict]):
        """Calcola le metriche aggregate come author experience e tempo tra commit."""
        author_stats = {}
        last_commit_by_author = {}
        
        for i, result in enumerate(ordered_results):
            author = result['author']
            commit_date = datetime.datetime.fromisoformat(result['date'])
            
            # Author experience
            author_experience = author_stats.get(author, 0)
            result['author_experience'] = author_experience
            author_stats[author] = author_experience + 1
            
            # Time since last commit
            time_since_last = 0.0
            if author in last_commit_by_author:
                last_date = last_commit_by_author[author]
                time_delta = commit_date - last_date
                time_since_last = time_delta.total_seconds() / 3600  # In ore
            
            result['time_since_last_commit'] = time_since_last
            last_commit_by_author[author] = commit_date
            
            # Calcola project complexity (approssimativa)
            # In una implementazione più completa, dovresti ricostruire lo stato del progetto
            result['project_cyclomatic_complexity'] = result['commit_cyclomatic_complexity']
    
    def _process_smell_evolution(self, ordered_results: List[Dict]) -> SmellEvolutionTracker:
        """Processa l'evoluzione degli smell attraverso i commit."""
        smell_tracker = SmellEvolutionTracker()
        
        try:
            for result in ordered_results:
                commit_hash = result['commit_hash']
                commit_date = datetime.datetime.fromisoformat(result['date'])
                changed_files = result.get('changed_files', [])
                smells_by_file = result.get('smells_by_file', {})
                
                # Processa gli smell per questo commit
                smell_tracker.process_commit_smells(commit_hash, commit_date, changed_files, smells_by_file)
                
                # Calcola statistiche smell per questo commit
                smell_events = []
                smells_introduced = 0
                smells_removed = 0
                smells_modified = 0
                smells_persisted = 0
                
                # Estrai eventi per questo commit
                for filename in changed_files:
                    if filename in smell_tracker.file_trackers:
                        file_tracker = smell_tracker.file_trackers[filename]
                        
                        # Cerca eventi per questo commit
                        for smell_history in list(file_tracker.active_smells.values()) + list(file_tracker.inactive_smells.values()):
                            for event in smell_history.events:
                                if event.commit_hash == commit_hash:
                                    event_data = {
                                        'filename': filename,
                                        'smell_id': smell_history.smell_id,
                                        'smell_name': smell_history.smell_name,
                                        'function_name': smell_history.function_name,
                                        'event_type': event.event_type.value,
                                        'line': event.current_line,
                                        'previous_line': event.previous_line,
                                        'notes': event.notes
                                    }
                                    smell_events.append(event_data)
                                    
                                    # Conta per tipo di evento
                                    if event.event_type.value == 'introduced':
                                        smells_introduced += 1
                                    elif event.event_type.value == 'removed':
                                        smells_removed += 1
                                    elif event.event_type.value == 'modified':
                                        smells_modified += 1
                                    elif event.event_type.value == 'persisted':
                                        smells_persisted += 1
                
                # Aggiorna il risultato con le informazioni sull'evoluzione degli smell
                result['smell_events'] = smell_events
                result['smells_introduced'] = smells_introduced
                result['smells_removed'] = smells_removed
                result['smells_modified'] = smells_modified
                result['smells_persisted'] = smells_persisted
                
                # Calcola file con smell
                files_with_smells = list(smells_by_file.keys())
                files_smell_introduced = [event['filename'] for event in smell_events 
                                        if event['event_type'] == 'introduced']
                files_smell_removed = [event['filename'] for event in smell_events 
                                     if event['event_type'] == 'removed']
                
                result['files_with_smells'] = files_with_smells
                result['files_smell_introduced'] = files_smell_introduced
                result['files_smell_removed'] = files_smell_removed
        
        except Exception as e:
            logger.error(f"Error processing smell evolution: {e}")
        
        return smell_tracker
    
    def _save_final_results(self, ordered_results: List[Dict], smell_tracker: SmellEvolutionTracker):
        """Salva i risultati finali in formato CSV e JSON."""
        try:
            # Prepara i dati per il DataFrame
            df_data = []
            for result in ordered_results:
                # Converte le liste/dict complessi in JSON per il CSV
                csv_row = result.copy()
                csv_row['changed_files_complexity'] = json.dumps(csv_row.get('changed_files_complexity', {}))
                csv_row['smells_by_file'] = json.dumps(csv_row.get('smells_by_file', {}))
                csv_row['smells_by_type'] = json.dumps(csv_row.get('smells_by_type', {}))
                csv_row['smell_events'] = json.dumps(csv_row.get('smell_events', []))
                csv_row['files_with_smells'] = json.dumps(csv_row.get('files_with_smells', []))
                csv_row['files_smell_introduced'] = json.dumps(csv_row.get('files_smell_introduced', []))
                csv_row['files_smell_removed'] = json.dumps(csv_row.get('files_smell_removed', []))
                
                # Rimuovi la lista dei file modificati per ridurre la dimensione del CSV
                csv_row.pop('changed_files', None)
                
                df_data.append(csv_row)
            
            # Crea DataFrame e salva CSV
            df = pd.DataFrame(df_data)
            df.to_csv(self.output_file, index=False)
            logger.info(f"Results saved to {self.output_file}")
            
            # Salva anche in formato JSON con dati completi
            json_file = os.path.splitext(self.output_file)[0] + '.json'
            with open(json_file, 'w') as f:
                json.dump(ordered_results, f, indent=2)
            logger.info(f"Results also saved to {json_file}")
            
            # Salva dati evoluzione smell
            smell_evolution_data = smell_tracker.get_smell_evolution_report()
            with open(self.smell_evolution_file, 'w') as f:
                json.dump(smell_evolution_data, f, indent=2)
            logger.info(f"Smell evolution data saved to {self.smell_evolution_file}")
            
            # Calcola e salva frequenze file
            self._calculate_and_save_file_frequencies(ordered_results)
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
            raise
    
    def _calculate_and_save_file_frequencies(self, ordered_results: List[Dict]):
        """Calcola e salva le frequenze dei file modificati e con bug."""
        try:
            edited_files_freq = defaultdict(int)
            bugged_files_freq = defaultdict(int)
            edited_files_complexity = {}
            
            for result in ordered_results:
                changed_files = result.get('changed_files', [])
                is_bug_fix = result.get('is_bug_fix', False)
                files_complexity = result.get('changed_files_complexity', {})
                
                # Conta file modificati
                for file_path in changed_files:
                    if CodeAnalysisUtils.is_code_file(file_path):
                        edited_files_freq[file_path] += 1
                        
                        # Aggiorna complessità
                        if isinstance(files_complexity, str):
                            try:
                                files_complexity = json.loads(files_complexity)
                            except:
                                files_complexity = {}
                        
                        if file_path in files_complexity:
                            edited_files_complexity[file_path] = files_complexity[file_path]
                
                # Conta file con bug se è un bug fix
                if is_bug_fix:
                    for file_path in changed_files:
                        if CodeAnalysisUtils.is_code_file(file_path):
                            bugged_files_freq[file_path] += 1
            
            # Salva frequenze
            freq_data = {
                'bugged_files': dict(bugged_files_freq),
                'edited_files': dict(edited_files_freq),
                'edited_files_complexity': edited_files_complexity
            }
            
            freq_file = os.path.join(self.repo_info.output_dir, "file_frequencies.json")
            with open(freq_file, 'w') as f:
                json.dump(freq_data, f, indent=2)
            logger.info(f"File frequencies saved to {freq_file}")
            
        except Exception as e:
            logger.error(f"Error calculating file frequencies: {e}")
    
    def _generate_reports(self, ordered_results: List[Dict]):
        """Genera i report finali."""
        try:
            # Crea DataFrame per i report
            df = pd.DataFrame(ordered_results)
            
            # Rimuovi colonne complesse per i report
            report_df = df.copy()
            columns_to_remove = ['changed_files_complexity', 'smells_by_file', 'smells_by_type', 
                               'smell_events', 'files_with_smells', 'files_smell_introduced', 
                               'files_smell_removed', 'analysis_timestamp']
            for col in columns_to_remove:
                if col in report_df.columns:
                    report_df = report_df.drop(columns=[col])
            
            # Genera report principale
            create_single_repo_report(self.repo_info.output_dir, report_df, self.repo_info.name)
            
            # Genera report evoluzione smell se disponibile
            if os.path.exists(self.smell_evolution_file):
                from ..utils.report_generator import (
                    create_smell_evolution_report, 
                    create_smell_csv_export,
                    create_comprehensive_analysis_report
                )
                
                create_smell_evolution_report(self.repo_info.output_dir, self.smell_evolution_file, self.repo_info.name)
                create_smell_csv_export(self.repo_info.output_dir, self.smell_evolution_file)
                create_comprehensive_analysis_report(self.repo_info.output_dir, self.repo_info.name)
            
            logger.info("All reports generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
    
    def get_analysis_summary(self) -> Dict:
        """Restituisce un riassunto dell'analisi."""
        return {
            'total_commits_analyzed': len(self.analyzed_commits),
            'total_commits_skipped': len(self.skipped_commits),
            'max_workers_used': self.max_workers,
            'output_directory': self.repo_info.output_dir,
            'results_available': len(self.commit_results)
        }

    # Metodi per compatibilità con l'interfaccia sequenziale
    def save_results(self):
        """Compatibilità con l'interfaccia sequenziale - i risultati sono già salvati."""
        logger.info("Results already saved during parallel processing")
    
    def save_checkpoint(self, last_commit_hash: str):
        """Compatibilità con l'interfaccia sequenziale."""
        self._save_parallel_state()
    
    def load_checkpoint(self):
        """Compatibilità con l'interfaccia sequenziale."""
        self._load_previous_state()
        return None  # Non restituisce un commit hash specifico
    
    @staticmethod
    def can_resume_from_sequential(repo_info: RepositoryInfo) -> bool:
        """
        Verifica se è possibile riprendere un'analisi sequenziale in modalità parallela.
        
        Args:
            repo_info: Informazioni del repository
            
        Returns:
            True se è possibile il resume, False altrimenti
        """
        # Controlla se esistono file di checkpoint sequenziali
        sequential_checkpoint = os.path.join(repo_info.output_dir or repo_info.name, "checkpoint.json")
        sequential_csv = os.path.join(repo_info.output_dir or repo_info.name, "commit_metrics.csv")
        
        return os.path.exists(sequential_checkpoint) or os.path.exists(sequential_csv)
    
    def resume_from_sequential(self):
        """
        Riprende un'analisi iniziata in modalità sequenziale.
        
        Questo metodo converte i risultati sequenziali in formato parallelo
        per permettere la continuazione dell'analisi.
        """
        try:
            logger.info("Attempting to resume from sequential analysis...")
            
            # Carica dati dal CSV sequenziale se disponibile
            sequential_csv = os.path.join(self.repo_info.output_dir, "commit_metrics.csv")
            if os.path.exists(sequential_csv):
                df = pd.read_csv(sequential_csv)
                
                for _, row in df.iterrows():
                    commit_hash = row.get('commit_hash', '')
                    if commit_hash:
                        # Converte la riga in formato parallelo
                        result = self._convert_sequential_row_to_parallel(row)
                        
                        # Salva come risultato parallelo
                        self._save_commit_result(commit_hash, result)
                        self.analyzed_commits.add(commit_hash)
                        self.commit_results[commit_hash] = result
                
                logger.info(f"Converted {len(df)} sequential results to parallel format")
            
            # Carica checkpoint sequenziale se disponibile
            sequential_checkpoint = os.path.join(self.repo_info.output_dir, "checkpoint.json")
            if os.path.exists(sequential_checkpoint):
                with open(sequential_checkpoint, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Estrai informazioni utili dal checkpoint sequenziale
                skipped_commits = checkpoint_data.get('skipped_commits', [])
                self.skipped_commits.update(skipped_commits)
                
                logger.info(f"Loaded {len(skipped_commits)} skipped commits from sequential checkpoint")
            
            # Salva lo stato parallelo
            self._save_parallel_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Error resuming from sequential analysis: {e}")
            return False
    
    def _convert_sequential_row_to_parallel(self, row: pd.Series) -> Dict:
        """
        Converte una riga del CSV sequenziale nel formato parallelo.
        
        Args:
            row: Riga pandas del CSV sequenziale
            
        Returns:
            Dizionario nel formato del risultato parallelo
        """
        try:
            # Parsing dei campi JSON
            def safe_json_loads(value, default=None):
                if pd.isna(value) or value == '':
                    return default or {}
                try:
                    if isinstance(value, str):
                        return json.loads(value)
                    return value
                except:
                    return default or {}
            
            result = {
                'commit_hash': row.get('commit_hash', ''),
                'author': row.get('author', ''),
                'date': row.get('date', ''),
                'message': row.get('message', ''),
                'LOC_added': int(row.get('LOC_added', 0)),
                'LOC_deleted': int(row.get('LOC_deleted', 0)),
                'files_changed': int(row.get('files_changed', 0)),
                'commit_cyclomatic_complexity': float(row.get('commit_cyclomatic_complexity', 0)),
                'project_cyclomatic_complexity': float(row.get('project_cyclomatic_complexity', 0)),
                'author_experience': int(row.get('author_experience', 0)),
                'time_since_last_commit': float(row.get('time_since_last_commit', 0)),
                'total_smells_found': int(row.get('total_smells_found', 0)),
                'smell_density': float(row.get('smell_density', 0)),
                'num_warnings': int(row.get('num_warnings', 0)),
                'is_pr': bool(row.get('is_pr', False)),
                'is_bug_fix': bool(row.get('is_bug_fix', False)),
                'repo_name': row.get('repo_name', self.repo_info.name),
                'new_feature': int(row.get('new_feature', 0)),
                'bug_fixing': int(row.get('bug_fixing', 0)),
                'enhancement': int(row.get('enhancement', 0)),
                'refactoring': int(row.get('refactoring', 0)),
                'changed_files_complexity': safe_json_loads(row.get('changed_files_complexity'), {}),
                'smells_by_file': safe_json_loads(row.get('smells_by_file'), {}),
                'smells_by_type': safe_json_loads(row.get('smells_by_type'), {}),
                'smell_events': safe_json_loads(row.get('smell_events'), []),
                'smells_introduced': int(row.get('smells_introduced', 0)),
                'smells_removed': int(row.get('smells_removed', 0)),
                'smells_modified': int(row.get('smells_modified', 0)),
                'smells_persisted': int(row.get('smells_persisted', 0)),
                'files_with_smells': safe_json_loads(row.get('files_with_smells'), []),
                'files_smell_introduced': safe_json_loads(row.get('files_smell_introduced'), []),
                'files_smell_removed': safe_json_loads(row.get('files_smell_removed'), []),
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'converted_from_sequential': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error converting sequential row: {e}")
            return {}


# Classe factory per creare l'analyzer appropriato
class RepoAnalyzerFactory:
    """Factory per creare l'analyzer appropriato basandosi sui parametri."""
    
    @staticmethod
    def create_analyzer(repo_info: RepositoryInfo, resume: bool = False, 
                       parallel: bool = True, max_workers: Optional[int] = None) -> Any:
        """
        Crea l'analyzer appropriato.
        
        Args:
            repo_info: Informazioni del repository
            resume: Flag per riprendere l'analisi
            parallel: True per modalità parallela, False per sequenziale
            max_workers: Numero massimo di worker (solo per modalità parallela)
            
        Returns:
            Istanza dell'analyzer appropriato
        """
        if parallel:
            analyzer = ParallelRepoAnalyzer(repo_info, resume, max_workers)
            
            # Se resume è richiesto e ci sono dati sequenziali, convertili
            if resume and ParallelRepoAnalyzer.can_resume_from_sequential(repo_info):
                if analyzer.resume_from_sequential():
                    logger.info("Successfully resumed from sequential analysis in parallel mode")
                else:
                    logger.warning("Could not resume from sequential analysis, starting fresh")
            
            return analyzer
        else:
            # Importa l'analyzer sequenziale originale
            from .repo_analyzer import RepoAnalyzer
            return RepoAnalyzer(repo_info, resume)
    
    @staticmethod
    def get_recommended_mode(repo_info: RepositoryInfo) -> Tuple[bool, int]:
        """
        Raccomanda la modalità di analisi basandosi sul repository.
        
        Args:
            repo_info: Informazioni del repository
            
        Returns:
            Tupla (parallel_recommended, suggested_workers)
        """
        try:
            # Stima la dimensione del repository
            if os.path.exists(repo_info.local_path):
                repo = Repo(repo_info.local_path)
                
                # Conta approssimativamente i commit
                try:
                    result = subprocess.run(
                        ['git', 'rev-list', '--count', 'HEAD'],
                        cwd=repo_info.local_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    commit_count = int(result.stdout.strip()) if result.returncode == 0 else 0
                except:
                    commit_count = 0
                
                # Raccomandazioni basate sulla dimensione
                if commit_count > 500:
                    # Repository grande - raccomanda parallelo
                    suggested_workers = max(2, min(8, multiprocessing.cpu_count() // 2))
                    return True, suggested_workers
                elif commit_count > 100:
                    # Repository medio - parallelo con pochi worker
                    suggested_workers = max(2, min(4, multiprocessing.cpu_count() // 4))
                    return True, suggested_workers
                else:
                    # Repository piccolo - sequenziale è sufficiente
                    return False, 1
            
            # Default per repository sconosciuti
            return True, max(2, min(4, multiprocessing.cpu_count() // 4))
            
        except Exception as e:
            logger.warning(f"Could not analyze repository for recommendations: {e}")
            return True, 4  # Default sicuro