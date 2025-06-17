"""Parallel analyzer per Git repositories - Architettura ripensata per parallelizzazione ottimale."""

import os
import subprocess
import tempfile
import datetime
import json
import logging
import shutil
import threading
import time
import multiprocessing
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
from git import Repo, Commit
import concurrent.futures
from collections import defaultdict
from uuid import uuid4

import pandas as pd

from ..models import CommitMetrics, RepositoryInfo
from ..utils import CodeAnalysisUtils, GitRepositoryManager, create_single_repo_report
from .smell_tracker import SmellEvolutionTracker, SmellDetection, SmellEventType

logger = logging.getLogger('repo_analyzer.parallel_repo_analyzer')

class CommitAnalysisTask:
    """Singola unità di lavoro per l'analisi di un commit."""
    
    def __init__(self, commit_hash: str, commit_data: Dict, repo_path: str, workspace_base: str):
        self.commit_hash = commit_hash
        self.commit_data = commit_data  # Pre-estratto per evitare GitPython threading issues
        self.repo_path = repo_path
        self.workspace_base = workspace_base
        self.workspace_id = f"{commit_hash[:12]}_{os.getpid()}_{threading.current_thread().ident}"
    
    def execute(self) -> Optional[Dict]:
        """Esegue l'analisi del commit in modo completamente isolato."""
        workspace_dir = None
        try:
            # Crea workspace isolato
            workspace_dir = os.path.join(self.workspace_base, self.workspace_id)
            os.makedirs(workspace_dir, exist_ok=True)
            
            # Estrai file modificati usando git direttamente
            changed_files = self._extract_changed_files()
            if not changed_files:
                return None
            
            # Scarica file per questo commit
            files_content = self._download_commit_files(changed_files, workspace_dir)
            if not files_content:
                return None
            
            # Analizza i file
            return self._analyze_files(changed_files, files_content)
            
        except Exception as e:
            logger.error(f"Error analyzing commit {self.commit_hash[:8]}: {e}")
            return None
        finally:
            # Cleanup garantito
            if workspace_dir and os.path.exists(workspace_dir):
                try:
                    shutil.rmtree(workspace_dir)
                except:
                    pass
    
    def _extract_changed_files(self) -> List[str]:
        """Estrae i file modificati usando git show."""
        try:
            result = subprocess.run(
                ['git', 'show', '--name-only', '--pretty=format:', self.commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                return [f for f in files if CodeAnalysisUtils.is_code_file(f)]
            
            return []
        except Exception as e:
            logger.warning(f"Error extracting changed files for {self.commit_hash[:8]}: {e}")
            return []
    
    def _download_commit_files(self, changed_files: List[str], workspace_dir: str) -> Dict[str, str]:
        """Scarica i file del commit nel workspace."""
        files_content = {}
        
        for file_path in changed_files:
            try:
                result = subprocess.run(
                    ['git', 'show', f"{self.commit_hash}:{file_path}"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    full_file_path = os.path.join(workspace_dir, file_path)
                    os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
                    
                    with open(full_file_path, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                    
                    files_content[file_path] = full_file_path
                    
            except Exception as e:
                logger.warning(f"Error downloading {file_path}: {e}")
        
        return files_content
    
    def _analyze_files(self, changed_files: List[str], files_content: Dict[str, str]) -> Dict:
        """Analizza i file e crea le metriche (salvando anche dati per project metrics)."""
        try:
            # Calcola metriche base per questo commit
            total_loc = sum(CodeAnalysisUtils.count_lines_of_code(path) for path in files_content.values())
            
            # Complessità ciclomatica solo dei file modificati
            commit_complexity = 0
            files_complexity = {}
            for rel_path, full_path in files_content.items():
                try:
                    complexity_data = CodeAnalysisUtils.calculate_cyclomatic_complexity(full_path)
                    commit_complexity += complexity_data.get('complexity_total', 0)
                    files_complexity[rel_path] = complexity_data
                except:
                    files_complexity[rel_path] = {'complexity_media': 0, 'complexity_total': 0}
            
            # *** NUOVO: Calcola e salva dati per project metrics ***
            project_data = self._collect_project_data_during_analysis()
            
            # Analisi qualità del codice solo per file modificati
            total_smells = 0
            total_warnings = 0
            smells_by_file = {}
            smells_by_type = {}
            
            for rel_path, full_path in files_content.items():
                try:
                    # Smell detection
                    smell_result = CodeAnalysisUtils.detect_code_smells(full_path)
                    if smell_result and isinstance(smell_result, dict):
                        file_smells = smell_result.get('total_smells', 0)
                        total_smells += file_smells
                        
                        if file_smells > 0:
                            smells_by_file[rel_path] = smell_result
                            for smell_type, count in smell_result.get('smells_by_type', {}).items():
                                smells_by_type[smell_type] = smells_by_type.get(smell_type, 0) + count
                    
                    # Warning detection
                    try:
                        warnings = CodeAnalysisUtils.detect_warnings(full_path)
                        total_warnings += warnings
                    except:
                        pass
                        
                except Exception as e:
                    logger.warning(f"Error analyzing quality for {rel_path}: {e}")
            
            # Classificazione commit
            is_pr = CodeAnalysisUtils.is_from_pull_request(self.commit_data['message'])
            is_bug_fix = CodeAnalysisUtils.is_bug_fix(self.commit_data['message'])
            commit_goals = CodeAnalysisUtils._classify_commit_message(self.commit_data['message'])
            
            # *** NUOVO: Ottieni LOC accurate subito ***
            accurate_loc_stats = self._get_accurate_loc_stats_during_analysis()
            
            # Risultato con dati project inclusi
            return {
                'commit_hash': self.commit_hash,
                'author': self.commit_data['author'],
                'date': self.commit_data['date'],
                'message': self.commit_data['message'],
                'LOC_added': accurate_loc_stats.get('added', total_loc) if accurate_loc_stats else total_loc,
                'LOC_deleted': accurate_loc_stats.get('deleted', 0) if accurate_loc_stats else 0,
                'files_changed': len(changed_files),
                'commit_cyclomatic_complexity': commit_complexity,
                'changed_files': changed_files,
                'changed_files_complexity': files_complexity,
                'total_smells_found': total_smells,
                'smell_density': total_smells / total_loc if total_loc > 0 else 0,
                'num_warnings': total_warnings,
                'is_pr': is_pr,
                'is_bug_fix': is_bug_fix,
                'repo_name': self.commit_data.get('repo_name', ''),
                'new_feature': commit_goals.get('new_feature', 0),
                'bug_fixing': commit_goals.get('bug_fixing', 0),
                'enhancement': commit_goals.get('enhancement', 0),
                'refactoring': commit_goals.get('refactoring', 0),
                'smells_by_file': smells_by_file,
                'smells_by_type': smells_by_type,
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                
                # *** NUOVO: Project data calcolati durante l'analisi ***
                'project_total_python_files': project_data.get('total_python_files', 0),
                'project_estimated_complexity': project_data.get('estimated_complexity', 0),
                'project_total_loc': project_data.get('total_loc', 0),
                
                # Placeholders per metriche che saranno calcolate nel merge
                'author_experience': 0,  # Calcolato in _calculate_aggregate_metrics
                'time_since_last_commit': 0.0,  # Calcolato in _calculate_aggregate_metrics
                'project_cyclomatic_complexity': 0  # Calcolato da project_data
            }
            
        except Exception as e:
            logger.error(f"Error in file analysis: {e}")
            return None
    
    def _collect_project_data_during_analysis(self) -> Dict:
        """Raccoglie dati del progetto durante l'analisi del commit (veloce)."""
        try:
            # Usa git ls-tree per ottenere tutti i file del progetto a questo commit
            result = subprocess.run(
                ['git', 'ls-tree', '-r', '--name-only', self.commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                all_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                python_files = [f for f in all_files if CodeAnalysisUtils.is_code_file(f)]
                
                # Stima complessità del progetto basata su euristica
                # Assumiamo ~15 unità di complessità per file Python medio
                estimated_complexity = len(python_files) * 15
                
                # Stima LOC totali del progetto
                # Assumiamo ~150 LOC per file Python medio
                estimated_total_loc = len(python_files) * 150
                
                return {
                    'total_python_files': len(python_files),
                    'estimated_complexity': estimated_complexity,
                    'total_loc': estimated_total_loc,
                    'all_python_files': python_files
                }
            
            return {'total_python_files': 0, 'estimated_complexity': 0, 'total_loc': 0}
            
        except Exception as e:
            logger.debug(f"Error collecting project data for {self.commit_hash[:8]}: {e}")
            return {'total_python_files': 0, 'estimated_complexity': 0, 'total_loc': 0}
    
    def _get_accurate_loc_stats_during_analysis(self) -> Optional[Dict]:
        """Ottiene statistiche LOC accurate durante l'analisi."""
        try:
            result = subprocess.run(
                ['git', 'show', '--stat', '--pretty=format:', self.commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5  # Timeout molto ridotto
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                
                # Cerca la linea di summary
                for line in reversed(lines):
                    if 'changed' in line and ('insertion' in line or 'deletion' in line):
                        added = 0
                        deleted = 0
                        
                        # Estrai insertions
                        if 'insertion' in line:
                            import re
                            match = re.search(r'(\d+) insertion', line)
                            if match:
                                added = int(match.group(1))
                        
                        # Estrai deletions  
                        if 'deletion' in line:
                            import re
                            match = re.search(r'(\d+) deletion', line)
                            if match:
                                deleted = int(match.group(1))
                        
                        return {'added': added, 'deleted': deleted}
                        
            return None
            
        except Exception as e:
            logger.debug(f"Error getting LOC stats for {self.commit_hash[:8]}: {e}")
            return None

class ParallelCommitProcessor:
    """Processore parallelo per commit con gestione thread-safe dei risultati."""
    
    def __init__(self, output_dir: str, max_workers: int):
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.results_dir = os.path.join(output_dir, "parallel_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self._lock = threading.Lock()
        self._processed_commits = set()
        self._progress_callback = None
        
        # Carica commit già processati
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Carica i risultati esistenti."""
        if os.path.exists(self.results_dir):
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    commit_hash = filename[:-5]
                    self._processed_commits.add(commit_hash)
    
    def set_progress_callback(self, callback):
        """Imposta callback per progress updates."""
        self._progress_callback = callback
    
    def process_commits(self, commit_tasks: List[CommitAnalysisTask]) -> int:
        """Processa i commit in parallelo."""
        # Filtra commit già processati
        pending_tasks = [task for task in commit_tasks 
                        if task.commit_hash not in self._processed_commits]
        
        if not pending_tasks:
            return 0
        
        processed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Sottometti tutti i task
            future_to_task = {
                executor.submit(task.execute): task 
                for task in pending_tasks
            }
            
            # Processa risultati
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result and self._save_result(task.commit_hash, result):
                        processed_count += 1
                        
                        if self._progress_callback:
                            self._progress_callback('analyzed', task.commit_hash, processed_count)
                    else:
                        if self._progress_callback:
                            self._progress_callback('skipped', task.commit_hash, processed_count)
                            
                except Exception as e:
                    logger.error(f"Error processing commit {task.commit_hash[:8]}: {e}")
                    if self._progress_callback:
                        self._progress_callback('error', task.commit_hash, processed_count)
        
        return processed_count
    
    def _save_result(self, commit_hash: str, result: Dict) -> bool:
        """Salva il risultato in modo thread-safe."""
        with self._lock:
            if commit_hash in self._processed_commits:
                return False
            
            try:
                result_file = os.path.join(self.results_dir, f"{commit_hash}.json")
                temp_file = f"{result_file}.tmp.{uuid4().hex[:8]}"
                
                with open(temp_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                os.rename(temp_file, result_file)
                
                # Verifica che il file sia stato creato correttamente
                if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                    self._processed_commits.add(commit_hash)
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Error saving result for {commit_hash}: {e}")
                return False
    
    def get_processed_count(self) -> int:
        """Restituisce il numero di commit processati."""
        with self._lock:
            return len(self._processed_commits)
    
    def get_all_results(self) -> List[Dict]:
        """Carica tutti i risultati salvati."""
        results = []
        
        for commit_hash in self._processed_commits:
            result_file = os.path.join(self.results_dir, f"{commit_hash}.json")
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                logger.error(f"Error loading result {result_file}: {e}")
        
        # Ordina per data
        results.sort(key=lambda x: x.get('date', ''))
        return results

class ParallelRepoAnalyzer:
    """Analyzer principale ripensato per parallelizzazione ottimale dei commit."""
    
    def __init__(self, repo_info: RepositoryInfo, resume: bool = False, max_workers: Optional[int] = None, 
                 smell_analysis_mode: str = "comprehensive"):
        """
        Initialize the parallel repository analyzer.
        
        Args:
            repo_info: Repository information
            resume: Flag to resume interrupted analysis  
            max_workers: Maximum number of parallel workers
            smell_analysis_mode: Strategy for commit filtering
                - "comprehensive": Include all commits relevant to smell evolution
                - "minimal": Only commits with direct Python changes
                - "all": All commits (no filtering except duplicates)
        """
        self.repo_info = repo_info
        self.resume = resume
        self.max_workers = max_workers or min(16, multiprocessing.cpu_count())
        self.smell_analysis_mode = smell_analysis_mode
        self._progress = None
        
        # Setup directories
        self._setup_directories()
        
        # Initialize repository
        self._initialize_repository()
        
        # Initialize parallel processor
        self.processor = ParallelCommitProcessor(self.repo_info.output_dir, self.max_workers)
        self.processor.set_progress_callback(self._handle_progress_update)
    
    def _setup_directories(self):
        """Setup delle directory necessarie."""
        if not self.repo_info.output_dir:
            self.repo_info.output_dir = self.repo_info.name
        
        os.makedirs(self.repo_info.output_dir, exist_ok=True)
        
        if not self.repo_info.temp_dir:
            self.repo_info.temp_dir = tempfile.mkdtemp(prefix=f"parallel_analyzer_{self.repo_info.name}_")
        
        self.workspace_base = os.path.join(self.repo_info.temp_dir, "workspaces")
        os.makedirs(self.workspace_base, exist_ok=True)
        
        if not self.repo_info.local_path:
            self.repo_info.local_path = os.path.join(self.repo_info.temp_dir, "repo")
    
    def _initialize_repository(self):
        """Inizializza il repository Git."""
        try:
            if not os.path.exists(self.repo_info.local_path):
                self.repo = GitRepositoryManager.clone_repository(
                    self.repo_info.url, 
                    self.repo_info.local_path
                )
            else:
                self.repo = Repo(self.repo_info.local_path)
                try:
                    self.repo.git.fetch('--all')
                except:
                    pass
        except Exception as e:
            logger.error(f"Error initializing repository: {e}")
            raise
    
    def _handle_progress_update(self, status: str, commit_hash: str, count: int):
        """Gestisce gli update di progress."""
        if self._progress:
            if status == 'analyzed':
                self._progress.increment_analyzed(commit_hash)
                self._progress.log(f"Analyzed commit {commit_hash[:8]} ({count} completed)")
            elif status == 'skipped':
                self._progress.increment_skipped()
                self._progress.log(f"Skipped commit {commit_hash[:8]}")
            elif status == 'error':
                self._progress.increment_skipped()
                self._progress.log(f"Error in commit {commit_hash[:8]}")
    
    def analyze_repository(self, max_commits: Optional[int] = None):
        """Analizza il repository con parallelizzazione ottimizzata."""
        logger.info("Starting parallel repository analysis...")
        
        if self._progress:
            self._progress.log("Extracting commits...")
        
        # Estrai dati dei commit
        commit_data_list = self._extract_all_commits(max_commits)
        
        if self._progress:
            self._progress.set_total_commits(len(commit_data_list))
            self._progress.log(f"Found {len(commit_data_list)} commits to analyze")
        
        # Crea task di analisi
        commit_tasks = [
            CommitAnalysisTask(
                commit_hash=data['commit_hash'],
                commit_data=data,
                repo_path=self.repo_info.local_path,
                workspace_base=self.workspace_base
            )
            for data in commit_data_list
        ]
        
        # Processa in parallelo
        if self._progress:
            self._progress.log(f"Starting parallel processing with {self.max_workers} workers...")
        
        processed_count = self.processor.process_commits(commit_tasks)
        
        logger.info(f"Processed {processed_count} commits")
        
        # Finalizza analisi
        self._finalize_analysis()
    

    def _get_git_args_for_smell_mode(self) -> List[str]:
        """Determina gli argomenti git log basati sulla modalità smell analysis."""
        if self.smell_analysis_mode == "all":
            # Modalità completa: tutti i commit per ricerca dettagliata
            return ['--reverse']
            
        elif self.smell_analysis_mode == "minimal":
            # Modalità minimale: solo commit diretti a file Python
            return [
                '--reverse',
                '--no-merges',
                '--first-parent',
                '--',  # Separator
                '*.py'  # Solo file Python
            ]
            
        else:  # comprehensive (default)
            # Modalità comprensiva: bilanciamento tra completezza e performance
            return [
                '--reverse',
                '--no-merges',  # Esclude merge commits standard
                '--first-parent'  # Solo primo parent per evitare duplicati branch
            ]

    def _extract_all_commits(self, max_commits: Optional[int] = None) -> List[Dict]:
        """Estrae tutti i commit reali escludendo merge, duplicati e parent commits."""
        try:
            # Configura strategia git log basata sulla modalità smell analysis
            git_args = self._get_git_args_for_smell_mode()
            
            # Esegui git log con strategia ottimizzata
            result = subprocess.run(
                [
                    'git', 'log',
                    '--pretty=format:%H|%an|%ad|%s|%P',  # Hash|Author|Date|Subject|Parents
                    '--date=iso',
                    *git_args
                ],
                cwd=self.repo_info.local_path,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise Exception(f"Git log failed: {result.stderr}")
            
            # Parse dei commit
            commit_lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            
            # Estrai dati strutturati
            raw_commits = []
            for line in commit_lines:
                try:
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        commit_hash, author, date_str, message = parts
                        
                        # Verifica che sia un hash valido
                        if self._is_valid_commit_hash(commit_hash):
                            raw_commits.append({
                                'hash': commit_hash,
                                'author': author,
                                'date_str': date_str,
                                'message': message
                            })
                except Exception as e:
                    logger.warning(f"Error parsing commit line '{line}': {e}")
            
            # Rimuovi duplicati mantenendo l'ordine
            unique_commits = {}

            for commit in raw_commits:
                commit_hash = commit['hash']
                if commit_hash not in unique_commits:
                    unique_commits[commit_hash] = commit
            
            unique_commit_list = list(unique_commits.values())
            
            # Filtra in base alla modalità di analisi smell
            if self.smell_analysis_mode == "all":
                # Nessun filtro eccetto duplicati
                filtered_commits = unique_commit_list
                logger.info("Using 'all' mode - no commit filtering except duplicates")
            elif self.smell_analysis_mode == "minimal":
                # Solo commit con modifiche dirette a Python
                filtered_commits = [c for c in unique_commit_list 
                                  if self._commit_has_python_changes(c['hash']) and self._is_real_commit(c['hash'])]
                logger.info("Using 'minimal' mode - only direct Python changes")
            else:  # comprehensive (default)
                # Filtra ma mantiene commit rilevanti per smell evolution
                filtered_commits = self._filter_parent_commits(unique_commit_list)
                logger.info("Using 'comprehensive' mode - smell-aware filtering")
            
            if max_commits:
                filtered_commits = filtered_commits[:max_commits]
            
            # Converti in formato finale
            commit_data_list = []
            for commit in filtered_commits:
                try:
                    # Parse della data
                    commit_date = datetime.datetime.fromisoformat(
                        commit['date_str'].replace(' ', 'T').rstrip('Z')
                    )
                    
                    commit_data = {
                        'commit_hash': commit['hash'],
                        'author': commit['author'] or "Unknown",
                        'date': commit_date.isoformat(),
                        'message': commit['message'] or "",
                        'repo_name': self.repo_info.name
                    }
                    
                    commit_data_list.append(commit_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing commit {commit['hash']}: {e}")
            
            # Log statistiche
            original_count = len(commit_lines)
            unique_count = len(unique_commit_list)
            filtered_count = len(filtered_commits)
            final_count = len(commit_data_list)
            
            logger.info(f"Commit filtering: {original_count} raw -> {unique_count} unique -> {filtered_count} real -> {final_count} final")
            
            if original_count != final_count:
                removed_count = original_count - final_count
                logger.info(f"Removed {removed_count} duplicate/merge/parent commits")
            
            return commit_data_list
            
        except Exception as e:
            logger.error(f"Error extracting commits: {e}")
            raise
    
    def _is_valid_commit_hash(self, commit_hash: str) -> bool:
        """Verifica se un hash commit è valido."""
        return (
            len(commit_hash) == 40 and  # SHA-1 completo
            all(c in '0123456789abcdef' for c in commit_hash.lower())
        )
    
    def _filter_parent_commits(self, commits: List[Dict]) -> List[Dict]:
        """Filtra i parent commits mantenendo quelli importanti per smell analysis."""
        if not commits:
            return commits
        
        try:
            # Ottieni tutti gli hash per verificare parent relationships
            commit_hashes = {commit['hash'] for commit in commits}
            
            # Verifica quali commit hanno modifiche rilevanti
            relevant_commits = []
            
            for commit in commits:
                commit_hash = commit['hash']
                
                # Criteri di inclusione per smell analysis
                should_include = False
                reason = ""
                
                # 1. Ha modifiche a file Python
                if self._commit_has_python_changes(commit_hash):
                    should_include = True
                    reason = "python_changes"
                
                # 2. È un merge commit che potrebbe impattare smell
                elif self._is_smell_relevant_merge(commit_hash):
                    should_include = True
                    reason = "smell_relevant_merge"
                
                # 3. È un commit di refactoring (importante per smell evolution)
                elif self._is_refactoring_commit(commit['message']):
                    should_include = True
                    reason = "refactoring"
                
                # 4. Ha modifiche a file di test (possono indicare fix di smell)
                elif self._has_test_changes(commit_hash):
                    should_include = True
                    reason = "test_changes"
                
                if should_include and self._is_real_commit(commit_hash):
                    relevant_commits.append(commit)
                    logger.debug(f"Included commit {commit_hash[:8]} - reason: {reason}")
                else:
                    logger.debug(f"Filtered commit {commit_hash[:8]} - no smell relevance")
            
            return relevant_commits
            
        except Exception as e:
            logger.warning(f"Error filtering parent commits: {e}")
            # Fallback: restituisci tutti i commit
            return commits
    
    def _is_smell_relevant_merge(self, commit_hash: str) -> bool:
        """Verifica se un merge commit è rilevante per l'analisi degli smell."""
        try:
            # Verifica se è un merge commit
            result = subprocess.run(
                ['git', 'show', '--pretty=format:%P', '--no-patch', commit_hash],
                cwd=self.repo_info.local_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                parents = result.stdout.strip().split()
                
                # Se ha più di un parent, è un merge commit
                if len(parents) > 1:
                    # Verifica se il merge introduce/rimuove file Python
                    merge_changes = subprocess.run(
                        ['git', 'show', '--name-only', '--pretty=format:', commit_hash],
                        cwd=self.repo_info.local_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if merge_changes.returncode == 0:
                        changed_files = [f.strip() for f in merge_changes.stdout.strip().split('\n') if f.strip()]
                        python_files = [f for f in changed_files if CodeAnalysisUtils.is_code_file(f)]
                        
                        # Include merge se tocca file Python
                        return len(python_files) > 0
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking merge relevance for {commit_hash[:8]}: {e}")
            return False
    
    def _is_refactoring_commit(self, message: str) -> bool:
        """Verifica se un commit è di refactoring (importante per smell evolution)."""
        refactoring_keywords = [
            'refactor', 'refactoring', 'restructure', 'reorganize',
            'cleanup', 'clean up', 'simplify', 'optimize',
            'extract', 'rename', 'move', 'split',
            'consolidate', 'merge functions', 'code smell',
            'technical debt', 'improve code quality'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in refactoring_keywords)
    
    def _has_test_changes(self, commit_hash: str) -> bool:
        """Verifica se il commit ha modifiche a file di test."""
        try:
            result = subprocess.run(
                ['git', 'show', '--name-only', '--pretty=format:', commit_hash],
                cwd=self.repo_info.local_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                changed_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                
                # Verifica file di test
                test_patterns = ['test_', '_test.py', '/test/', '/tests/', 'spec_', '_spec.py']
                test_files = [f for f in changed_files 
                             if any(pattern in f.lower() for pattern in test_patterns)]
                
                return len(test_files) > 0
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking test changes for {commit_hash[:8]}: {e}")
            return False
    
    def _commit_has_python_changes(self, commit_hash: str) -> bool:
        """Verifica se un commit ha modifiche a file Python."""
        try:
            result = subprocess.run(
                ['git', 'show', '--name-only', '--pretty=format:', commit_hash],
                cwd=self.repo_info.local_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                changed_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                python_files = [f for f in changed_files if CodeAnalysisUtils.is_code_file(f)]
                return len(python_files) > 0
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking Python changes for {commit_hash[:8]}: {e}")
            return True  # Assume ha modifiche se non possiamo verificare
    
    def _is_real_commit(self, commit_hash: str) -> bool:
        """Verifica se è un commit reale con modifiche substantive."""
        try:
            # Verifica il numero di parent
            result = subprocess.run(
                ['git', 'show', '--pretty=format:%P', '--no-patch', commit_hash],
                cwd=self.repo_info.local_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                parents = result.stdout.strip().split()
                
                # Se ha più di un parent, è un merge commit -> salta
                if len(parents) > 1:
                    return False
                
                # Verifica che abbia effettivamente modifiche ai contenuti
                diff_result = subprocess.run(
                    ['git', 'show', '--stat', '--pretty=format:', commit_hash],
                    cwd=self.repo_info.local_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if diff_result.returncode == 0:
                    # Se ha output di diff, ha modifiche reali
                    return bool(diff_result.stdout.strip())
            
            return True  # Assume sia reale se non possiamo verificare
            
        except Exception as e:
            logger.debug(f"Error checking real commit for {commit_hash[:8]}: {e}")
            return True  # Assume sia reale se non possiamo verificare
    
    def _finalize_analysis(self):
        """Finalizza l'analisi e genera i report."""
        logger.info("Finalizing analysis...")
        
        if self._progress:
            self._progress.log("Loading results for finalization...")
        
        try:
            # Carica tutti i risultati
            all_results = self.processor.get_all_results()
            
            if not all_results:
                logger.warning("No results to finalize")
                return
            
            logger.info(f"Loaded {len(all_results)} results for finalization")
            
            # 1. Calcola metriche aggregate (veloce)
            if self._progress:
                self._progress.log("Calculating aggregate metrics...")
            self._calculate_aggregate_metrics(all_results)
            
            # 2. Usa dati project già raccolti (istantaneo!)
            if self._progress:
                self._progress.log("Processing project metrics from collected data...")
            self._process_collected_project_metrics(all_results)
            
            # 3. Processa smell evolution
            if self._progress:
                self._progress.log("Processing smell evolution...")
            smell_tracker = self._process_smell_evolution(all_results)
            
            # 4. Salva risultati finali
            if self._progress:
                self._progress.log("Saving final results...")
            self._save_final_results(all_results, smell_tracker)
            
            # 5. Genera report
            if self._progress:
                self._progress.log("Generating reports...")
            self._generate_reports(all_results)
            
            logger.info(f"Analysis finalized successfully. Processed {len(all_results)} commits.")
            
            if self._progress:
                self._progress.log("✅ Analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Error finalizing analysis: {e}")
            if self._progress:
                self._progress.log(f"❌ Error during finalization: {str(e)[:50]}...")
            raise
    
    def _process_collected_project_metrics(self, results: List[Dict]):
        """Processa le metriche di progetto usando i dati già raccolti durante l'analisi."""
        logger.info("Processing project metrics from collected data...")
        
        try:
            processed_count = 0
            
            for result in results:
                # Usa i dati project già raccolti durante l'analisi
                project_estimated_complexity = result.get('project_estimated_complexity', 0)
                
                # Se abbiamo dati project raccolti, usali
                if project_estimated_complexity > 0:
                    result['project_cyclomatic_complexity'] = project_estimated_complexity
                else:
                    # Fallback: usa la complessità del commit come approssimazione
                    result['project_cyclomatic_complexity'] = result.get('commit_cyclomatic_complexity', 0)
                
                # Rimuovi i campi temporanei di project data per risparmiare spazio
                for temp_field in ['project_total_python_files', 'project_estimated_complexity', 'project_total_loc']:
                    result.pop(temp_field, None)
                
                processed_count += 1
                
                # Progress ogni 1000 risultati
                if processed_count % 1000 == 0:
                    logger.info(f"Processed project metrics for {processed_count}/{len(results)} commits")
                    if self._progress:
                        self._progress.log(f"Project metrics: {processed_count}/{len(results)} processed")
            
            logger.info(f"Project metrics processing completed for {processed_count} commits")
            
        except Exception as e:
            logger.error(f"Error processing collected project metrics: {e}")
            # Fallback semplice
            for result in results:
                if 'project_cyclomatic_complexity' not in result:
                    result['project_cyclomatic_complexity'] = result.get('commit_cyclomatic_complexity', 0)
    
    def _calculate_project_metrics_optimized(self, results: List[Dict]):
        """Calcola metriche di progetto in modo ottimizzato."""
        logger.info("Calculating optimized project metrics...")
        
        try:
            total_results = len(results)
            
            # Strategia ottimizzata: calcola solo per campioni rappresentativi
            if total_results > 1000:
                # Per repository grandi, campiona ogni 50 commit + primi/ultimi 100
                sample_indices = set()
                
                # Primi 100 commit
                sample_indices.update(range(min(100, total_results)))
                
                # Ultimi 100 commit  
                sample_indices.update(range(max(0, total_results - 100), total_results))
                
                # Campione ogni 50 commit nel mezzo
                for i in range(100, total_results - 100, 50):
                    sample_indices.add(i)
                
                sample_indices = sorted(sample_indices)
                logger.info(f"Large repository: sampling {len(sample_indices)} commits out of {total_results}")
                
            else:
                # Per repository piccoli, analizza tutti ma con timeout
                sample_indices = list(range(total_results))
                logger.info(f"Small repository: analyzing all {total_results} commits")
            
            # Calcola metriche per commit campionati
            calculated_metrics = {}
            
            for i, idx in enumerate(sample_indices):
                result = results[idx]
                commit_hash = result['commit_hash']
                
                try:
                    # Timeout per singolo commit per evitare blocchi
                    project_complexity, accurate_loc_stats = self._analyze_project_at_commit_fast(commit_hash)
                    
                    calculated_metrics[idx] = {
                        'project_complexity': project_complexity,
                        'loc_stats': accurate_loc_stats
                    }
                    
                    # Progress ogni 10 commit
                    if (i + 1) % 10 == 0:
                        progress_pct = (i + 1) / len(sample_indices) * 100
                        logger.info(f"Project metrics progress: {i + 1}/{len(sample_indices)} ({progress_pct:.1f}%)")
                        
                        if self._progress:
                            self._progress.log(f"Project metrics: {i + 1}/{len(sample_indices)} calculated")
                
                except Exception as e:
                    logger.debug(f"Failed to calculate project metrics for commit {commit_hash[:8]}: {e}")
                    calculated_metrics[idx] = {
                        'project_complexity': 0,
                        'loc_stats': None
                    }
            
            # Interpola metriche per commit non campionati
            self._interpolate_project_metrics(results, calculated_metrics, sample_indices)
            
            logger.info("Project metrics calculation completed")
            
        except Exception as e:
            logger.warning(f"Error in project metrics calculation: {e}")
            # Fallback: usa metriche approssimate
            for result in results:
                result['project_cyclomatic_complexity'] = result.get('commit_cyclomatic_complexity', 0)
            logger.info("Using approximate project metrics as fallback")
    
    def _analyze_project_at_commit_fast(self, commit_hash: str) -> Tuple[int, Optional[Dict]]:
        """Versione veloce dell'analisi di progetto con timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Project analysis timeout")
        
        try:
            # Imposta timeout di 30 secondi per commit
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                # Usa git ls-tree invece di checkout completo (molto più veloce)
                result = subprocess.run(
                    ['git', 'ls-tree', '-r', '--name-only', commit_hash],
                    cwd=self.repo_info.local_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                project_complexity = 0
                
                if result.returncode == 0:
                    all_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                    python_files = [f for f in all_files if CodeAnalysisUtils.is_code_file(f)]
                    
                    # Stima complessità basata sul numero di file Python (veloce)
                    # Usa euristica: file Python medio ha ~10 unità di complessità
                    project_complexity = len(python_files) * 10
                
                # LOC accurate usando git show --stat (veloce)
                loc_stats = self._get_accurate_loc_stats(commit_hash)
                
                return project_complexity, loc_stats
                
            finally:
                signal.alarm(0)  # Cancella timeout
                
        except (TimeoutError, subprocess.TimeoutExpired):
            logger.debug(f"Timeout calculating project metrics for {commit_hash[:8]}")
            return 0, None
        except Exception as e:
            logger.debug(f"Error in fast project analysis for {commit_hash[:8]}: {e}")
            return 0, None
    
    def _interpolate_project_metrics(self, results: List[Dict], calculated_metrics: Dict[int, Dict], sample_indices: List[int]):
        """Interpola le metriche di progetto per commit non campionati."""
        try:
            # Applica metriche calcolate
            for idx, metrics in calculated_metrics.items():
                result = results[idx]
                result['project_cyclomatic_complexity'] = metrics['project_complexity']
                
                if metrics['loc_stats']:
                    result['LOC_added'] = metrics['loc_stats'].get('added', result['LOC_added'])
                    result['LOC_deleted'] = metrics['loc_stats'].get('deleted', result['LOC_deleted'])
            
            # Interpola linearmente per commit intermedi
            for i in range(len(results)):
                if i not in calculated_metrics:
                    # Trova i due campioni più vicini
                    prev_idx = max([idx for idx in sample_indices if idx < i], default=0)
                    next_idx = min([idx for idx in sample_indices if idx > i], default=len(results) - 1)
                    
                    if prev_idx in calculated_metrics and next_idx in calculated_metrics:
                        # Interpolazione lineare
                        prev_complexity = calculated_metrics[prev_idx]['project_complexity']
                        next_complexity = calculated_metrics[next_idx]['project_complexity']
                        
                        if next_idx != prev_idx:
                            ratio = (i - prev_idx) / (next_idx - prev_idx)
                            interpolated = int(prev_complexity + ratio * (next_complexity - prev_complexity))
                        else:
                            interpolated = prev_complexity
                        
                        results[i]['project_cyclomatic_complexity'] = interpolated
                    else:
                        # Fallback
                        results[i]['project_cyclomatic_complexity'] = results[i].get('commit_cyclomatic_complexity', 0)
            
            logger.info("Project metrics interpolation completed")
            
        except Exception as e:
            logger.warning(f"Error interpolating project metrics: {e}")
            # Fallback semplice
            for result in results:
                if 'project_cyclomatic_complexity' not in result:
                    result['project_cyclomatic_complexity'] = result.get('commit_cyclomatic_complexity', 0)
    
    def _get_accurate_loc_stats(self, commit_hash: str) -> Optional[Dict]:
        """Ottiene statistiche LOC accurate per un commit (con timeout)."""
        try:
            result = subprocess.run(
                ['git', 'show', '--stat', '--pretty=format:', commit_hash],
                cwd=self.repo_info.local_path,
                capture_output=True,
                text=True,
                timeout=10  # Timeout ridotto
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                
                # Cerca la linea di summary (ultima linea con pattern)
                for line in reversed(lines):
                    if 'changed' in line and ('insertion' in line or 'deletion' in line):
                        added = 0
                        deleted = 0
                        
                        # Estrai insertions
                        if 'insertion' in line:
                            import re
                            match = re.search(r'(\d+) insertion', line)
                            if match:
                                added = int(match.group(1))
                        
                        # Estrai deletions
                        if 'deletion' in line:
                            import re
                            match = re.search(r'(\d+) deletion', line)
                            if match:
                                deleted = int(match.group(1))
                        
                        return {'added': added, 'deleted': deleted}
                        
            return None
            
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Error/timeout getting LOC stats for {commit_hash[:8]}: {e}")
            return None
    
    def _calculate_aggregate_metrics(self, results: List[Dict]):
        """Calcola metriche aggregate che dipendono dalla sequenza completa dei commit."""
        logger.info("Calculating aggregate metrics...")
        
        author_stats = {}
        last_commit_by_author = {}
        
        for i, result in enumerate(results):
            author = result['author']
            commit_date = datetime.datetime.fromisoformat(result['date'])
            
            # Author experience: numero di commit precedenti dello stesso autore
            result['author_experience'] = author_stats.get(author, 0)
            author_stats[author] = author_stats.get(author, 0) + 1
            
            # Time since last commit: tempo dall'ultimo commit dello stesso autore
            time_since_last = 0.0
            if author in last_commit_by_author:
                time_delta = commit_date - last_commit_by_author[author]
                time_since_last = time_delta.total_seconds() / 3600  # In ore
            
            result['time_since_last_commit'] = time_since_last
            last_commit_by_author[author] = commit_date
            
            # Progress logging ogni 500 commit
            if (i + 1) % 500 == 0:
                logger.info(f"Calculated aggregate metrics for {i + 1}/{len(results)} commits")
        
        # Statistiche finali
        total_authors = len(author_stats)
        avg_commits_per_author = sum(author_stats.values()) / total_authors if total_authors > 0 else 0
        
        logger.info(f"Aggregate metrics completed: {total_authors} authors, avg {avg_commits_per_author:.1f} commits/author")
    
    def _get_accurate_loc_stats(self, commit_hash: str) -> Optional[Dict]:
        """Ottiene statistiche LOC accurate per un commit (con timeout)."""
        try:
            result = subprocess.run(
                ['git', 'show', '--stat', '--pretty=format:', commit_hash],
                cwd=self.repo_info.local_path,
                capture_output=True,
                text=True,
                timeout=10  # Timeout ridotto
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                
                # Cerca la linea di summary (ultima linea con pattern)
                for line in reversed(lines):
                    if 'changed' in line and ('insertion' in line or 'deletion' in line):
                        added = 0
                        deleted = 0
                        
                        # Estrai insertions
                        if 'insertion' in line:
                            import re
                            match = re.search(r'(\d+) insertion', line)
                            if match:
                                added = int(match.group(1))
                        
                        # Estrai deletions
                        if 'deletion' in line:
                            import re
                            match = re.search(r'(\d+) deletion', line)
                            if match:
                                deleted = int(match.group(1))
                        
                        return {'added': added, 'deleted': deleted}
                        
            return None
            
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Error/timeout getting LOC stats for {commit_hash[:8]}: {e}")
            return None
    
    def _process_smell_evolution(self, results: List[Dict]) -> SmellEvolutionTracker:
        """Processa l'evoluzione degli smell."""
        smell_tracker = SmellEvolutionTracker()
        
        try:
            for result in results:
                commit_hash = result['commit_hash']
                commit_date = datetime.datetime.fromisoformat(result['date'])
                changed_files = result.get('changed_files', [])
                smells_by_file = result.get('smells_by_file', {})
                
                smell_tracker.process_commit_smells(commit_hash, commit_date, changed_files, smells_by_file)
                
                # Aggiungi eventi smell al risultato
                result['smell_events'] = []
                result['smells_introduced'] = 0
                result['smells_removed'] = 0
                result['smells_modified'] = 0
                result['smells_persisted'] = 0
                result['files_with_smells'] = list(smells_by_file.keys())
                result['files_smell_introduced'] = []
                result['files_smell_removed'] = []
        
        except Exception as e:
            logger.error(f"Error processing smell evolution: {e}")
        
        return smell_tracker
    
    def _save_final_results(self, results: List[Dict], smell_tracker: SmellEvolutionTracker):
        """Salva i risultati finali."""
        try:
            # Prepara dati per CSV
            csv_data = []
            for result in results:
                csv_row = result.copy()
                # Converte oggetti complessi in JSON
                for field in ['changed_files_complexity', 'smells_by_file', 'smells_by_type', 
                             'smell_events', 'files_with_smells', 'files_smell_introduced', 'files_smell_removed']:
                    csv_row[field] = json.dumps(csv_row.get(field, {}))
                csv_row.pop('changed_files', None)
                csv_data.append(csv_row)
            
            # Salva CSV
            df = pd.DataFrame(csv_data)
            output_file = os.path.join(self.repo_info.output_dir, "commit_metrics.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # Salva JSON
            json_file = os.path.join(self.repo_info.output_dir, "commit_metrics.json")
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Salva smell evolution
            smell_file = os.path.join(self.repo_info.output_dir, "smell_evolution.json")
            with open(smell_file, 'w') as f:
                json.dump(smell_tracker.get_smell_evolution_report(), f, indent=2)
            
            # Salva file frequencies
            self._save_file_frequencies(results)
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
    
    def _save_file_frequencies(self, results: List[Dict]):
        """Salva le frequenze dei file."""
        edited_files = defaultdict(int)
        bugged_files = defaultdict(int)
        
        for result in results:
            changed_files = result.get('changed_files', [])
            is_bug_fix = result.get('is_bug_fix', False)
            
            for file_path in changed_files:
                if CodeAnalysisUtils.is_code_file(file_path):
                    edited_files[file_path] += 1
                    if is_bug_fix:
                        bugged_files[file_path] += 1
        
        freq_data = {
            'edited_files': dict(edited_files),
            'bugged_files': dict(bugged_files),
            'edited_files_complexity': {}
        }
        
        freq_file = os.path.join(self.repo_info.output_dir, "file_frequencies.json")
        with open(freq_file, 'w') as f:
            json.dump(freq_data, f, indent=2)
    
    def _generate_reports(self, results: List[Dict]):
        """Genera i report finali."""
        try:
            df = pd.DataFrame(results)
            
            # Rimuovi colonne complesse per il report
            report_columns = ['changed_files_complexity', 'smells_by_file', 'smells_by_type', 
                            'smell_events', 'files_with_smells', 'files_smell_introduced', 
                            'files_smell_removed', 'analysis_timestamp']
            
            report_df = df.drop(columns=[col for col in report_columns if col in df.columns])
            
            create_single_repo_report(self.repo_info.output_dir, report_df, self.repo_info.name)
            logger.info("Reports generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
    
    def cleanup(self):
        """Cleanup delle directory temporanee."""
        if hasattr(self, 'repo_info') and self.repo_info.temp_dir and os.path.exists(self.repo_info.temp_dir):
            try:
                shutil.rmtree(self.repo_info.temp_dir)
                logger.info("Temporary directories cleaned")
            except Exception as e:
                logger.warning(f"Error cleaning up: {e}")
    
    def get_analysis_summary(self) -> Dict:
        """Restituisce un riassunto dell'analisi."""
        processed_count = self.processor.get_processed_count()
        
        return {
            'total_commits_analyzed': processed_count,
            'total_commits_skipped': 0,  # Semplificato
            'total_commits_saved': processed_count,
            'commits_currently_processing': 0,
            'max_workers_used': self.max_workers,
            'output_directory': self.repo_info.output_dir
        }
    
    # Metodi per compatibilità
    def save_results(self):
        """Compatibilità - i risultati sono già salvati."""
        pass
    
    def save_checkpoint(self, last_commit_hash: str):
        """Compatibilità - checkpoint automatico."""
        pass
    
    def load_checkpoint(self):
        """Compatibilità - caricamento automatico."""
        return None
    
    @staticmethod
    def can_resume_from_sequential(repo_info: RepositoryInfo) -> bool:
        """Verifica possibilità di resume da sequenziale."""
        return os.path.exists(os.path.join(repo_info.output_dir or repo_info.name, "commit_metrics.csv"))
    
    def resume_from_sequential(self):
        """Resume da analisi sequenziale (semplificato)."""
        return True

# Factory per compatibilità
class RepoAnalyzerFactory:
    """Factory per creare l'analyzer appropriato."""
    
    @staticmethod
    def create_analyzer(repo_info: RepositoryInfo, resume: bool = False, 
                       parallel: bool = True, max_workers: Optional[int] = None,
                       smell_analysis_mode: str = "comprehensive") -> Any:
        """
        Create appropriate analyzer.
        
        Args:
            repo_info: Repository information
            resume: Resume interrupted analysis
            parallel: Use parallel mode
            max_workers: Max parallel workers
            smell_analysis_mode: Commit filtering strategy for smell analysis
                - "comprehensive": Include smell-relevant commits (recommended)
                - "minimal": Only direct Python changes
                - "all": All commits (most complete but slower)
        """
        if parallel:
            return ParallelRepoAnalyzer(repo_info, resume, max_workers, smell_analysis_mode)
        else:
            from .repo_analyzer import RepoAnalyzer
            return RepoAnalyzer(repo_info, resume)
    
    @staticmethod
    def get_recommended_mode(repo_info: RepositoryInfo) -> Tuple[bool, int]:
        """Raccomanda modalità di analisi."""
        return True, min(16, multiprocessing.cpu_count())