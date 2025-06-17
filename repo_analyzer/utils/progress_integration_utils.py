#!/usr/bin/env python3
"""
Utilità complete per integrare il progress display avanzato negli analyzer esistenti.
Questo modulo fornisce funzioni helper e decoratori per una facile integrazione.
"""

import functools
import logging
import traceback
import time
from typing import Optional, Callable, Any, Dict, List, Set, Tuple
from .progress_display import EnhancedProgressDisplay, LogLevel, EnhancedProgressContext

# ===============================================
# CONFIGURAZIONE LOGGING
# ===============================================

class ProgressLogHandler(logging.Handler):
    """Handler personalizzato per catturare log e mostrarli nel progress display."""
    
    def __init__(self, progress_display: EnhancedProgressDisplay):
        super().__init__()
        self.progress = progress_display
        
    def emit(self, record):
        """Emette il record di log nel progress display."""
        try:
            # Mappa i livelli di logging ai nostri LogLevel
            level_mapping = {
                logging.DEBUG: LogLevel.DEBUG,
                logging.INFO: LogLevel.INFO,
                logging.WARNING: LogLevel.WARNING,
                logging.ERROR: LogLevel.ERROR,
                logging.CRITICAL: LogLevel.ERROR
            }
            
            log_level = level_mapping.get(record.levelno, LogLevel.INFO)
            message = record.getMessage()
            
            # Estrai commit hash dal messaggio se possibile
            commit_hash = None
            if hasattr(record, 'commit_hash'):
                commit_hash = record.commit_hash
            else:
                # Prova a estrarre dall'inizio del messaggio (formato comune: "commit abc123: message")
                import re
                match = re.match(r'.*?([a-f0-9]{7,40})', message)
                if match:
                    commit_hash = match.group(1)
            
            # Aggiungi dettagli extra se disponibili
            details = None
            if hasattr(record, 'details'):
                details = record.details
            elif record.exc_info:
                details = traceback.format_exception(*record.exc_info)[0].strip()
            
            self.progress.log(message, log_level, commit_hash, details)
            
        except Exception:
            # Non fare crash se c'è un problema con il logging
            pass

def setup_progress_logging(progress_display: EnhancedProgressDisplay, 
                          logger_names: list = None) -> list:
    """
    Configura il logging per catturare messaggi degli analyzer.
    
    Args:
        progress_display: Display progress dove mostrare i log
        logger_names: Lista dei nomi dei logger da catturare (default: logger analyzer comuni)
    
    Returns:
        Lista degli handler aggiunti (per cleanup successivo)
    """
    if logger_names is None:
        logger_names = [
            'repo_analyzer',
            'repo_analyzer.repo_analyzer',
            'repo_analyzer.parallel_repo_analyzer', 
            'repo_analyzer.multi_repo_analyzer',
            'repo_analyzer.code_analysis',
            'repo_analyzer.git_utils'
        ]
    
    handlers = []
    progress_handler = ProgressLogHandler(progress_display)
    
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.addHandler(progress_handler)
        handlers.append((logger, progress_handler))
    
    return handlers

def cleanup_progress_logging(handlers: list):
    """Rimuove gli handler del progress logging."""
    for logger, handler in handlers:
        logger.removeHandler(handler)

# ===============================================
# ANALISI ERRORI E SKIP REASONS
# ===============================================

def analyze_skip_reason(commit, exception=None, context=None) -> tuple:
    """
    Analizza un commit per determinare la ragione dello skip.
    
    Args:
        commit: Oggetto commit Git
        exception: Eventuale eccezione che ha causato lo skip
        context: Contesto aggiuntivo (dict con info extra)
    
    Returns:
        Tupla (reason, details)
    """
    context = context or {}
    
    # Se c'è un'eccezione, analizzala prima
    if exception:
        return analyze_exception_for_skip(exception)
    
    # Analisi del commit stesso
    try:
        # Controlla il messaggio del commit
        if hasattr(commit, 'message'):
            message = commit.message.lower()
            
            # Merge commits
            if any(keyword in message for keyword in ['merge', 'merged']):
                if 'pull request' in message or 'pr #' in message:
                    return ("Merge Commit - PR", f"Pull request merge: '{commit.message[:50]}...'")
                elif 'branch' in message:
                    return ("Merge Commit - Branch", f"Branch merge: '{commit.message[:50]}...'")
                else:
                    return ("Merge Commit", f"General merge: '{commit.message[:50]}...'")
            
            # Commit automatici
            if any(keyword in message for keyword in ['automatic', 'auto-generated', 'bot']):
                return ("Automated Commit", "Detected automated/bot commit")
            
            # Commit di documentazione solo
            if any(keyword in message for keyword in ['readme', 'doc', 'documentation', 'comment']):
                if not any(keyword in message for keyword in ['fix', 'add', 'update', 'implement']):
                    return ("Documentation Only", "Appears to be documentation-only change")
        
        # Analizza i file modificati
        if hasattr(commit, 'stats') and hasattr(commit.stats, 'files'):
            files = list(commit.stats.files.keys())
            
            if not files:
                return ("Empty Commit", "No files changed in commit")
            
            # Conta i file Python
            python_files = [f for f in files if f.endswith('.py')]
            config_files = [f for f in files if any(f.endswith(ext) for ext in ['.json', '.yml', '.yaml', '.toml', '.ini', '.cfg'])]
            doc_files = [f for f in files if any(f.endswith(ext) for ext in ['.md', '.rst', '.txt']) or 'doc' in f.lower()]
            test_files = [f for f in files if 'test' in f.lower() or f.startswith('test_')]
            
            total_files = len(files)
            
            if not python_files:
                if config_files:
                    return ("Configuration Changes", f"Only config files changed: {len(config_files)} files")
                elif doc_files:
                    return ("Documentation Changes", f"Only documentation changed: {len(doc_files)} files")
                elif test_files:
                    return ("Test Files Only", f"Only test files changed: {len(test_files)} files")
                else:
                    return ("No Python Files", f"No Python files in {total_files} changed files")
            
            # Pochi file Python rispetto al totale
            python_ratio = len(python_files) / total_files
            if python_ratio < 0.1 and total_files > 10:
                return ("Few Python Files", f"Only {len(python_files)}/{total_files} files are Python ({python_ratio:.1%})")
        
        # Controlla la dimensione del commit
        if hasattr(commit, 'stats') and hasattr(commit.stats, 'total'):
            total_changes = commit.stats.total.get('lines', 0)
            if total_changes == 0:
                return ("No Changes", "Commit has no line changes")
            elif total_changes > 10000:
                return ("Large Commit", f"Very large commit: {total_changes} line changes")
        
        # Controlla i parent (commit iniziale, etc.)
        if hasattr(commit, 'parents'):
            if len(commit.parents) == 0:
                return ("Initial Commit", "Repository initial commit")
            elif len(commit.parents) > 2:
                return ("Complex Merge", f"Merge commit with {len(commit.parents)} parents")
        
        # Analisi basata sul contesto fornito
        if 'no_code_files' in context:
            return ("No Code Files", context.get('details', 'No analyzable code files found'))
        
        if 'analysis_timeout' in context:
            return ("Analysis Timeout", context.get('details', 'Commit analysis timed out'))
        
        if 'memory_limit' in context:
            return ("Memory Limit", context.get('details', 'Insufficient memory for analysis'))
        
        # Ragione generica
        return ("Analysis Failed", "Unable to complete commit analysis")
        
    except Exception as e:
        return ("Skip Analysis Error", f"Error determining skip reason: {str(e)}")

def analyze_exception_for_skip(exception) -> tuple:
    """
    Analizza un'eccezione per determinare ragione e dettagli dello skip.
    
    Args:
        exception: Eccezione da analizzare
    
    Returns:
        Tupla (reason, details)
    """
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__
    
    # Errori di Git
    if 'git' in exception_type.lower() or 'git' in exception_str:
        if 'checkout' in exception_str:
            return ("Git Checkout Failed", f"Cannot checkout commit: {str(exception)}")
        elif 'clone' in exception_str:
            return ("Git Clone Failed", f"Cannot clone repository: {str(exception)}")
        elif 'permission' in exception_str:
            return ("Git Permission Error", f"Git permission denied: {str(exception)}")
        else:
            return ("Git Operation Failed", f"Git error: {str(exception)}")
    
    # Errori di file system
    elif 'permission' in exception_str or 'access denied' in exception_str:
        return ("Permission Denied", f"File access denied: {str(exception)}")
    elif 'file not found' in exception_str or 'no such file' in exception_str:
        return ("File Not Found", f"Required file missing: {str(exception)}")
    elif 'disk' in exception_str or 'space' in exception_str:
        return ("Disk Space Error", f"Insufficient disk space: {str(exception)}")
    
    # Errori di memoria e performance
    elif 'memory' in exception_str or 'memoryerror' in exception_type.lower():
        return ("Out of Memory", f"Insufficient memory: {str(exception)}")
    elif 'timeout' in exception_str or 'time' in exception_str:
        return ("Operation Timeout", f"Operation timed out: {str(exception)}")
    
    # Errori di encoding
    elif 'encoding' in exception_str or 'unicode' in exception_str or 'decode' in exception_str:
        return ("Encoding Error", f"File encoding issue: {str(exception)}")
    
    # Errori di parsing/formato
    elif 'json' in exception_str or 'parse' in exception_str or 'syntax' in exception_str:
        return ("Format Error", f"Data format/parsing error: {str(exception)}")
    
    # Errori di import/dipendenze
    elif 'import' in exception_str or 'module' in exception_str:
        return ("Dependency Error", f"Missing dependency: {str(exception)}")
    
    # Errore generico
    else:
        return (f"{exception_type}", f"Unexpected error: {str(exception)}")

def classify_error_severity(exception) -> str:
    """
    Classifica la severità di un errore.
    
    Args:
        exception: Eccezione da classificare
    
    Returns:
        Severità: 'low', 'medium', 'high', 'critical'
    """
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__
    
    # Errori critici che indicano problemi sistemici gravi
    if any(keyword in exception_str for keyword in ['system', 'kernel', 'core dump', 'segmentation']):
        return 'critical'
    elif 'memoryerror' in exception_type.lower() or 'out of memory' in exception_str:
        return 'critical'
    
    # Errori ad alta severità che potrebbero richiedere intervento
    elif any(keyword in exception_str for keyword in ['permission', 'access denied', 'disk space']):
        return 'high'
    elif 'git' in exception_str and 'fatal' in exception_str:
        return 'high'
    
    # Errori medi - problemi che potrebbero essere risolti
    elif any(keyword in exception_str for keyword in ['timeout', 'connection', 'network']):
        return 'medium'
    elif 'encoding' in exception_str or 'unicode' in exception_str:
        return 'medium'
    
    # Errori bassi - problemi minori o specifici del commit
    elif any(keyword in exception_str for keyword in ['file not found', 'no such file']):
        return 'low'
    elif 'parse' in exception_str or 'format' in exception_str:
        return 'low'
    
    # Default medio per errori non classificati
    else:
        return 'medium'

# ===============================================
# DECORATORI E WRAPPER
# ===============================================

def create_progress_wrapper(progress_display: EnhancedProgressDisplay):
    """
    Crea un wrapper che può essere usato come decorator per metodi di analisi.
    
    Args:
        progress_display: Display progress da usare
    
    Returns:
        Decorator function
    """
    def progress_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Estrai commit dalle args se possibile
            commit = None
            commit_hash = None
            
            # Cerca il commit nei parametri
            for arg in args:
                if hasattr(arg, 'hexsha'):  # Oggetto commit Git
                    commit = arg
                    commit_hash = arg.hexsha
                    break
                elif isinstance(arg, str) and len(arg) in [7, 8, 40]:  # Hash commit
                    commit_hash = arg
                    break
            
            try:
                # Esegui la funzione originale
                result = func(*args, **kwargs)
                
                if result is None:
                    # Funzione ha restituito None - commit skippato
                    reason, details = analyze_skip_reason(commit, context=kwargs)
                    progress_display.increment_skipped(commit_hash or "unknown", reason, details)
                else:
                    # Successo
                    progress_display.increment_analyzed(commit_hash or "unknown")
                    
                    # Controlla per warning nel risultato
                    if hasattr(result, 'num_warnings') and result.num_warnings > 0:
                        progress_display.log_warning(
                            f"Analysis completed with {result.num_warnings} warnings",
                            commit_hash
                        )
                
                return result
                
            except Exception as e:
                # Errore durante l'analisi
                reason, details = analyze_exception_for_skip(e)
                severity = classify_error_severity(e)
                
                if severity in ['low', 'medium']:
                    # Tratta come skip per errori minori
                    progress_display.increment_skipped(commit_hash or "unknown", reason, details)
                else:
                    # Tratta come errore per problemi gravi
                    progress_display.increment_errors(commit_hash or "unknown", reason, details)
                
                # Re-raise l'eccezione se è critica
                if severity == 'critical':
                    raise
                
                return None
        
        return wrapper
    return progress_decorator

def monitor_analysis_performance(progress_display: EnhancedProgressDisplay):
    """
    Decorator per monitorare le performance dell'analisi.
    
    Args:
        progress_display: Display progress dove loggare le performance
    
    Returns:
        Decorator function
    """
    def performance_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Calcola tempo elapsed
                elapsed_time = time.time() - start_time
                
                # Log performance se significativo
                if elapsed_time > 1.0:  # Solo per operazioni > 1 secondo
                    progress_display.log(
                        f"Performance: {func.__name__} took {elapsed_time:.2f}s",
                        LogLevel.DEBUG
                    )
                
                return result
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                progress_display.log(
                    f"Performance: {func.__name__} failed after {elapsed_time:.2f}s",
                    LogLevel.DEBUG
                )
                raise
        
        return wrapper
    return performance_decorator

# ===============================================
# MIXIN CLASS
# ===============================================

class ProgressAwareAnalyzer:
    """
    Mixin class che può essere usata per rendere gli analyzer progress-aware.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_display: Optional[EnhancedProgressDisplay] = None
        self._progress_handlers = []
        self._analysis_stats = {
            'start_time': None,
            'commits_processed': 0,
            'errors_encountered': 0,
            'warnings_generated': 0
        }
    
    def set_progress_display(self, progress_display: EnhancedProgressDisplay):
        """Configura il progress display."""
        self._progress_display = progress_display
        self._progress_handlers = setup_progress_logging(progress_display)
        self._analysis_stats['start_time'] = time.time()
        
        progress_display.log("Progress tracking enabled", LogLevel.INFO)
    
    def cleanup_progress_display(self):
        """Pulisce il progress display."""
        if self._progress_handlers:
            cleanup_progress_logging(self._progress_handlers)
            self._progress_handlers = []
        
        if self._progress_display:
            self._progress_display.log("Progress tracking disabled", LogLevel.INFO)
            self._progress_display = None
    
    def log_progress(self, message: str, level: LogLevel = LogLevel.INFO, 
                    commit_hash: Optional[str] = None, details: Optional[str] = None):
        """Log un messaggio nel progress display se disponibile."""
        if self._progress_display:
            self._progress_display.log(message, level, commit_hash, details)
    
    def log_commit_success(self, commit_hash: str, metrics: Optional[Dict] = None):
        """Log il successo dell'analisi di un commit."""
        if self._progress_display:
            self._progress_display.increment_analyzed(commit_hash)
            
            # Log dettagli addizionali se disponibili
            if metrics:
                details = []
                if 'total_smells_found' in metrics and metrics['total_smells_found'] > 0:
                    details.append(f"{metrics['total_smells_found']} smells")
                if 'num_warnings' in metrics and metrics['num_warnings'] > 0:
                    details.append(f"{metrics['num_warnings']} warnings")
                if 'commit_cyclomatic_complexity' in metrics:
                    details.append(f"complexity: {metrics['commit_cyclomatic_complexity']:.1f}")
                
                if details:
                    self.log_progress(
                        f"Analysis complete: {', '.join(details)}",
                        LogLevel.SUCCESS,
                        commit_hash
                    )
        
        self._analysis_stats['commits_processed'] += 1
    
    def log_commit_skip(self, commit_hash: str, reason: str, details: Optional[str] = None):
        """Log il skip di un commit."""
        if self._progress_display:
            self._progress_display.increment_skipped(commit_hash, reason, details)
    
    def log_commit_error(self, commit_hash: str, error: Exception, context: Optional[Dict] = None):
        """Log un errore nell'analisi di un commit."""
        reason, details = analyze_exception_for_skip(error)
        severity = classify_error_severity(error)
        
        if self._progress_display:
            if severity in ['critical', 'high']:
                self._progress_display.increment_errors(commit_hash, reason, details)
            else:
                self._progress_display.increment_skipped(commit_hash, reason, details)
        
        self._analysis_stats['errors_encountered'] += 1
    
    def log_warning(self, message: str, commit_hash: Optional[str] = None):
        """Log un warning."""
        if self._progress_display:
            self._progress_display.log_warning(message, commit_hash)
        
        self._analysis_stats['warnings_generated'] += 1
    
    def get_analysis_performance_stats(self) -> Dict:
        """Ottieni statistiche di performance dell'analisi."""
        current_time = time.time()
        start_time = self._analysis_stats.get('start_time', current_time)
        elapsed_time = current_time - start_time
        
        stats = {
            'elapsed_time_seconds': elapsed_time,
            'commits_processed': self._analysis_stats['commits_processed'],
            'errors_encountered': self._analysis_stats['errors_encountered'],
            'warnings_generated': self._analysis_stats['warnings_generated'],
            'average_time_per_commit': elapsed_time / max(self._analysis_stats['commits_processed'], 1),
            'processing_rate_per_minute': (self._analysis_stats['commits_processed'] / elapsed_time * 60) if elapsed_time > 0 else 0
        }
        
        return stats

# ===============================================
# UTILITY FUNCTIONS PER ANALYZER ESISTENTI
# ===============================================

def patch_analyzer_with_progress(analyzer_instance, progress_display: EnhancedProgressDisplay):
    """
    Applica il progress tracking a un'istanza di analyzer esistente.
    
    Args:
        analyzer_instance: Istanza dell'analyzer da modificare
        progress_display: Progress display da usare
    """
    # Salva i metodi originali
    original_methods = {}
    
    # Lista dei metodi da wrappare
    methods_to_wrap = [
        'analyze_commit_sequential',
        '_analyze_single_commit',
        'analyze_commit',
        '_analyze_commit_common'
    ]
    
    progress_wrapper = create_progress_wrapper(progress_display)
    
    for method_name in methods_to_wrap:
        if hasattr(analyzer_instance, method_name):
            original_method = getattr(analyzer_instance, method_name)
            original_methods[method_name] = original_method
            
            # Wrap il metodo
            wrapped_method = progress_wrapper(original_method)
            setattr(analyzer_instance, method_name, wrapped_method)
    
    # Aggiungi metodi di utility
    analyzer_instance._progress_display = progress_display
    analyzer_instance._original_methods = original_methods
    analyzer_instance._progress_handlers = setup_progress_logging(progress_display)
    
    def restore_original_methods():
        """Ripristina i metodi originali."""
        for method_name, original_method in original_methods.items():
            setattr(analyzer_instance, method_name, original_method)
        cleanup_progress_logging(analyzer_instance._progress_handlers)
    
    analyzer_instance.restore_original_methods = restore_original_methods
    
    progress_display.log("Progress tracking patched into analyzer", LogLevel.INFO)

def enhance_analyzer_class(analyzer_class, progress_display: EnhancedProgressDisplay):
    """
    Migliora una classe analyzer aggiungendo progress tracking.
    
    Args:
        analyzer_class: Classe analyzer da migliorare
        progress_display: Display progress da usare
    
    Returns:
        Classe analyzer migliorata
    """
    # Crea il wrapper per i metodi di analisi
    progress_wrapper = create_progress_wrapper(progress_display)
    
    # Lista dei metodi che tipicamente analizzano commit
    commit_analysis_methods = [
        'analyze_commit',
        'analyze_commit_sequential', 
        '_analyze_single_commit',
        '_analyze_commit_common',
        'process_commit'
    ]
    
    # Applica il wrapper ai metodi rilevanti
    for method_name in commit_analysis_methods:
        if hasattr(analyzer_class, method_name):
            original_method = getattr(analyzer_class, method_name)
            wrapped_method = progress_wrapper(original_method)
            setattr(analyzer_class, method_name, wrapped_method)
    
    # Aggiungi metodi helper alla classe
    def set_progress_display(self, progress_display):
        """Imposta il progress display per questo analyzer."""
        self._progress_display = progress_display
        
        # Configura il logging per catturare i messaggi
        if not hasattr(self, '_progress_handlers'):
            self._progress_handlers = setup_progress_logging(progress_display)
    
    def cleanup_progress_display(self):
        """Pulisce il progress display e rimuove i handler."""
        if hasattr(self, '_progress_handlers'):
            cleanup_progress_logging(self._progress_handlers)
            delattr(self, '_progress_handlers')
    
    # Aggiungi i metodi alla classe
    analyzer_class.set_progress_display = set_progress_display
    analyzer_class.cleanup_progress_display = cleanup_progress_display
    
    return analyzer_class

def create_progress_aware_analyzer(base_analyzer_class):
    """
    Crea una versione progress-aware di una classe analyzer.
    
    Args:
        base_analyzer_class: Classe analyzer base
    
    Returns:
        Classe analyzer con progress tracking integrato
    """
    class ProgressAwareAnalyzerImpl(ProgressAwareAnalyzer, base_analyzer_class):
        """Implementazione progress-aware dell'analyzer."""
        
        def __init__(self, *args, **kwargs):
            # Estrai progress display dai kwargs se presente
            progress_display = kwargs.pop('progress_display', None)
            
            # Inizializza le classi parent
            super().__init__(*args, **kwargs)
            
            # Configura progress se fornito
            if progress_display:
                self.set_progress_display(progress_display)
        
        def analyze_repository(self, max_commits: Optional[int] = None):
            """Override per aggiungere progress tracking."""
            if self._progress_display:
                repo_name = getattr(self.repo_info, 'name', 'unknown-repo')
                self._progress_display.log(f"Starting analysis of {repo_name}", LogLevel.INFO)
            
            try:
                # Chiama il metodo originale
                result = super().analyze_repository(max_commits)
                
                if self._progress_display:
                    # Log statistiche finali
                    stats = self.get_analysis_performance_stats()
                    self._progress_display.log(
                        f"Repository analysis completed in {stats['elapsed_time_seconds']:.1f}s",
                        LogLevel.SUCCESS
                    )
                
                return result
                
            except Exception as e:
                if self._progress_display:
                    self._progress_display.log_error("", "Repository Analysis Failed", str(e))
                raise
            finally:
                if self._progress_display:
                    self.cleanup_progress_display()
        
        def analyze_commit_sequential(self, commit):
            """Override con progress tracking per analisi sequenziale."""
            commit_hash = getattr(commit, 'hexsha', str(commit))
            
            try:
                # Log inizio analisi
                self.log_progress(f"Analyzing commit", LogLevel.INFO, commit_hash)
                
                # Chiama il metodo originale
                result = super().analyze_commit_sequential(commit)
                
                if result is None:
                    # Commit skippato - determina la ragione
                    reason, details = analyze_skip_reason(commit)
                    self.log_commit_skip(commit_hash, reason, details)
                else:
                    # Successo - log con metriche se disponibili
                    metrics = {}
                    if hasattr(result, '__dict__'):
                        metrics = {k: v for k, v in result.__dict__.items() 
                                 if k in ['total_smells_found', 'num_warnings', 'commit_cyclomatic_complexity']}
                    
                    self.log_commit_success(commit_hash, metrics)
                
                return result
                
            except Exception as e:
                self.log_commit_error(commit_hash, e)
                return None
        
        def __del__(self):
            """Cleanup automatico."""
            try:
                self.cleanup_progress_display()
            except:
                pass
    
    return ProgressAwareAnalyzerImpl

# ===============================================
# UTILITY FUNCTIONS QUICK-PATCH
# ===============================================

def quick_patch_analyzer(analyzer, progress_display):
    """Patch veloce per analyzer esistente senza modificare il codice."""
    
    # Salva metodo originale
    if hasattr(analyzer, 'analyze_commit_sequential'):
        original_analyze = analyzer.analyze_commit_sequential
        
        def patched_analyze(commit):
            commit_hash = getattr(commit, 'hexsha', 'unknown')
            
            try:
                result = original_analyze(commit)
                
                if result is None:
                    # Cerca di capire perché è None
                    if hasattr(analyzer, 'skipped_commits') and commit_hash in analyzer.skipped_commits:
                        progress_display.log_skip(commit_hash, "Analysis Skipped", "Commit was added to skipped_commits")
                    else:
                        progress_display.log_skip(commit_hash, "Analysis Failed", "analyze_commit_sequential returned None")
                else:
                    progress_display.increment_analyzed(commit_hash)
                
                return result
                
            except Exception as e:
                progress_display.log_error(commit_hash, type(e).__name__, str(e)[:100])
                return None
        
        # Sostituisci il metodo
        analyzer.analyze_commit_sequential = patched_analyze
        analyzer._progress = progress_display

def simple_progress_integration(analyzer_instance, progress_display):
    """
    Integrazione semplice che funziona con qualsiasi analyzer.
    
    Args:
        analyzer_instance: Istanza analyzer da modificare
        progress_display: Progress display da collegare
    """
    # Riferimento al progress
    analyzer_instance._progress = progress_display
    
    # Funzione helper per logging semplificato
    def log_progress_simple(message, level="info", commit_hash=None):
        if hasattr(progress_display, 'log'):
            # Usa il nuovo sistema se disponibile
            log_level = getattr(LogLevel, level.upper(), LogLevel.INFO)
            progress_display.log(message, log_level, commit_hash)
        else:
            # Fallback per sistema base
            print(f"[{level.upper()}] {message}")
    
    # Aggiungi metodi helper all'analyzer
    analyzer_instance.log_progress = log_progress_simple
    
    # Wrap metodi esistenti se presenti
    original_methods = {}
    
    # Metodi comuni da wrappare
    method_names = [
        'analyze_commit_sequential',
        '_analyze_single_commit', 
        'analyze_commit'
    ]
    
    for method_name in method_names:
        if hasattr(analyzer_instance, method_name):
            original_method = getattr(analyzer_instance, method_name)
            original_methods[method_name] = original_method
            
            def create_wrapped_method(orig_method, m_name):
                def wrapped_method(*args, **kwargs):
                    # Estrai commit hash se possibile
                    commit_hash = "unknown"
                    for arg in args:
                        if hasattr(arg, 'hexsha'):
                            commit_hash = arg.hexsha
                            break
                    
                    try:
                        result = orig_method(*args, **kwargs)
                        
                        if result is None:
                            analyzer_instance.log_progress(f"Commit skipped", "warning", commit_hash)
                            if hasattr(progress_display, 'increment_skipped'):
                                progress_display.increment_skipped(commit_hash)
                        else:
                            analyzer_instance.log_progress(f"Commit analyzed", "success", commit_hash)
                            if hasattr(progress_display, 'increment_analyzed'):
                                progress_display.increment_analyzed(commit_hash)
                        
                        return result
                        
                    except Exception as e:
                        analyzer_instance.log_progress(f"Error: {str(e)[:50]}...", "error", commit_hash)
                        if hasattr(progress_display, 'increment_errors'):
                            progress_display.increment_errors(commit_hash, type(e).__name__, str(e))
                        elif hasattr(progress_display, 'increment_skipped'):
                            progress_display.increment_skipped(commit_hash)
                        return None
                
                return wrapped_method
            
            # Applica il wrapper
            setattr(analyzer_instance, method_name, create_wrapped_method(original_method, method_name))
    
    # Salva metodi originali per ripristino
    analyzer_instance._original_methods = original_methods
    
    return analyzer_instance

# ===============================================
# ESEMPI E TEST
# ===============================================

def test_progress_integration():
    """Test delle funzionalità di integrazione."""
    
    print("🧪 Testing Progress Integration Utils...")
    
    # Test 1: Analisi skip reasons
    class MockCommit:
        def __init__(self, message="", files=None):
            self.hexsha = "abc123def456"
            self.message = message
            self.stats = MockStats(files or [])
            self.parents = []
    
    class MockStats:
        def __init__(self, files):
            self.files = {f: {} for f in files}
            self.total = {'lines': len(files) * 10}
    
    # Test vari tipi di commit
    test_cases = [
        (MockCommit("Merge pull request #123", ["file1.py"]), "Merge Commit - PR"),
        (MockCommit("Update documentation", ["README.md"]), "Documentation Changes"),
        (MockCommit("Add new feature", []), "Empty Commit"),
        (MockCommit("Fix bug in parser", ["config.json", "setup.py"]), None),  # Dovrebbe passare
    ]
    
    for commit, expected in test_cases:
        reason, details = analyze_skip_reason(commit)
        if expected:
            print(f"✅ {expected}: {details}")
        else:
            print(f"ℹ️ Normal commit: {reason}")
    
    # Test 2: Classificazione errori
    test_exceptions = [
        FileNotFoundError("File not found"),
        PermissionError("Access denied"),
        MemoryError("Out of memory"),
        Exception("Unknown error")
    ]
    
    for exc in test_exceptions:
        reason, details = analyze_exception_for_skip(exc)
        severity = classify_error_severity(exc)
        print(f"🔍 {type(exc).__name__}: {reason} (severity: {severity})")
    
    print("✅ Integration utils test completed!")

def example_usage_scenarios():
    """Esempi pratici di utilizzo."""
    
    print("\n📚 USAGE EXAMPLES:")
    print("="*50)
    
    # Scenario 1: Integrazione rapida
    print("\n1. INTEGRAZIONE RAPIDA (main.py):")
    print("""
    from repo_analyzer.utils import EnhancedProgressContext, patch_analyzer_with_progress
    
    with EnhancedProgressContext("my-repo") as progress:
        analyzer = RepoAnalyzer(repo_info)
        patch_analyzer_with_progress(analyzer, progress)
        analyzer.analyze_repository(max_commits=100)
    """)
    
    # Scenario 2: Wrapper personalizzato
    print("\n2. WRAPPER PERSONALIZZATO:")
    print("""
    from repo_analyzer.utils import create_progress_aware_analyzer
    
    # Crea versione enhanced una volta
    EnhancedRepoAnalyzer = create_progress_aware_analyzer(RepoAnalyzer)
    
    # Usa normalmente
    with EnhancedProgressContext("my-repo") as progress:
        analyzer = EnhancedRepoAnalyzer(repo_info, progress_display=progress)
        analyzer.analyze_repository()
    """)
    
    # Scenario 3: Integrazione manuale
    print("\n3. INTEGRAZIONE MANUALE:")
    print("""
    class MyAnalyzer(ProgressAwareAnalyzer, RepoAnalyzer):
        def analyze_commit_custom(self, commit):
            try:
                result = self.do_analysis(commit)
                self.log_commit_success(commit.hexsha, {"smells": 5})
                return result
            except Exception as e:
                self.log_commit_error(commit.hexsha, e)
                return None
    """)
    
    # Scenario 4: Logging esistente
    print("\n4. CATTURA LOGGING ESISTENTE:")
    print("""
    import logging
    from repo_analyzer.utils import setup_progress_logging
    
    logger = logging.getLogger('my_analyzer')
    
    with EnhancedProgressContext("my-repo") as progress:
        handlers = setup_progress_logging(progress, ['my_analyzer'])
        
        # Ora tutti i log dell'analyzer appaiono nel progress
        logger.info("Starting analysis")
        logger.error("Failed to process commit abc123")
        
        cleanup_progress_logging(handlers)
    """)

def create_integration_template():
    """Crea un template per l'integrazione personalizzata."""
    
    template = '''
# ===============================================
# TEMPLATE INTEGRAZIONE PROGRESS DISPLAY
# ===============================================

from repo_analyzer.utils import EnhancedProgressContext, LogLevel

class ProgressIntegratedAnalyzer:
    """Template per analyzer con progress integrato."""
    
    def __init__(self, repo_info, progress_display=None):
        self.repo_info = repo_info
        self.progress = progress_display
        
        # I tuoi campi esistenti...
        self.skipped_commits = set()
        self.analyzed_commits = set()
    
    def analyze_repository(self, max_commits=None):
        """Metodo principale con progress tracking."""
        
        if self.progress:
            self.progress.log("Starting repository analysis", LogLevel.INFO)
        
        try:
            # Ottieni lista commit
            commits = self._get_commits(max_commits)
            
            if self.progress:
                self.progress.set_total_commits(len(commits))
            
            # Analizza ogni commit
            for i, commit in enumerate(commits):
                self._analyze_single_commit_with_progress(commit, i)
            
            if self.progress:
                self.progress.log("Repository analysis completed", LogLevel.SUCCESS)
                
        except Exception as e:
            if self.progress:
                self.progress.log_error("", "Repository Analysis", str(e))
            raise
    
    def _analyze_single_commit_with_progress(self, commit, index):
        """Analizza singolo commit con progress tracking."""
        commit_hash = commit.hexsha
        
        try:
            # Pre-check condizioni di skip
            skip_reason = self._check_skip_conditions(commit)
            if skip_reason:
                reason, details = skip_reason
                if self.progress:
                    self.progress.log_skip(commit_hash, reason, details)
                self.skipped_commits.add(commit_hash)
                return None
            
            # Log inizio analisi
            if self.progress:
                msg_preview = commit.message.split('\\n')[0][:40] if commit.message else "No message"
                self.progress.log(f"Analyzing: {msg_preview}...", LogLevel.INFO, commit_hash)
            
            # === LA TUA LOGICA DI ANALISI QUI ===
            result = self._do_actual_analysis(commit)
            
            if result:
                # Successo - log con dettagli
                if self.progress:
                    details = self._extract_analysis_details(result)
                    self.progress.log(f"Completed: {details}", LogLevel.SUCCESS, commit_hash)
                    self.progress.increment_analyzed(commit_hash)
                
                self.analyzed_commits.add(commit_hash)
                return result
            else:
                # Analisi fallita ma non errore
                if self.progress:
                    self.progress.log_skip(commit_hash, "Analysis Failed", "Analysis returned no results")
                self.skipped_commits.add(commit_hash)
                return None
                
        except Exception as e:
            # Gestione errori con classificazione
            from repo_analyzer.utils import analyze_exception_for_skip, classify_error_severity
            
            reason, details = analyze_exception_for_skip(e)
            severity = classify_error_severity(e)
            
            if self.progress:
                if severity in ['critical', 'high']:
                    self.progress.log_error(commit_hash, reason, details)
                else:
                    self.progress.log_skip(commit_hash, reason, details)
            
            self.skipped_commits.add(commit_hash)
            
            # Re-raise solo errori critici
            if severity == 'critical':
                raise
            
            return None
    
    def _check_skip_conditions(self, commit):
        """Controlla condizioni che richiedono skip del commit."""
        # Esempio: controlla se è merge commit
        if 'merge' in commit.message.lower() and 'pull request' in commit.message.lower():
            return ("Merge Commit - PR", f"Pull request merge: {commit.message[:50]}...")
        
        # Esempio: controlla file modificati
        if hasattr(commit, 'stats') and hasattr(commit.stats, 'files'):
            python_files = [f for f in commit.stats.files.keys() if f.endswith('.py')]
            if not python_files:
                total_files = len(commit.stats.files)
                return ("No Python Files", f"No Python files in {total_files} changed files")
        
        # Nessuna condizione di skip
        return None
    
    def _do_actual_analysis(self, commit):
        """LA TUA LOGICA DI ANALISI EFFETTIVA."""
        # Sostituisci con la tua implementazione
        import time
        time.sleep(0.1)  # Simula analisi
        
        # Ritorna risultato dell'analisi o None se fallisce
        return {"success": True, "metrics": {"smells": 3, "warnings": 1}}
    
    def _extract_analysis_details(self, result):
        """Estrai dettagli dall'analisi per il logging."""
        if isinstance(result, dict) and 'metrics' in result:
            metrics = result['metrics']
            details = []
            if 'smells' in metrics and metrics['smells'] > 0:
                details.append(f"{metrics['smells']} smells")
            if 'warnings' in metrics and metrics['warnings'] > 0:
                details.append(f"{metrics['warnings']} warnings")
            return ", ".join(details) if details else "clean analysis"
        return "analysis completed"
    
    def _get_commits(self, max_commits):
        """Ottieni lista commit da analizzare."""
        # Implementa la tua logica per ottenere commit
        # Questo è solo un placeholder
        class MockCommit:
            def __init__(self, i):
                self.hexsha = f"commit{i:04d}"
                self.message = f"Test commit {i}"
        
        commits = [MockCommit(i) for i in range(min(max_commits or 10, 10))]
        return commits

# ===============================================
# ESEMPIO DI USO DEL TEMPLATE
# ===============================================

def example_template_usage():
    """Esempio di come usare il template."""
    from repo_analyzer.models import RepositoryInfo
    
    repo_info = RepositoryInfo(url="https://github.com/user/repo.git")
    
    # Opzione 1: Con progress display
    with EnhancedProgressContext("template-test") as progress:
        analyzer = ProgressIntegratedAnalyzer(repo_info, progress)
        analyzer.analyze_repository(max_commits=5)
    
    # Opzione 2: Senza progress display
    analyzer = ProgressIntegratedAnalyzer(repo_info)
    analyzer.analyze_repository(max_commits=5)

if __name__ == "__main__":
    print("Testing and template generation...")
    
    try:
        test_progress_integration()
        example_usage_scenarios()
        print("\\n🎉 All tests passed!")
        print("\\n📋 Template created - copy the ProgressIntegratedAnalyzer class above")
        print("\\n🚀 Ready for integration!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
'''
    
    return template

# ===============================================
# FUNZIONI DI VALIDAZIONE
# ===============================================

def validate_progress_integration(analyzer_instance):
    """
    Valida che un analyzer abbia l'integrazione progress corretta.
    
    Args:
        analyzer_instance: Istanza analyzer da validare
    
    Returns:
        Dict con risultati validazione
    """
    validation = {
        'has_progress_display': False,
        'has_progress_methods': False,
        'has_wrapped_methods': False,
        'integration_type': 'none',
        'recommendations': []
    }
    
    # Controlla progress display
    if hasattr(analyzer_instance, '_progress_display') or hasattr(analyzer_instance, '_progress'):
        validation['has_progress_display'] = True
    
    # Controlla metodi progress
    progress_methods = ['log_progress', 'log_commit_success', 'log_commit_skip', 'log_commit_error']
    if any(hasattr(analyzer_instance, method) for method in progress_methods):
        validation['has_progress_methods'] = True
        validation['integration_type'] = 'mixin_based'
    
    # Controlla metodi wrappati
    if hasattr(analyzer_instance, '_original_methods'):
        validation['has_wrapped_methods'] = True
        validation['integration_type'] = 'patched'
    
    # Genera raccomandazioni
    if not validation['has_progress_display']:
        validation['recommendations'].append("Add progress display with set_progress_display() or patch_analyzer_with_progress()")
    
    if not validation['has_progress_methods'] and not validation['has_wrapped_methods']:
        validation['recommendations'].append("No progress integration detected. Use patch_analyzer_with_progress() for quick setup")
    
    if validation['integration_type'] == 'none':
        validation['recommendations'].append("Consider using create_progress_aware_analyzer() for full integration")
    
    return validation

def get_integration_recommendations(analyzer_class_name: str) -> List[str]:
    """
    Fornisce raccomandazioni specifiche per l'integrazione basate sul tipo di analyzer.
    
    Args:
        analyzer_class_name: Nome della classe analyzer
    
    Returns:
        Lista di raccomandazioni specifiche
    """
    recommendations = []
    
    if 'Parallel' in analyzer_class_name:
        recommendations.extend([
            "For parallel analyzers, use thread-safe progress logging",
            "Consider using simple_progress_integration() for minimal overhead",
            "Be aware that detailed per-commit logging may impact performance"
        ])
    
    if 'Multi' in analyzer_class_name:
        recommendations.extend([
            "For multi-repo analyzers, create separate progress displays per repo",
            "Use EnhancedProgressContext as context manager for each repository",
            "Consider aggregate reporting across repositories"
        ])
    
    if 'Sequential' in analyzer_class_name or analyzer_class_name == 'RepoAnalyzer':
        recommendations.extend([
            "Sequential analyzers work best with full progress integration",
            "Use create_progress_aware_analyzer() for comprehensive tracking",
            "Add detailed logging in analyze_commit_sequential method"
        ])
    
    # Raccomandazioni generali
    recommendations.extend([
        "Test integration with small repositories first",
        "Use --no-progress flag as fallback option",
        "Implement proper error classification for better skip analysis"
    ])
    
    return recommendations

# Export delle funzioni principali
__all__ = [
    # Core integration functions
    'patch_analyzer_with_progress',
    'create_progress_aware_analyzer', 
    'enhance_analyzer_class',
    'simple_progress_integration',
    'quick_patch_analyzer',
    
    # Analysis functions
    'analyze_skip_reason',
    'analyze_exception_for_skip',
    'classify_error_severity',
    
    # Logging setup
    'setup_progress_logging',
    'cleanup_progress_logging',
    'ProgressLogHandler',
    
    # Decorators and utilities
    'create_progress_wrapper',
    'monitor_analysis_performance',
    
    # Classes
    'ProgressAwareAnalyzer',
    
    # Validation and recommendations
    'validate_progress_integration',
    'get_integration_recommendations',
    
    # Testing and examples
    'test_progress_integration',
    'example_usage_scenarios',
    'create_integration_template'
]