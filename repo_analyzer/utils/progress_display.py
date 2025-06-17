#!/usr/bin/env python3
"""
Enhanced Progress Display per Repository Analyzer
Interfaccia migliorata con tracking dettagliato di errori, warning e skip reasons
"""

import os
import sys
import time
import threading
import shutil
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from enum import Enum

class LogLevel(Enum):
    """Livelli di log con colori associati."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIP = "skip"
    DEBUG = "debug"

@dataclass
class ProgressStats:
    """Statistiche dettagliate per il progress display."""
    analyzed_commits: int = 0
    skipped_commits: int = 0
    error_commits: int = 0
    total_commits: int = 0
    current_commit: str = ""
    
    # Tracking dettagliato degli errori
    skip_reasons: Dict[str, int] = None
    error_types: Dict[str, int] = None
    warning_count: int = 0
    
    def __post_init__(self):
        if self.skip_reasons is None:
            self.skip_reasons = {}
        if self.error_types is None:
            self.error_types = {}
    
    @property
    def percentage(self) -> float:
        """Calcola la percentuale di completamento."""
        if self.total_commits == 0:
            return 0.0
        return min(100.0, (self.analyzed_commits + self.skipped_commits + self.error_commits) / self.total_commits * 100)
    
    @property
    def processed_total(self) -> int:
        """Totale commit processati (analizzati + skippati + errori)."""
        return self.analyzed_commits + self.skipped_commits + self.error_commits
    
    @property
    def success_rate(self) -> float:
        """Calcola il tasso di successo."""
        total_processed = self.processed_total
        if total_processed == 0:
            return 100.0
        return (self.analyzed_commits / total_processed) * 100

@dataclass
class LogMessage:
    """Messaggio di log con metadati."""
    timestamp: datetime
    level: LogLevel
    message: str
    commit_hash: Optional[str] = None
    details: Optional[str] = None
    
    def format_for_display(self, use_colors: bool = True) -> str:
        """Formatta il messaggio per la visualizzazione."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        
        # Icone per i diversi livelli
        icons = {
            LogLevel.INFO: "ℹ️",
            LogLevel.SUCCESS: "✅",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌", 
            LogLevel.SKIP: "⏭️",
            LogLevel.DEBUG: "🔍"
        }
        
        icon = icons.get(self.level, "•")
        commit_info = f" [{self.commit_hash[:8]}]" if self.commit_hash else ""
        
        base_msg = f"[{time_str}] {icon} {self.message}{commit_info}"
        
        # Aggiungi dettagli se presenti
        if self.details:
            base_msg += f" - {self.details}"
        
        return base_msg

class EnhancedProgressDisplay:
    """Display avanzato con tracking dettagliato di errori e warning."""
    
    def __init__(self, repo_name: str = ""):
        """Inizializza il display avanzato."""
        self.repo_name = repo_name
        self.stats = ProgressStats()
        
        # Storage dei messaggi con metadati
        self.log_messages: deque = deque(maxlen=200)  # Aumentato per più storia
        self.error_history: List[LogMessage] = []
        self.skip_history: List[LogMessage] = []
        
        # Thread control
        self.running = False
        self.display_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Terminal info
        self.terminal_width = 80
        self.terminal_height = 25
        self._update_terminal_size()
        
        # Colori avanzati
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'cyan': '\033[96m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'white': '\033[97m',
            'bg_red': '\033[101m',
            'bg_yellow': '\033[103m'
        }
        
        # Check supporto colori
        self.use_colors = (
            os.getenv('TERM') != 'dumb' and 
            hasattr(sys.stdout, 'isatty') and 
            sys.stdout.isatty()
        )
        
        # Modalità display (può essere cambiata dinamicamente)
        self.show_details = True
        self.auto_scroll = True
        
        # Statistiche di performance
        self.start_time = time.time()
        self.last_activity = time.time()
    
    def _update_terminal_size(self):
        """Aggiorna dimensioni terminale."""
        try:
            size = shutil.get_terminal_size()
            self.terminal_width = max(size.columns, 80)
            self.terminal_height = max(size.lines, 20)
        except:
            self.terminal_width = 80
            self.terminal_height = 20
    
    def _color(self, text: str, color: str) -> str:
        """Applica colore al testo."""
        if not self.use_colors:
            return text
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
    
    def start(self):
        """Avvia il display."""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Nascondi cursore e pulisci schermo
        if self.use_colors:
            print('\033[?25l', end='')  # Nascondi cursore
            print('\033[2J', end='')    # Pulisci schermo
        
        # Avvia thread display
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        # Log iniziale
        self.log("Analysis started", LogLevel.INFO)
    
    def stop(self):
        """Ferma il display."""
        if not self.running:
            return
        
        self.running = False
        
        # Log finale
        self.log("Analysis completed", LogLevel.SUCCESS)
        
        # Aspetta che il thread finisca
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1)
        
        # Mostra cursore e vai a fine schermo
        if self.use_colors:
            print('\033[?25h', end='')  # Mostra cursore
            print(f'\033[{self.terminal_height};1H')  # Vai a fine schermo
        
        # Stampa sommario finale
        self._print_final_summary()
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO, 
            commit_hash: Optional[str] = None, details: Optional[str] = None):
        """Aggiungi messaggio al log con livello e metadati."""
        log_msg = LogMessage(
            timestamp=datetime.now(),
            level=level,
            message=message,
            commit_hash=commit_hash,
            details=details
        )
        
        with self.lock:
            self.log_messages.append(log_msg)
            
            # Tracking speciale per errori e skip
            if level == LogLevel.ERROR:
                self.error_history.append(log_msg)
                self.stats.error_commits += 1
            elif level == LogLevel.SKIP:
                self.skip_history.append(log_msg)
            elif level == LogLevel.WARNING:
                self.stats.warning_count += 1
            
            self.last_activity = time.time()
    
    def log_skip(self, commit_hash: str, reason: str, details: Optional[str] = None):
        """Log specifico per commit skippati con ragione."""
        with self.lock:
            # Aggiorna statistiche skip reasons
            self.stats.skip_reasons[reason] = self.stats.skip_reasons.get(reason, 0) + 1
        
        self.log(f"Skipped commit: {reason}", LogLevel.SKIP, commit_hash, details)
    
    def log_error(self, commit_hash: str, error_type: str, error_msg: str):
        """Log specifico per errori con tipo."""
        with self.lock:
            # Aggiorna statistiche error types
            self.stats.error_types[error_type] = self.stats.error_types.get(error_type, 0) + 1
        
        self.log(f"Error: {error_type}", LogLevel.ERROR, commit_hash, error_msg)
    
    def log_warning(self, message: str, commit_hash: Optional[str] = None, details: Optional[str] = None):
        """Log specifico per warning."""
        self.log(message, LogLevel.WARNING, commit_hash, details)
    
    def log_success(self, message: str, commit_hash: Optional[str] = None):
        """Log specifico per successi."""
        self.log(message, LogLevel.SUCCESS, commit_hash)
    
    def set_total_commits(self, total: int):
        """Imposta il numero totale di commit."""
        with self.lock:
            self.stats.total_commits = total
        self.log(f"Found {total} commits to analyze", LogLevel.INFO)
    
    def update_progress(self, analyzed: int, skipped: int, errors: int = 0, current_commit: str = ""):
        """Aggiorna il progress completo."""
        with self.lock:
            self.stats.analyzed_commits = analyzed
            self.stats.skipped_commits = skipped
            self.stats.error_commits = errors
            if current_commit:
                self.stats.current_commit = current_commit[:8]
    
    def increment_analyzed(self, commit_hash: str = ""):
        """Incrementa commit analizzati."""
        with self.lock:
            self.stats.analyzed_commits += 1
            if commit_hash:
                self.stats.current_commit = commit_hash[:8]
        
        if commit_hash:
            self.log_success(f"Analyzed successfully", commit_hash[:8])
    
    def increment_skipped(self, commit_hash: str = "", reason: str = "Unknown", details: str = ""):
        """Incrementa commit skippati con ragione."""
        with self.lock:
            self.stats.skipped_commits += 1
        
        if commit_hash:
            self.log_skip(commit_hash, reason, details)
        else:
            self.log("Commit skipped", LogLevel.SKIP, details=reason)
    
    def increment_errors(self, commit_hash: str = "", error_type: str = "Unknown", error_msg: str = ""):
        """Incrementa errori con tipo."""
        # Non incrementare qui perché è già fatto in log_error
        if commit_hash:
            self.log_error(commit_hash, error_type, error_msg)
        else:
            self.log(f"Error: {error_type}", LogLevel.ERROR, details=error_msg)
    
    def _display_loop(self):
        """Loop principale del display."""
        while self.running:
            try:
                self._draw_display()
                time.sleep(0.3)  # Aggiorna ~3 volte al secondo
            except:
                pass  # Continua silenziosamente in caso di errore
    
    def _draw_display(self):
        """Disegna l'interfaccia completa avanzata."""
        if not self.use_colors:
            return
        
        # Vai a inizio schermo
        print('\033[1;1H', end='')
        
        # Aggiorna dimensioni terminale
        self._update_terminal_size()
        
        # Calcola dimensioni pannelli (60% log, 40% status)
        log_height = int(self.terminal_height * 0.6) - 1
        status_height = self.terminal_height - log_height - 1
        
        # Disegna pannello log
        self._draw_log_panel(log_height)
        
        # Disegna separatore
        self._draw_separator()
        
        # Disegna pannello status avanzato
        self._draw_status_panel(status_height)
        
        # Flush output
        sys.stdout.flush()
    
    def _draw_log_panel(self, height: int):
        """Disegna il pannello dei log avanzato."""
        # Header con info repo e tempo elapsed
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        header = f"📋 Analysis Log - {self.repo_name} | Runtime: {elapsed_str}"
        header_padding = "─" * max(0, self.terminal_width - len(header) - 4)
        print("┌─ " + self._color(header, 'cyan') + " " + header_padding + "┐")
        
        # Contenuto log con filtri
        with self.lock:
            recent_logs = list(self.log_messages)[-(height-2):]
        
        for i in range(height - 2):
            if i < len(recent_logs):
                log_msg = recent_logs[i]
                display_text = log_msg.format_for_display(self.use_colors)
                
                # Tronca se troppo lungo
                max_width = self.terminal_width - 4
                if len(display_text) > max_width:
                    display_text = display_text[:max_width - 3] + "..."
                
                # Colora in base al livello
                if log_msg.level == LogLevel.ERROR:
                    display_text = self._color(display_text, 'red')
                elif log_msg.level == LogLevel.SUCCESS:
                    display_text = self._color(display_text, 'green')
                elif log_msg.level == LogLevel.WARNING:
                    display_text = self._color(display_text, 'yellow')
                elif log_msg.level == LogLevel.SKIP:
                    display_text = self._color(display_text, 'magenta')
                elif log_msg.level == LogLevel.DEBUG:
                    display_text = self._color(display_text, 'dim')
                
                # Padding
                padding = " " * max(0, self.terminal_width - len(display_text) - 4)
                print("│ " + display_text + padding + " │")
            else:
                # Riga vuota
                padding = " " * (self.terminal_width - 4)
                print("│" + padding + "│")
        
        # Footer
        print("└" + "─" * (self.terminal_width - 2) + "┘")
    
    def _draw_separator(self):
        """Disegna il separatore tra i pannelli."""
        print("├" + "─" * (self.terminal_width - 2) + "┤")
    
    def _draw_status_panel(self, height: int):
        """Disegna il pannello status avanzato."""
        # Header
        with self.lock:
            stats = self.stats
        
        header = f"🚀 Status & Statistics | Success Rate: {stats.success_rate:.1f}%"
        header_padding = "─" * max(0, self.terminal_width - len(header) - 4)
        print("┌─ " + self._color(header, 'blue') + " " + header_padding + "┐")
        
        lines_drawn = 1  # Header
        
        # Progress bar
        if height >= 4:
            self._draw_enhanced_progress_bar(stats)
            lines_drawn += 1
        
        # Statistiche principali
        main_stats = [
            f"📊 Analyzed: {self._color(str(stats.analyzed_commits), 'green')} | "
            f"Skipped: {self._color(str(stats.skipped_commits), 'yellow')} | "
            f"Errors: {self._color(str(stats.error_commits), 'red')}",
            f"📈 Total: {stats.total_commits} | Warnings: {self._color(str(stats.warning_count), 'yellow')}",
        ]
        
        if stats.current_commit:
            main_stats.append(f"🔄 Current: {self._color(stats.current_commit, 'cyan')}")
        
        # Disegna statistiche principali
        for line in main_stats:
            if lines_drawn < height - 1:
                padding = " " * max(0, self.terminal_width - len(line) - 4)
                print("│ " + line + padding + " │")
                lines_drawn += 1
        
        # Separatore per dettagli
        if lines_drawn < height - 1 and (stats.skip_reasons or stats.error_types):
            padding = " " * (self.terminal_width - 4)
            print("│" + self._color("─" * (self.terminal_width - 4), 'dim') + "│")
            lines_drawn += 1
        
        # Top skip reasons
        if stats.skip_reasons and lines_drawn < height - 1:
            top_skips = sorted(stats.skip_reasons.items(), key=lambda x: x[1], reverse=True)[:2]
            skip_line = "⏭️ Top skips: " + ", ".join([f"{reason}({count})" for reason, count in top_skips])
            if len(skip_line) > self.terminal_width - 4:
                skip_line = skip_line[:self.terminal_width - 7] + "..."
            padding = " " * max(0, self.terminal_width - len(skip_line) - 4)
            print("│ " + self._color(skip_line, 'magenta') + padding + " │")
            lines_drawn += 1
        
        # Top error types
        if stats.error_types and lines_drawn < height - 1:
            top_errors = sorted(stats.error_types.items(), key=lambda x: x[1], reverse=True)[:2]
            error_line = "❌ Top errors: " + ", ".join([f"{error}({count})" for error, count in top_errors])
            if len(error_line) > self.terminal_width - 4:
                error_line = error_line[:self.terminal_width - 7] + "..."
            padding = " " * max(0, self.terminal_width - len(error_line) - 4)
            print("│ " + self._color(error_line, 'red') + padding + " │")
            lines_drawn += 1
        
        # Performance stats
        if lines_drawn < height - 1:
            elapsed = time.time() - self.start_time
            if stats.processed_total > 0:
                avg_time = elapsed / stats.processed_total
                perf_line = f"⚡ Avg: {avg_time:.2f}s/commit | Rate: {stats.processed_total/elapsed*60:.1f}/min"
            else:
                perf_line = f"⚡ Runtime: {elapsed:.1f}s"
            
            if len(perf_line) > self.terminal_width - 4:
                perf_line = perf_line[:self.terminal_width - 7] + "..."
            padding = " " * max(0, self.terminal_width - len(perf_line) - 4)
            print("│ " + self._color(perf_line, 'cyan') + padding + " │")
            lines_drawn += 1
        
        # Riempi righe vuote
        while lines_drawn < height - 1:
            padding = " " * (self.terminal_width - 4)
            print("│" + padding + "│")
            lines_drawn += 1
        
        # Footer
        print("└" + "─" * (self.terminal_width - 2) + "┘")
    
    def _draw_enhanced_progress_bar(self, stats: ProgressStats):
        """Disegna una barra di progress avanzata con segmenti colorati."""
        # Calcola dimensioni barra
        bar_width = self.terminal_width - 25  # Spazio per testo
        
        if stats.total_commits > 0:
            analyzed_width = int((stats.analyzed_commits / stats.total_commits) * bar_width)
            skipped_width = int((stats.skipped_commits / stats.total_commits) * bar_width)
            error_width = int((stats.error_commits / stats.total_commits) * bar_width)
        else:
            analyzed_width = skipped_width = error_width = 0
        
        # Assicurati che la somma non superi bar_width
        total_drawn = analyzed_width + skipped_width + error_width
        if total_drawn > bar_width:
            # Ridimensiona proporzionalmente
            factor = bar_width / total_drawn
            analyzed_width = int(analyzed_width * factor)
            skipped_width = int(skipped_width * factor)
            error_width = int(error_width * factor)
        
        empty_width = bar_width - (analyzed_width + skipped_width + error_width)
        
        # Caratteri barra con colori
        analyzed_bar = self._color("█" * analyzed_width, 'green')
        skipped_bar = self._color("█" * skipped_width, 'yellow') 
        error_bar = self._color("█" * error_width, 'red')
        empty_bar = "░" * empty_width
        
        # Crea barra completa
        progress_bar = analyzed_bar + skipped_bar + error_bar + empty_bar
        
        # Testo progress
        percentage_text = f"{stats.percentage:.1f}%"
        count_text = f"({stats.processed_total}/{stats.total_commits})"
        progress_text = f"{percentage_text} {count_text}"
        
        # Padding per allineare
        total_bar_line = f"│ {progress_bar} {progress_text}"
        padding_needed = self.terminal_width - len(total_bar_line) - 1
        padding = " " * max(0, padding_needed)
        
        print(total_bar_line + padding + "│")
    
    def _print_final_summary(self):
        """Stampa un sommario finale dettagliato."""
        print("\n" + "=" * 60)
        print(f"🎯 ANALYSIS SUMMARY - {self.repo_name}")
        print("=" * 60)
        
        with self.lock:
            stats = self.stats
        
        elapsed = time.time() - self.start_time
        
        # Statistiche principali
        print(f"📊 Results:")
        print(f"   ✅ Successfully analyzed: {stats.analyzed_commits}")
        print(f"   ⏭️  Skipped commits: {stats.skipped_commits}")
        print(f"   ❌ Error commits: {stats.error_commits}")
        print(f"   ⚠️  Warnings generated: {stats.warning_count}")
        print(f"   📈 Success rate: {stats.success_rate:.1f}%")
        
        # Performance
        print(f"\n⚡ Performance:")
        print(f"   🕐 Total runtime: {int(elapsed//60):02d}:{int(elapsed%60):02d}")
        if stats.processed_total > 0:
            avg_time = elapsed / stats.processed_total
            rate = stats.processed_total / elapsed * 60
            print(f"   📊 Average time per commit: {avg_time:.2f}s")
            print(f"   🚀 Processing rate: {rate:.1f} commits/minute")
        
        # Top skip reasons
        if stats.skip_reasons:
            print(f"\n⏭️  Top Skip Reasons:")
            sorted_skips = sorted(stats.skip_reasons.items(), key=lambda x: x[1], reverse=True)
            for i, (reason, count) in enumerate(sorted_skips[:5], 1):
                percentage = (count / stats.skipped_commits) * 100 if stats.skipped_commits > 0 else 0
                print(f"   {i}. {reason}: {count} ({percentage:.1f}%)")
        
        # Top error types
        if stats.error_types:
            print(f"\n❌ Top Error Types:")
            sorted_errors = sorted(stats.error_types.items(), key=lambda x: x[1], reverse=True)
            for i, (error_type, count) in enumerate(sorted_errors[:5], 1):
                percentage = (count / stats.error_commits) * 100 if stats.error_commits > 0 else 0
                print(f"   {i}. {error_type}: {count} ({percentage:.1f}%)")
        
        print("=" * 60)
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Genera un report dettagliato dell'analisi."""
        with self.lock:
            stats = self.stats
        
        elapsed = time.time() - self.start_time
        
        return {
            'summary': {
                'repository': self.repo_name,
                'total_runtime_seconds': elapsed,
                'total_commits': stats.total_commits,
                'analyzed_commits': stats.analyzed_commits,
                'skipped_commits': stats.skipped_commits,
                'error_commits': stats.error_commits,
                'warning_count': stats.warning_count,
                'success_rate_percent': stats.success_rate
            },
            'performance': {
                'average_time_per_commit': elapsed / max(stats.processed_total, 1),
                'processing_rate_per_minute': stats.processed_total / elapsed * 60 if elapsed > 0 else 0,
                'start_time': self.start_time,
                'end_time': time.time()
            },
            'skip_analysis': {
                'skip_reasons': dict(stats.skip_reasons),
                'top_skip_reason': max(stats.skip_reasons.items(), key=lambda x: x[1])[0] if stats.skip_reasons else None
            },
            'error_analysis': {
                'error_types': dict(stats.error_types),
                'top_error_type': max(stats.error_types.items(), key=lambda x: x[1])[0] if stats.error_types else None,
                'error_details': [
                    {
                        'timestamp': msg.timestamp.isoformat(),
                        'commit_hash': msg.commit_hash,
                        'message': msg.message,
                        'details': msg.details
                    }
                    for msg in self.error_history[-10:]  # Ultimi 10 errori
                ]
            }
        }

# Context manager per utilizzo semplice
class EnhancedProgressContext:
    """Context manager avanzato per il progress display."""
    
    def __init__(self, repo_name: str = ""):
        self.display = EnhancedProgressDisplay(repo_name)
    
    def __enter__(self):
        self.display.start()
        return self.display
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log finale basato sul risultato
        if exc_type:
            if exc_type == KeyboardInterrupt:
                self.display.log("Analysis interrupted by user", LogLevel.WARNING)
            else:
                self.display.log_error("", "Analysis Exception", str(exc_val))
        else:
            self.display.log("Analysis completed successfully", LogLevel.SUCCESS)
        
        time.sleep(1.5)  # Mostra messaggio finale più a lungo
        self.display.stop()

# Compatibilità con il codice esistente
class ProgressContext(EnhancedProgressContext):
    """Alias per compatibilità."""
    pass

class SimpleProgressDisplay(EnhancedProgressDisplay):
    """Alias per compatibilità."""
    pass

# Demo function avanzata
def demo_enhanced_progress():
    """Demo del progress display avanzato con errori e skip."""
    import random
    
    with EnhancedProgressContext("enhanced-demo-repo") as progress:
        total_commits = 50
        progress.set_total_commits(total_commits)
        
        # Simula diversi tipi di eventi
        skip_reasons = [
            "No Python files", "Empty commit", "Merge commit", 
            "Binary files only", "Generated code"
        ]
        
        error_types = [
            "Git checkout failed", "File permission error", 
            "Analysis timeout", "Memory limit", "Invalid encoding"
        ]
        
        for i in range(total_commits):
            commit_hash = f"abc{i:04d}ef"
            
            # Simula tempo di processing variabile
            time.sleep(0.1 + random.random() * 0.2)
            
            # Simula risultati diversi con probabilità realistiche
            rand = random.random()
            
            if rand < 0.75:  # 75% successo
                progress.increment_analyzed(commit_hash)
                
                # Occasionali warning durante l'analisi
                if random.random() < 0.3:
                    progress.log_warning("High complexity detected", commit_hash)
                
            elif rand < 0.90:  # 15% skip
                reason = random.choice(skip_reasons)
                details = f"Detected {random.randint(0, 3)} Python files"
                progress.increment_skipped(commit_hash, reason, details)
                
            else:  # 10% errore
                error_type = random.choice(error_types)
                error_msg = f"Error code: {random.randint(100, 999)}"
                progress.increment_errors(commit_hash, error_type, error_msg)
            
            # Log periodici di status
            if i > 0 and i % 10 == 0:
                progress.log(f"Checkpoint: processed {i}/{total_commits} commits", LogLevel.INFO)
        
        # Genera report finale
        report = progress.get_analysis_report()
        progress.log(f"Generated final analysis report", LogLevel.SUCCESS)

def integrate_with_analyzer_example():
    """Esempio di integrazione con gli analyzer esistenti."""
    
    # Esempio di come integrare il nuovo progress display negli analyzer esistenti
    class AnalyzerIntegration:
        def __init__(self, repo_name: str):
            self.progress = None
            self.repo_name = repo_name
        
        def analyze_with_enhanced_progress(self, commits):
            """Esempio di analisi con progress avanzato."""
            with EnhancedProgressContext(self.repo_name) as progress:
                self.progress = progress
                progress.set_total_commits(len(commits))
                
                for commit in commits:
                    try:
                        # Simula analisi commit
                        result = self._analyze_single_commit(commit)
                        
                        if result is None:
                            # Determina la ragione dello skip
                            reason = self._determine_skip_reason(commit)
                            details = self._get_skip_details(commit)
                            progress.increment_skipped(commit.hexsha, reason, details)
                        else:
                            progress.increment_analyzed(commit.hexsha)
                            
                            # Log warning se ci sono problemi minori
                            if hasattr(result, 'warnings') and result.warnings:
                                progress.log_warning(
                                    f"Analysis completed with {len(result.warnings)} warnings",
                                    commit.hexsha
                                )
                    
                    except Exception as e:
                        # Classifica il tipo di errore
                        error_type = self._classify_error(e)
                        progress.increment_errors(commit.hexsha, error_type, str(e))
                
                # Ottieni report finale
                return progress.get_analysis_report()
        
        def _analyze_single_commit(self, commit):
            """Placeholder per analisi commit."""
            import random
            if random.random() < 0.1:  # 10% chance di skip
                return None
            if random.random() < 0.05:  # 5% chance di errore
                raise Exception("Simulated analysis error")
            return {"success": True, "warnings": [] if random.random() < 0.7 else ["minor issue"]}
        
        def _determine_skip_reason(self, commit):
            """Determina la ragione dello skip basandosi sul commit."""
            # In un analyzer reale, questo analizzerebbe il commit
            reasons = [
                "No Python files changed",
                "Merge commit", 
                "Empty commit",
                "Binary files only",
                "Large files excluded"
            ]
            import random
            return random.choice(reasons)
        
        def _get_skip_details(self, commit):
            """Ottieni dettagli aggiuntivi sul perché è stato skippato."""
            # In un analyzer reale, questo fornirebbe dettagli specifici
            import random
            details = [
                f"Found {random.randint(0, 5)} changed files, {random.randint(0, 2)} Python",
                f"Commit message indicates merge: '{commit.message[:30]}...'",
                f"No substantial changes detected",
                f"Files too large (>{random.randint(10, 100)}MB)",
                f"Generated code detected in {random.randint(1, 3)} files"
            ]
            return random.choice(details)
        
        def _classify_error(self, exception):
            """Classifica il tipo di errore."""
            error_str = str(exception).lower()
            
            if "permission" in error_str or "access" in error_str:
                return "Permission Error"
            elif "timeout" in error_str or "time" in error_str:
                return "Timeout Error"
            elif "memory" in error_str or "memoryerror" in str(type(exception)):
                return "Memory Error"
            elif "git" in error_str or "checkout" in error_str:
                return "Git Operation Error"
            elif "encoding" in error_str or "unicode" in error_str:
                return "Encoding Error"
            elif "file" in error_str and ("not found" in error_str or "missing" in error_str):
                return "File Not Found"
            else:
                return "General Error"

# Funzioni di utilità per integrazione con analyzer esistenti
def wrap_analyzer_with_enhanced_progress(analyzer_class):
    """Decorator per wrappare un analyzer esistente con progress avanzato."""
    
    class WrappedAnalyzer(analyzer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._enhanced_progress = None
        
        def analyze_repository(self, max_commits=None):
            """Override del metodo analyze_repository per usare progress avanzato."""
            repo_name = getattr(self.repo_info, 'name', 'unknown-repo')
            
            with EnhancedProgressContext(repo_name) as progress:
                self._enhanced_progress = progress
                
                # Log iniziale
                progress.log("Starting repository analysis", LogLevel.INFO)
                
                try:
                    # Chiama il metodo originale
                    result = super().analyze_repository(max_commits)
                    progress.log("Repository analysis completed", LogLevel.SUCCESS)
                    return result
                    
                except Exception as e:
                    progress.log_error("", "Repository Analysis", str(e))
                    raise
        
        def _analyze_single_commit_with_progress(self, commit):
            """Metodo helper per analizzare un singolo commit con progress."""
            if not self._enhanced_progress:
                # Fallback al metodo originale se non c'è progress
                return self._analyze_single_commit_original(commit)
            
            progress = self._enhanced_progress
            commit_hash = commit.hexsha if hasattr(commit, 'hexsha') else str(commit)
            
            try:
                # Prova ad analizzare il commit
                result = self._analyze_single_commit_original(commit)
                
                if result is None:
                    # Commit skippato - determina la ragione
                    skip_reason = self._determine_skip_reason_from_context(commit)
                    progress.increment_skipped(commit_hash, skip_reason['reason'], skip_reason['details'])
                else:
                    progress.increment_analyzed(commit_hash)
                    
                    # Controlla per warning
                    if hasattr(result, 'num_warnings') and result.num_warnings > 0:
                        progress.log_warning(
                            f"Found {result.num_warnings} code warnings", 
                            commit_hash
                        )
                    
                    if hasattr(result, 'total_smells_found') and result.total_smells_found > 0:
                        progress.log_warning(
                            f"Detected {result.total_smells_found} code smells",
                            commit_hash
                        )
                
                return result
                
            except Exception as e:
                error_type = self._classify_analysis_error(e)
                progress.increment_errors(commit_hash, error_type, str(e))
                return None
        
        def _determine_skip_reason_from_context(self, commit):
            """Determina la ragione dello skip dal contesto dell'analyzer."""
            # Controlla se il commit è in skipped_commits con ragioni
            if hasattr(self, 'skipped_commits') and hasattr(commit, 'hexsha'):
                if commit.hexsha in self.skipped_commits:
                    # Prova a determinare la ragione dall'ultimo log o context
                    return {
                        'reason': 'Previously Skipped',
                        'details': 'Commit was skipped in previous analysis'
                    }
            
            # Controlla il tipo di commit
            if hasattr(commit, 'message'):
                msg = commit.message.lower()
                if 'merge' in msg and ('pull request' in msg or 'branch' in msg):
                    return {
                        'reason': 'Merge Commit',
                        'details': f"Merge commit detected: '{commit.message[:50]}...'"
                    }
            
            # Controlla i file modificati
            try:
                if hasattr(commit, 'stats') and hasattr(commit.stats, 'files'):
                    files = list(commit.stats.files.keys())
                    python_files = [f for f in files if f.endswith('.py')]
                    
                    if not python_files:
                        return {
                            'reason': 'No Python Files',
                            'details': f"No Python files in {len(files)} changed files"
                        }
                    elif len(python_files) < len(files) * 0.1:  # Meno del 10% sono Python
                        return {
                            'reason': 'Few Python Files',
                            'details': f"Only {len(python_files)}/{len(files)} files are Python"
                        }
            except:
                pass
            
            # Ragione generica
            return {
                'reason': 'Analysis Failed',
                'details': 'Could not complete commit analysis'
            }
        
        def _classify_analysis_error(self, exception):
            """Classifica errori specifici dell'analyzer."""
            error_str = str(exception).lower()
            exception_type = type(exception).__name__
            
            # Errori Git specifici
            if 'gitcommand' in exception_type.lower() or 'git' in error_str:
                if 'checkout' in error_str:
                    return 'Git Checkout Error'
                elif 'clone' in error_str:
                    return 'Git Clone Error'
                else:
                    return 'Git Operation Error'
            
            # Errori di file system
            elif 'permission' in error_str or 'access denied' in error_str:
                return 'File Permission Error'
            elif 'file not found' in error_str or 'no such file' in error_str:
                return 'File Not Found Error'
            elif 'disk' in error_str or 'space' in error_str:
                return 'Disk Space Error'
            
            # Errori di analisi del codice
            elif 'lizard' in error_str or 'complexity' in error_str:
                return 'Complexity Analysis Error'
            elif 'pylint' in error_str or 'warning' in error_str:
                return 'Code Warning Analysis Error'
            elif 'codesmile' in error_str or 'smell' in error_str:
                return 'Code Smell Analysis Error'
            
            # Errori di sistema
            elif 'memory' in error_str or 'memoryerror' in exception_type.lower():
                return 'Memory Error'
            elif 'timeout' in error_str or 'time' in error_str:
                return 'Analysis Timeout'
            
            # Errori di encoding
            elif 'encoding' in error_str or 'unicode' in error_str or 'decode' in error_str:
                return 'File Encoding Error'
            
            # Errori di formato
            elif 'json' in error_str or 'parse' in error_str:
                return 'Data Format Error'
            
            # Errore generico
            else:
                return f'Unclassified Error ({exception_type})'
        
        def _analyze_single_commit_original(self, commit):
            """Chiama il metodo originale di analisi commit."""
            # Questo dovrebbe chiamare il metodo originale della classe parent
            if hasattr(super(), 'analyze_commit_sequential'):
                return super().analyze_commit_sequential(commit)
            elif hasattr(super(), '_analyze_single_commit'):
                return super()._analyze_single_commit(commit)
            else:
                # Fallback - chiama qualsiasi metodo che sembri appropriato
                for method_name in dir(super()):
                    if 'analyze' in method_name and 'commit' in method_name:
                        method = getattr(super(), method_name)
                        if callable(method):
                            try:
                                return method(commit)
                            except TypeError:
                                continue
                return None
    
    return WrappedAnalyzer

# Esempio di utilizzo con gli analyzer esistenti
def example_integration_usage():
    """Esempio di come usare il wrapper con gli analyzer esistenti."""
    
    # Supponiamo di avere l'analyzer originale
    # from repo_analyzer.analyzers import RepoAnalyzer
    # 
    # # Wrappa l'analyzer con progress avanzato
    # EnhancedRepoAnalyzer = wrap_analyzer_with_enhanced_progress(RepoAnalyzer)
    # 
    # # Usa normalmente
    # repo_info = RepositoryInfo(url="https://github.com/user/repo.git")
    # analyzer = EnhancedRepoAnalyzer(repo_info, resume=False)
    # analyzer.analyze_repository(max_commits=100)
    
    print("Example integration code structure shown above")

if __name__ == "__main__":
    print("🚀 Enhanced Progress Display Demo")
    print("=" * 50)
    print("Features:")
    print("• Detailed error tracking and classification")
    print("• Skip reason analysis and statistics") 
    print("• Real-time performance monitoring")
    print("• Color-coded log levels")
    print("• Success rate calculation")
    print("• Final comprehensive summary")
    print("\nPress Ctrl+C to stop demo early")
    print("=" * 50)
    
    try:
        demo_enhanced_progress()
        print("\n✅ Demo completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    
    print("\n🔧 Integration example:")
    example_integration_usage()