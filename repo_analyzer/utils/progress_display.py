#!/usr/bin/env python3
"""
Simple Progress Display per Repository Analyzer
Interfaccia pulita con 2 pannelli: 70% log + 30% progress
"""

import os
import sys
import time
import threading
import shutil
from typing import Optional, List
from dataclasses import dataclass
from collections import deque
from datetime import datetime

@dataclass
class ProgressStats:
    """Statistiche semplici per il progress display."""
    analyzed_commits: int = 0
    skipped_commits: int = 0
    total_commits: int = 0
    current_commit: str = ""
    
    @property
    def percentage(self) -> float:
        """Calcola la percentuale di completamento."""
        if self.total_commits == 0:
            return 0.0
        return min(100.0, (self.analyzed_commits + self.skipped_commits) / self.total_commits * 100)
    
    @property
    def processed_total(self) -> int:
        """Totale commit processati (analizzati + skippati)."""
        return self.analyzed_commits + self.skipped_commits

class SimpleProgressDisplay:
    """Display semplice con 2 pannelli: log (70%) + progress (30%)."""
    
    def __init__(self, repo_name: str = ""):
        """Inizializza il display."""
        self.repo_name = repo_name
        self.stats = ProgressStats()
        
        # Log storage (ultimi 100 messaggi)
        self.log_messages: deque = deque(maxlen=100)
        
        # Thread control
        self.running = False
        self.display_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Terminal info
        self.terminal_width = 80
        self.terminal_height = 25
        self._update_terminal_size()
        
        # Colori
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'cyan': '\033[96m',
            'blue': '\033[94m'
        }
        
        # Check supporto colori
        self.use_colors = (
            os.getenv('TERM') != 'dumb' and 
            hasattr(sys.stdout, 'isatty') and 
            sys.stdout.isatty()
        )
    
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
        
        # Nascondi cursore e pulisci schermo
        if self.use_colors:
            print('\033[?25l', end='')  # Nascondi cursore
            print('\033[2J', end='')    # Pulisci schermo
        
        # Avvia thread display
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
    
    def stop(self):
        """Ferma il display."""
        if not self.running:
            return
        
        self.running = False
        
        # Aspetta che il thread finisca
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1)
        
        # Mostra cursore e vai a fine schermo
        if self.use_colors:
            print('\033[?25h', end='')  # Mostra cursore
            print(f'\033[{self.terminal_height};1H')  # Vai a fine schermo
        
        print()  # Nuova linea finale
    
    def log(self, message: str):
        """Aggiungi messaggio al log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        
        with self.lock:
            self.log_messages.append(formatted_msg)
    
    def set_total_commits(self, total: int):
        """Imposta il numero totale di commit."""
        with self.lock:
            self.stats.total_commits = total
    
    def update_progress(self, analyzed: int, skipped: int, current_commit: str = ""):
        """Aggiorna il progress."""
        with self.lock:
            self.stats.analyzed_commits = analyzed
            self.stats.skipped_commits = skipped
            if current_commit:
                self.stats.current_commit = current_commit[:8]
    
    def increment_analyzed(self, commit_hash: str = ""):
        """Incrementa commit analizzati."""
        with self.lock:
            self.stats.analyzed_commits += 1
            if commit_hash:
                self.stats.current_commit = commit_hash[:8]
    
    def increment_skipped(self):
        """Incrementa commit skippati."""
        with self.lock:
            self.stats.skipped_commits += 1
    
    def _display_loop(self):
        """Loop principale del display."""
        while self.running:
            try:
                self._draw_display()
                time.sleep(0.5)  # Aggiorna 2 volte al secondo
            except:
                pass  # Continua silenziosamente in caso di errore
    
    def _draw_display(self):
        """Disegna l'interfaccia completa."""
        if not self.use_colors:
            return
        
        # Vai a inizio schermo
        print('\033[1;1H', end='')
        
        # Aggiorna dimensioni terminale
        self._update_terminal_size()
        
        # Calcola dimensioni pannelli
        log_height = int(self.terminal_height * 0.7) - 1  # 70% per log
        progress_height = self.terminal_height - log_height - 1  # 30% per progress
        
        # Disegna pannello log
        self._draw_log_panel(log_height)
        
        # Disegna separatore
        self._draw_separator()
        
        # Disegna pannello progress
        self._draw_progress_panel(progress_height)
        
        # Flush output
        sys.stdout.flush()
    
    def _draw_log_panel(self, height: int):
        """Disegna il pannello dei log (70%)."""
        # Header
        header = f"📋 Analysis Log - {self.repo_name}"
        header_padding = "─" * max(0, self.terminal_width - len(header) - 4)
        print("┌─ " + self._color(header, 'cyan') + " " + header_padding + "┐")
        
        # Contenuto log
        with self.lock:
            recent_logs = list(self.log_messages)[-(height-2):]  # -2 per header e footer
        
        for i in range(height - 2):
            if i < len(recent_logs):
                log_line = recent_logs[i]
                # Tronca se troppo lungo
                if len(log_line) > self.terminal_width - 4:
                    log_line = log_line[:self.terminal_width - 7] + "..."
                
                # Colora in base al contenuto
                if "ERROR" in log_line or "failed" in log_line.lower():
                    log_line = self._color(log_line, 'red')
                elif "SUCCESS" in log_line or "completed" in log_line.lower():
                    log_line = self._color(log_line, 'green')
                elif "WARNING" in log_line or "skipped" in log_line.lower():
                    log_line = self._color(log_line, 'yellow')
                
                # Padding
                padding = " " * max(0, self.terminal_width - len(log_line) - 4)
                print("│ " + log_line + padding + " │")
            else:
                # Riga vuota
                padding = " " * (self.terminal_width - 4)
                print("│" + padding + "│")
        
        # Footer
        print("└" + "─" * (self.terminal_width - 2) + "┘")
    
    def _draw_separator(self):
        """Disegna il separatore tra i pannelli."""
        print("├" + "─" * (self.terminal_width - 2) + "┤")
    
    def _draw_progress_panel(self, height: int):
        """Disegna il pannello del progress (30%)."""
        # Header
        header = "🚀 Progress Status"
        header_padding = "─" * max(0, self.terminal_width - len(header) - 4)
        print("┌─ " + self._color(header, 'blue') + " " + header_padding + "┐")
        
        with self.lock:
            stats = self.stats
        
        # Progress bar
        if height >= 4:  # Solo se c'è spazio sufficiente
            self._draw_progress_bar(stats)
        
        # Statistiche
        stats_lines = [
            f"Analyzed: {self._color(str(stats.analyzed_commits), 'green')}",
            f"Skipped:  {self._color(str(stats.skipped_commits), 'yellow')}",
            f"Total:    {stats.total_commits}",
            f"Current:  {stats.current_commit}" if stats.current_commit else ""
        ]
        
        # Rimuovi linee vuote
        stats_lines = [line for line in stats_lines if line.strip()]
        
        # Disegna le statistiche
        lines_drawn = 1  # Header già disegnato
        for line in stats_lines:
            if lines_drawn < height - 1:  # -1 per footer
                padding = " " * max(0, self.terminal_width - len(line) - 4)
                print("│ " + line + padding + " │")
                lines_drawn += 1
        
        # Riempi righe vuote
        while lines_drawn < height - 1:
            padding = " " * (self.terminal_width - 4)
            print("│" + padding + "│")
            lines_drawn += 1
        
        # Footer
        print("└" + "─" * (self.terminal_width - 2) + "┘")
    
    def _draw_progress_bar(self, stats: ProgressStats):
        """Disegna la barra di progress."""
        # Calcola dimensioni barra
        bar_width = self.terminal_width - 20  # Spazio per testo
        
        if stats.total_commits > 0:
            filled_width = int((stats.processed_total / stats.total_commits) * bar_width)
        else:
            filled_width = 0
        
        empty_width = bar_width - filled_width
        
        # Caratteri barra
        filled_char = "█"
        empty_char = "░"
        
        # Crea barra
        progress_bar = (
            self._color(filled_char * filled_width, 'green') +
            empty_char * empty_width
        )
        
        # Testo progress
        percentage_text = f"{stats.percentage:.1f}%"
        count_text = f"({stats.processed_total}/{stats.total_commits})"
        progress_text = f"{percentage_text} {count_text}"
        
        # Padding per centrare
        total_bar_line = f"│ {progress_bar} {progress_text}"
        padding_needed = self.terminal_width - len(total_bar_line) - 1
        padding = " " * max(0, padding_needed)
        
        print(total_bar_line + padding + "│")

# Context manager per utilizzo semplice
class ProgressContext:
    """Context manager per il progress display."""
    
    def __init__(self, repo_name: str = ""):
        self.display = SimpleProgressDisplay(repo_name)
    
    def __enter__(self):
        self.display.start()
        return self.display
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log finale
        if exc_type:
            self.display.log(f"Analysis failed: {exc_val}")
        else:
            self.display.log("Analysis completed successfully")
        
        time.sleep(1)  # Mostra messaggio finale
        self.display.stop()

# Demo function
def demo_simple_progress():
    """Demo del progress display semplice."""
    import random
    
    with ProgressContext("example-repo") as progress:
        total_commits = 100
        progress.set_total_commits(total_commits)
        progress.log("Starting analysis...")
        
        for i in range(total_commits):
            # Simula tempo di processing
            time.sleep(0.05 + random.random() * 0.1)
            
            # Simula risultati diversi
            if random.random() < 0.9:  # 90% successo
                commit_hash = f"abc{i:04d}ef"
                progress.increment_analyzed(commit_hash)
                progress.log(f"Analyzed commit {commit_hash}")
            else:  # 10% skip
                progress.increment_skipped()
                progress.log(f"Skipped commit {i:04d} (error)")
            
            # Log occasionali
            if i % 20 == 0 and i > 0:
                progress.log(f"Processed {i} commits...")

if __name__ == "__main__":
    print("Running Simple Progress Display Demo...")
    print("Press Ctrl+C to stop early")
    
    try:
        demo_simple_progress()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    print("Demo completed!")