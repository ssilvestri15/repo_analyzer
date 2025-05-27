#!/usr/bin/env python3
"""
Repository Analyzer - Strumento per analizzare metriche del codice Python commit per commit.

Questo script permette di analizzare repository Git remoti o locali, supporta l'analisi parallela
di più repository e si concentra sui file Python.
"""

import os
import sys
import argparse
import time
import logging
import multiprocessing
from typing import Optional

from .models import RepositoryInfo
from .analyzers import RepoAnalyzer, MultiRepoAnalyzer

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('repo_analyzer.log')
    ]
)
logger = logging.getLogger('repo_analyzer')

def main():
    """Funzione principale che gestisce la CLI."""
    parser = argparse.ArgumentParser(description='Analizza repository Git Python e estrae metriche di qualità del codice.')
    subparsers = parser.add_subparsers(dest='command', help='Comandi disponibili')
    
    # Parser per l'analisi di un singolo repository
    single_parser = subparsers.add_parser('single', help='Analizza un singolo repository')
    single_parser.add_argument('repo_url', help='URL della repository Git da analizzare')
    single_parser.add_argument('--output-dir', '-o', help='Directory di output per i risultati (default: nome della repository)')
    single_parser.add_argument('--temp-dir', help='Directory temporanea per la ricostruzione del progetto')
    single_parser.add_argument('--max-commits', type=int, help='Numero massimo di commit da analizzare')
    single_parser.add_argument('--resume', '-r', action='store_true', help='Riprendi l\'analisi da dove era stata interrotta')
    single_parser.add_argument('--max-workers', type=int, help='Numero massimo di worker paralleli per l\'analisi dei commit (default: auto, 1=sequenziale)')
    single_parser.add_argument('--sequential', '-s', action='store_true', help='Usa la modalità sequenziale per l\'analisi dei commit (equivale a --max-workers=1)')
    
    # Parser per l'analisi di più repository da lista
    multi_parser = subparsers.add_parser('multi', help='Analizza più repository da una lista di URL')
    multi_parser.add_argument('repo_urls', nargs='+', help='Lista di URL delle repository Git da analizzare')
    multi_parser.add_argument('--output-dir', '-o', default='repo_analysis_results', help='Directory principale per i risultati')
    multi_parser.add_argument('--max-commits', type=int, help='Numero massimo di commit da analizzare per repository')
    multi_parser.add_argument('--repo-workers', type=int, help='Numero massimo di worker paralleli per repository (default: auto)')
    multi_parser.add_argument('--commit-workers', type=int, help='Numero massimo di worker paralleli per commit (default: auto, 1=sequenziale)')
    multi_parser.add_argument('--sequential', '-s', action='store_true', help='Usa la modalità sequenziale per l\'analisi dei commit (equivale a --commit-workers=1)')
    multi_parser.add_argument('--resume', '-r', action='store_true', help='Riprendi l\'analisi da dove era stata interrotta')
    
    # Parser per l'analisi di più repository da file
    file_parser = subparsers.add_parser('file', help='Analizza più repository leggendo gli URL da un file')
    file_parser.add_argument('file_path', help='Percorso al file con gli URL (uno per riga)')
    file_parser.add_argument('--output-dir', '-o', default='repo_analysis_results', help='Directory principale per i risultati')
    file_parser.add_argument('--max-commits', type=int, help='Numero massimo di commit da analizzare per repository')
    file_parser.add_argument('--repo-workers', type=int, help='Numero massimo di worker paralleli per repository (default: auto)')
    file_parser.add_argument('--commit-workers', type=int, help='Numero massimo di worker paralleli per commit (default: auto, 1=sequenziale)')
    file_parser.add_argument('--sequential', '-s', action='store_true', help='Usa la modalità sequenziale per l\'analisi dei commit (equivale a --commit-workers=1)')
    file_parser.add_argument('--resume', '-r', action='store_true', help='Riprendi l\'analisi da dove era stata interrotta')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        if args.command == 'single':
            # Imposta la modalità sequenziale se richiesto
            if args.sequential:
                max_workers = 1
            else:
                max_workers = args.max_workers
                
            # Analisi di un singolo repository
            repo_info = RepositoryInfo(
                url=args.repo_url,
                output_dir=args.output_dir,
                temp_dir=args.temp_dir
            )
            analyzer = RepoAnalyzer(repo_info, args.resume)
            analyzer.analyze_repository(args.max_commits)
            analyzer.cleanup()
        
        elif args.command == 'multi':
            # Imposta la modalità sequenziale se richiesto
            if args.sequential:
                commit_workers = 1
            else:
                commit_workers = args.commit_workers
                
            # Analisi di più repository da lista
            multi_analyzer = MultiRepoAnalyzer(args.output_dir, args.repo_workers, commit_workers)
            multi_analyzer.analyze_from_url_list(args.repo_urls, args.max_commits, args.resume)
        
        elif args.command == 'file':
            # Imposta la modalità sequenziale se richiesto
            if args.sequential:
                commit_workers = 1
            else:
                commit_workers = args.commit_workers
                
            # Analisi di più repository da file
            multi_analyzer = MultiRepoAnalyzer(args.output_dir, args.repo_workers, commit_workers)
            multi_analyzer.analyze_from_file(args.file_path, args.max_commits, args.resume)
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"\nErrore durante l'analisi: {e}")
        logger.info("Lo stato è stato salvato. Puoi riprendere l'analisi con l'opzione --resume")
        sys.exit(1)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analisi completata in {elapsed_time:.2f} secondi.")

if __name__ == "__main__":
    main()