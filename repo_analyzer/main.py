#!/usr/bin/env python3
"""
Repository Analyzer - Strumento per analizzare metriche del codice Python commit per commit.

Questo script permette di analizzare repository Git remoti o locali, supporta l'analisi parallela
di più repository e si concentra sui file Python. Include progress display pulito in tempo reale.
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
from .analyzers.parallel_repo_analyzer import RepoAnalyzerFactory

# DISABILITA COMPLETAMENTE IL LOGGING
logging.disable(logging.CRITICAL)

def main():
    """Funzione principale che gestisce la CLI con progress display pulito."""
    parser = argparse.ArgumentParser(description='Analizza repository Git Python e estrae metriche di qualità del codice.')
    subparsers = parser.add_subparsers(dest='command', help='Comandi disponibili')
    
    # Parser per l'analisi di un singolo repository
    single_parser = subparsers.add_parser('single', help='Analizza un singolo repository')
    single_parser.add_argument('repo_url', help='URL della repository Git da analizzare')
    single_parser.add_argument('--output-dir', '-o', help='Directory di output per i risultati (default: nome della repository)')
    single_parser.add_argument('--temp-dir', help='Directory temporanea per la ricostruzione del progetto')
    single_parser.add_argument('--max-commits', type=int, help='Numero massimo di commit da analizzare')
    single_parser.add_argument('--resume', '-r', action='store_true', help='Riprendi l\'analisi da dove era stata interrotta')
    
    # Opzioni per l'analisi parallela
    single_parser.add_argument('--parallel', '-p', action='store_true', 
                              help='Usa la modalità di analisi parallela dei commit (più veloce)')
    single_parser.add_argument('--sequential', '-s', action='store_true', 
                              help='Forza la modalità sequenziale (più stabile, ma più lenta)')
    single_parser.add_argument('--max-workers', type=int, 
                              help='Numero massimo di worker paralleli (default: auto-calcolato)')
    single_parser.add_argument('--auto-mode', action='store_true',
                              help='Seleziona automaticamente la modalità ottimale basandosi sul repository')
    single_parser.add_argument('--no-progress', action='store_true',
                              help='Disabilita il progress display in tempo reale')
    
    # Parser per l'analisi di più repository da lista
    multi_parser = subparsers.add_parser('multi', help='Analizza più repository da una lista di URL')
    multi_parser.add_argument('repo_urls', nargs='+', help='Lista di URL delle repository Git da analizzare')
    multi_parser.add_argument('--output-dir', '-o', default='repo_analysis_results', help='Directory principale per i risultati')
    multi_parser.add_argument('--max-commits', type=int, help='Numero massimo di commit da analizzare per repository')
    multi_parser.add_argument('--repo-workers', type=int, help='Numero massimo di worker paralleli per repository (default: auto)')
    multi_parser.add_argument('--commit-workers', type=int, help='Numero massimo di worker paralleli per commit (default: auto, 1=sequenziale)')
    multi_parser.add_argument('--sequential', '-s', action='store_true', help='Usa la modalità sequenziale per l\'analisi dei commit')
    multi_parser.add_argument('--parallel', '-p', action='store_true', help='Usa la modalità parallela per l\'analisi dei commit')
    multi_parser.add_argument('--resume', '-r', action='store_true', help='Riprendi l\'analisi da dove era stata interrotta')
    multi_parser.add_argument('--no-progress', action='store_true', help='Disabilita il progress display per i singoli repository')
    
    # Parser per l'analisi di più repository da file
    file_parser = subparsers.add_parser('file', help='Analizza più repository leggendo gli URL da un file')
    file_parser.add_argument('file_path', help='Percorso al file con gli URL (uno per riga)')
    file_parser.add_argument('--output-dir', '-o', default='repo_analysis_results', help='Directory principale per i risultati')
    file_parser.add_argument('--max-commits', type=int, help='Numero massimo di commit da analizzare per repository')
    file_parser.add_argument('--repo-workers', type=int, help='Numero massimo di worker paralleli per repository (default: auto)')
    file_parser.add_argument('--commit-workers', type=int, help='Numero massimo di worker paralleli per commit (default: auto, 1=sequenziale)')
    file_parser.add_argument('--sequential', '-s', action='store_true', help='Usa la modalità sequenziale per l\'analisi dei commit')
    file_parser.add_argument('--parallel', '-p', action='store_true', help='Usa la modalità parallela per l\'analisi dei commit')
    file_parser.add_argument('--resume', '-r', action='store_true', help='Riprendi l\'analisi da dove era stata interrotta')
    file_parser.add_argument('--no-progress', action='store_true', help='Disabilita il progress display per i singoli repository')
    
    # Parser per informazioni di sistema
    info_parser = subparsers.add_parser('info', help='Mostra informazioni sul sistema e raccomandazioni')
    info_parser.add_argument('--repo-url', help='URL del repository per raccomandazioni specifiche')
    
    args = parser.parse_args()
    
    # Valida argomenti
    _validate_arguments(args)
    
    start_time = time.time()
    
    try:
        if args.command == 'single':
            _analyze_single_repository(args)
        
        elif args.command == 'multi':
            _analyze_multiple_repositories_from_list(args)
        
        elif args.command == 'file':
            _analyze_multiple_repositories_from_file(args)
        
        elif args.command == 'info':
            _show_system_info(args.repo_url)
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Analisi interrotta dall'utente")
        print("💾 Lo stato è stato salvato - puoi riprendere con --resume")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Errore durante l'analisi: {e}")
        print("💾 Lo stato è stato salvato - puoi riprendere con --resume")
        sys.exit(1)
    
    elapsed_time = time.time() - start_time
    print(f"🎉 Analisi completata in {elapsed_time:.2f} secondi")

def _analyze_single_repository(args):
    """Analizza un singolo repository con progress display pulito."""
    # Determina la modalità di analisi
    parallel_mode, max_workers = _determine_analysis_mode(args)
    
    # Crea repository info
    repo_info = RepositoryInfo(
        url=args.repo_url,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir
    )
    
    # Crea l'analyzer appropriato
    analyzer = RepoAnalyzerFactory.create_analyzer(
        repo_info=repo_info,
        resume=args.resume,
        parallel=parallel_mode,
        max_workers=max_workers
    )
    
    if args.no_progress:
        # Modalità senza progress display - output minimal console
        print(f"Analyzing {repo_info.name} ({'parallel' if parallel_mode else 'sequential'} mode)")
        analyzer.analyze_repository(args.max_commits)
        analyzer.cleanup()
        print("Analysis completed")
    else:
        # Modalità con progress display pulito
        try:
            from .utils.progress_display import ProgressContext
        except ImportError:
            print("Progress display not available, falling back to minimal output")
            _analyze_single_repository_fallback(args, analyzer)
            return
        
        with ProgressContext(repo_info.name) as progress:
            try:
                # Log iniziale
                mode = "Parallel" if parallel_mode else "Sequential"
                progress.log(f"🚀 {mode} analysis starting...")
                
                # Collega progress all'analyzer
                analyzer._progress = progress
                
                # Pre-scan dei commit se possibile
                try:
                    if hasattr(analyzer, '_get_ordered_commits'):
                        progress.log("🔍 Scanning repository for commits...")
                        commits = analyzer._get_ordered_commits()
                        total_commits = len(commits)
                        
                        if args.max_commits:
                            total_commits = min(total_commits, args.max_commits)
                        
                        progress.set_total_commits(total_commits)
                        progress.log(f"📊 Found {total_commits} commits to analyze")
                    else:
                        progress.log("📊 Starting analysis...")
                        
                except Exception as scan_error:
                    progress.log(f"⚠️ Could not pre-scan: {str(scan_error)[:40]}...")
                
                # Esegui l'analisi principale
                analyzer.analyze_repository(args.max_commits)
                
                # Cleanup
                progress.log("🧹 Cleaning up temporary files...")
                analyzer.cleanup()
                
                # Status finale
                if hasattr(analyzer, 'get_analysis_summary'):
                    summary = analyzer.get_analysis_summary()
                    analyzed = summary.get('total_commits_analyzed', 0)
                    skipped = summary.get('total_commits_skipped', 0)
                    progress.log(f"✅ Completed: {analyzed} analyzed, {skipped} skipped")
                else:
                    progress.log("✅ Analysis completed successfully")
                    
            except KeyboardInterrupt:
                progress.log("⏹️ Analysis interrupted by user")
                if hasattr(analyzer, 'cleanup'):
                    analyzer.cleanup()
                raise
                
            except Exception as analysis_error:
                progress.log(f"❌ Analysis failed: {str(analysis_error)[:40]}...")
                if hasattr(analyzer, 'cleanup'):
                    analyzer.cleanup()
                raise

def _analyze_single_repository_fallback(args, analyzer):
    """Fallback per analisi senza progress display."""
    print("Running analysis...")
    analyzer.analyze_repository(args.max_commits)
    analyzer.cleanup()
    print("Analysis completed")

def _analyze_multiple_repositories_from_list(args):
    """Analizza più repository da lista URL."""
    use_parallel_commits = _determine_multi_repo_mode(args)
    commit_workers = args.commit_workers if not args.sequential else 1
    
    print(f"📊 Multi-repository analysis ({'parallel' if use_parallel_commits else 'sequential'} commits)")
    
    multi_analyzer = MultiRepoAnalyzer(args.output_dir, args.repo_workers)
    
    # Prova con parametri estesi se supportati
    try:
        multi_analyzer.analyze_from_url_list(
            args.repo_urls, 
            args.max_commits, 
            args.resume,
            parallel_commits=use_parallel_commits,
            commit_workers=commit_workers
        )
    except TypeError:
        # Fallback per versioni che non supportano i nuovi parametri
        multi_analyzer.analyze_from_url_list(
            args.repo_urls, 
            args.max_commits, 
            args.resume
        )

def _analyze_multiple_repositories_from_file(args):
    """Analizza più repository da file."""
    use_parallel_commits = _determine_multi_repo_mode(args)
    commit_workers = args.commit_workers if not args.sequential else 1
    
    print(f"📊 File-based analysis ({'parallel' if use_parallel_commits else 'sequential'} commits)")
    
    multi_analyzer = MultiRepoAnalyzer(args.output_dir, args.repo_workers)
    
    # Prova con parametri estesi se supportati
    try:
        multi_analyzer.analyze_from_file(
            args.file_path, 
            args.max_commits, 
            args.resume,
            parallel_commits=use_parallel_commits,
            commit_workers=commit_workers
        )
    except TypeError:
        # Fallback per versioni che non supportano i nuovi parametri
        multi_analyzer.analyze_from_file(
            args.file_path, 
            args.max_commits, 
            args.resume
        )

def _determine_analysis_mode(args) -> tuple[bool, int]:
    """
    Determina la modalità di analisi per singolo repository.
    
    Returns:
        Tupla (parallel_mode, max_workers)
    """
    if args.sequential:
        return False, 1
    
    if args.parallel:
        return True, args.max_workers or _calculate_default_workers()
    
    if args.auto_mode:
        repo_info = RepositoryInfo(url=args.repo_url, output_dir=args.output_dir)
        try:
            parallel_recommended, suggested_workers = RepoAnalyzerFactory.get_recommended_mode(repo_info)
            return parallel_recommended, args.max_workers or suggested_workers
        except:
            # Fallback se get_recommended_mode non disponibile
            return True, args.max_workers or _calculate_default_workers()
    
    # Default: parallelo
    return True, args.max_workers or _calculate_default_workers()

def _determine_multi_repo_mode(args) -> bool:
    """Determina modalità per multi-repository analysis."""
    if args.sequential:
        return False
    if args.parallel:
        return True
    return True  # Default parallelo

def _calculate_default_workers() -> int:
    """Calcola numero default di worker basato sul sistema."""
    cpu_count = multiprocessing.cpu_count()
    return max(2, min(6, cpu_count // 2))

def _show_system_info(repo_url: Optional[str] = None):
    """Mostra informazioni sistema e raccomandazioni."""
    try:
        import psutil
    except ImportError:
        print("❌ psutil non disponibile - installare per info dettagliate sistema")
        return
    
    print("=== INFORMAZIONI SISTEMA ===")
    
    # Info CPU
    cpu_count = multiprocessing.cpu_count()
    print(f"💻 CPU: {cpu_count} core")
    
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            print(f"⚡ Frequenza: {cpu_freq.current:.0f} MHz")
    except:
        pass
    
    # Info memoria
    try:
        memory = psutil.virtual_memory()
        print(f"🧠 RAM: {memory.total / (1024**3):.1f} GB totale")
        print(f"📊 RAM disponibile: {memory.available / (1024**3):.1f} GB")
    except:
        print("🧠 Informazioni memoria non disponibili")
    
    # Info disco
    try:
        disk = psutil.disk_usage('/')
        print(f"💾 Disco: {disk.total / (1024**3):.1f} GB totale")
        print(f"📈 Spazio libero: {disk.free / (1024**3):.1f} GB")
    except:
        print("💾 Informazioni disco non disponibili")
    
    print("\n=== RACCOMANDAZIONI ===")
    
    # Raccomandazioni generali
    if cpu_count >= 8:
        print("🚀 Sistema potente - raccomandato: --parallel --max-workers 6-8")
    elif cpu_count >= 4:
        print("⚡ Sistema medio - raccomandato: --parallel --max-workers 4")
    else:
        print("🐌 Sistema limitato - raccomandato: --sequential")
    
    # Raccomandazioni specifiche per repository
    if repo_url:
        print(f"\n=== RACCOMANDAZIONI PER {repo_url} ===")
        try:
            repo_info = RepositoryInfo(url=repo_url)
            parallel_recommended, suggested_workers = RepoAnalyzerFactory.get_recommended_mode(repo_info)
            
            if parallel_recommended:
                print(f"✅ Raccomandato: --parallel --max-workers {suggested_workers}")
            else:
                print("✅ Raccomandato: --sequential")
        except Exception as e:
            print(f"⚠️ Impossibile analizzare repository: {e}")
    
    print("\n=== ESEMPI COMANDI ===")
    print("• Analisi parallela: python -m repo_analyzer single <URL> --parallel")
    print("• Analisi sequenziale: python -m repo_analyzer single <URL> --sequential") 
    print("• Modalità automatica: python -m repo_analyzer single <URL> --auto-mode")
    print("• Senza progress: python -m repo_analyzer single <URL> --no-progress")

def _validate_arguments(args):
    """Valida gli argomenti della command line."""
    # Controlla conflitti modalità
    if hasattr(args, 'parallel') and hasattr(args, 'sequential'):
        if args.parallel and args.sequential:
            print("❌ Errore: Non puoi usare sia --parallel che --sequential")
            sys.exit(1)
    
    # Valida max_workers
    if hasattr(args, 'max_workers') and args.max_workers:
        if args.max_workers < 1:
            print("❌ Errore: --max-workers deve essere almeno 1")
            sys.exit(1)
        if args.max_workers > multiprocessing.cpu_count() * 2:
            print(f"⚠️ Attenzione: --max-workers ({args.max_workers}) molto alto per questo sistema")
    
    # Valida URL repository (basic check)
    if hasattr(args, 'repo_url') and args.repo_url:
        if not (args.repo_url.startswith(('http', 'git')) or os.path.exists(args.repo_url)):
            print(f"⚠️ Attenzione: URL repository potrebbe non essere valido: {args.repo_url}")

def _check_dependencies():
    """Controlla dipendenze necessarie."""
    required_modules = ['git', 'pandas', 'lizard', 'psutil']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Moduli mancanti: {', '.join(missing_modules)}")
        print("📦 Installa con: pip install -r requirements.txt")
        sys.exit(1)

def _show_usage_examples():
    """Mostra esempi di utilizzo."""
    print("🔍 Repository Analyzer - Esempi di Utilizzo")
    print("=" * 50)
    
    examples = [
        ("Analisi base con progress display", 
         "repo_analyzer single https://github.com/user/repo.git"),
        
        ("Analisi parallela veloce", 
         "repo_analyzer single https://github.com/user/repo.git --parallel --max-workers 6"),
        
        ("Analisi sequenziale stabile", 
         "repo_analyzer single https://github.com/user/repo.git --sequential"),
        
        ("Modalità automatica", 
         "repo_analyzer single https://github.com/user/repo.git --auto-mode"),
        
        ("Analisi senza progress display", 
         "repo_analyzer single https://github.com/user/repo.git --no-progress"),
        
        ("Riprendi analisi interrotta", 
         "repo_analyzer single https://github.com/user/repo.git --resume"),
        
        ("Analisi limitata", 
         "repo_analyzer single https://github.com/user/repo.git --max-commits 100"),
        
        ("Multi-repository", 
         "repo_analyzer multi url1 url2 url3 --parallel"),
        
        ("Da file", 
         "repo_analyzer file repos.txt --max-commits 200"),
        
        ("Info sistema", 
         "repo_analyzer info --repo-url https://github.com/user/repo.git")
    ]
    
    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"\n{i:2d}. {desc}:")
        print(f"    {cmd}")
    
    print(f"\n{'='*50}")
    print("Opzioni principali:")
    print("  --parallel/-p     Analisi parallela (veloce)")
    print("  --sequential/-s   Analisi sequenziale (stabile)")  
    print("  --auto-mode       Selezione automatica modalità")
    print("  --max-workers N   Numero worker paralleli")
    print("  --resume/-r       Riprendi analisi interrotta")
    print("  --max-commits N   Limita numero commit")
    print("  --no-progress     Disabilita progress display")
    print("  --output-dir DIR  Directory output personalizzata")

if __name__ == "__main__":
    # Controlla dipendenze
    _check_dependencies()
    
    # Mostra esempi se nessun argomento
    if len(sys.argv) == 1:
        _show_usage_examples()
        sys.exit(0)
    
    main()