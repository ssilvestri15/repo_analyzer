"""Generazione di report aggiornati per l'analisi di repository con tracking degli smell."""

import os
import datetime
import json
import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger('repo_analyzer.report_generator')

def create_single_repo_report(output_dir: str, metrics_df: pd.DataFrame, repo_name: str):
    """
    Crea un report riassuntivo per un singolo repository con informazioni sugli smell.
    
    Args:
        output_dir: Directory di output
        metrics_df: DataFrame con le metriche
        repo_name: Nome del repository
    """
    report_file = os.path.join(output_dir, "report_summary.txt")
    
    with open(report_file, 'w') as f:
        f.write(f"Report di analisi per: {repo_name}\n")
        f.write(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Numero totale di commit analizzati: {len(metrics_df)}\n\n")
        
        if not metrics_df.empty:
            f.write("=== STATISTICHE GENERALI ===\n")
            f.write(f"- LOC aggiunte per commit: {metrics_df['LOC_added'].mean():.2f}\n")
            f.write(f"- LOC eliminate per commit: {metrics_df['LOC_deleted'].mean():.2f}\n")
            f.write(f"- File modificati per commit: {metrics_df['files_changed'].mean():.2f}\n")
            f.write(f"- Complessità ciclomatica media dei commit: {metrics_df['commit_cyclomatic_complexity'].mean():.2f}\n")
            f.write(f"- Complessità ciclomatica media del progetto: {metrics_df['project_cyclomatic_complexity'].mean():.2f}\n")
            f.write(f"- Esperienza media degli autori: {metrics_df['author_experience'].mean():.2f} commit precedenti\n")
            f.write(f"- Tempo medio dall'ultimo commit: {metrics_df['time_since_last_commit'].mean():.2f} ore\n")
            f.write(f"- Numero medio di warning: {metrics_df['num_warnings'].mean():.2f}\n")
            
            f.write(f"\n=== STATISTICHE DEGLI SMELL ===\n")
            f.write(f"- Smell totali trovati: {metrics_df['total_smells_found'].sum()}\n")
            f.write(f"- Smell medi per commit: {metrics_df['total_smells_found'].mean():.2f}\n")
            f.write(f"- Densità di smell media: {metrics_df['smell_density'].mean():.4f}\n")
            f.write(f"- Smell introdotti totali: {metrics_df['smells_introduced'].sum()}\n")
            f.write(f"- Smell rimossi totali: {metrics_df['smells_removed'].sum()}\n")
            f.write(f"- Smell modificati totali: {metrics_df['smells_modified'].sum()}\n")
            f.write(f"- Smell persistenti totali: {metrics_df['smells_persisted'].sum()}\n")
            
            # Analisi dei commit con smell
            commits_with_smells = metrics_df[metrics_df['total_smells_found'] > 0]
            if len(commits_with_smells) > 0:
                f.write(f"- Commit con smell: {len(commits_with_smells)} ({len(commits_with_smells)/len(metrics_df)*100:.2f}%)\n")
                f.write(f"- Media smell per commit con smell: {commits_with_smells['total_smells_found'].mean():.2f}\n")
            
            # Analisi dei file con smell
            files_with_smells_lists = metrics_df['files_with_smells'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
            )
            all_files_with_smells = set()
            for file_list in files_with_smells_lists:
                all_files_with_smells.update(file_list)
            
            f.write(f"- File unici con smell: {len(all_files_with_smells)}\n")
            
            f.write(f"\n=== TIPOLOGIE DI COMMIT ===\n")
            f.write(f"- Commit di nuove funzionalità: {metrics_df['new_feature'].sum()} ({metrics_df['new_feature'].mean()*100:.2f}%)\n")
            f.write(f"- Commit di bug fixing: {metrics_df['bug_fixing'].sum()} ({metrics_df['bug_fixing'].mean()*100:.2f}%)\n")
            f.write(f"- Commit di enhancement: {metrics_df['enhancement'].sum()} ({metrics_df['enhancement'].mean()*100:.2f}%)\n")
            f.write(f"- Commit di refactoring: {metrics_df['refactoring'].sum()} ({metrics_df['refactoring'].mean()*100:.2f}%)\n")
            f.write(f"- Bug fix (detection avanzata): {metrics_df['is_bug_fix'].sum()} ({metrics_df['is_bug_fix'].mean()*100:.2f}%)\n")
            
            f.write(f"\n=== INFORMAZIONI GENERALI ===\n")
            f.write(f"- Numero di Pull Request: {metrics_df['is_pr'].sum()} ({metrics_df['is_pr'].mean()*100:.2f}%)\n")
            f.write(f"- Numero di autori distinti: {len(metrics_df['author'].unique())}\n")
            f.write(f"- Repository: {metrics_df['repo_name'].iloc[0] if len(metrics_df) > 0 else 'N/A'}\n")
            
            f.write(f"\n=== STATISTICHE AVANZATE ===\n")
            f.write(f"- Commit totali analizzati: {len(metrics_df)}\n")
            f.write(f"- Media LOC nette per commit: {(metrics_df['LOC_added'] - metrics_df['LOC_deleted']).mean():.2f}\n")
            f.write(f"- Rapporto LOC eliminate/aggiunte: {(metrics_df['LOC_deleted'].sum() / max(metrics_df['LOC_added'].sum(), 1)):.2f}\n")
            
            # Analisi dei trend degli smell
            smell_intro_per_commit = metrics_df['smells_introduced'].mean()
            smell_removal_per_commit = metrics_df['smells_removed'].mean()
            net_smell_change = smell_intro_per_commit - smell_removal_per_commit
            
            f.write(f"\n=== TREND DEGLI SMELL ===\n")
            f.write(f"- Smell introdotti per commit: {smell_intro_per_commit:.2f}\n")
            f.write(f"- Smell rimossi per commit: {smell_removal_per_commit:.2f}\n")
            f.write(f"- Variazione netta smell per commit: {net_smell_change:.2f}\n")
            if net_smell_change > 0:
                f.write("- Trend: Aumento degli smell nel tempo\n")
            elif net_smell_change < 0:
                f.write("- Trend: Diminuzione degli smell nel tempo\n")
            else:
                f.write("- Trend: Stabile\n")
                
        else:
            f.write("Nessun dato disponibile per l'analisi.\n")
    
    logger.info(f"Report riassuntivo salvato in {report_file}")
    
    # Crea anche un report dedicato agli smell se esiste il file di evoluzione
    smell_evolution_file = os.path.join(output_dir, "smell_evolution.json")
    if os.path.exists(smell_evolution_file):
        create_smell_evolution_report(output_dir, smell_evolution_file, repo_name)

def create_smell_evolution_report(output_dir: str, smell_evolution_file: str, repo_name: str):
    """
    Crea un report dedicato all'evoluzione degli smell.
    
    Args:
        output_dir: Directory di output
        smell_evolution_file: File JSON con i dati di evoluzione degli smell
        repo_name: Nome del repository
    """
    try:
        with open(smell_evolution_file, 'r') as f:
            smell_data = json.load(f)
        
        report_file = os.path.join(output_dir, "smell_evolution_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"REPORT EVOLUZIONE SMELL - {repo_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            summary = smell_data.get('summary', {})
            f.write("=== RIASSUNTO GENERALE ===\n")
            f.write(f"Smell totali tracciati: {summary.get('total_smells_tracked', 0)}\n")
            f.write(f"Smell attivi: {summary.get('active_smells', 0)}\n")
            f.write(f"Smell rimossi: {summary.get('removed_smells', 0)}\n")
            f.write(f"File con smell: {summary.get('files_with_smells', 0)}\n")
            f.write(f"File totali tracciati: {summary.get('total_files_tracked', 0)}\n\n")
            
            # Smell per tipo
            smells_by_type = smell_data.get('smells_by_type', {})
            if smells_by_type:
                f.write("=== SMELL PER TIPO ===\n")
                for smell_type, counts in smells_by_type.items():
                    f.write(f"{smell_type}:\n")
                    f.write(f"  - Totale: {counts.get('total', 0)}\n")
                    f.write(f"  - Attivi: {counts.get('active', 0)}\n")
                    f.write(f"  - Rimossi: {counts.get('removed', 0)}\n")
                f.write("\n")
            
            # File più problematici
            file_summaries = smell_data.get('file_summaries', {})
            if file_summaries:
                problematic_files = sorted(
                    [(filename, data) for filename, data in file_summaries.items()],
                    key=lambda x: x[1].get('total_smells_ever', 0),
                    reverse=True
                )[:10]
                
                f.write("=== FILE PIÙ PROBLEMATICI (Top 10) ===\n")
                for i, (filename, data) in enumerate(problematic_files, 1):
                    f.write(f"{i}. {filename}\n")
                    f.write(f"   - Smell totali: {data.get('total_smells_ever', 0)}\n")
                    f.write(f"   - Smell attivi: {data.get('active_smells_count', 0)}\n")
                    f.write(f"   - Smell rimossi: {data.get('inactive_smells_count', 0)}\n")
                    f.write(f"   - Modifiche totali: {data.get('total_modifications', 0)}\n")
                f.write("\n")
            
            # Smell più longevi
            smell_histories = smell_data.get('smell_histories', [])
            long_living_smells = [
                smell for smell in smell_histories 
                if smell.get('lifespan_days') and smell.get('lifespan_days') > 0
            ]
            long_living_smells.sort(key=lambda x: x.get('lifespan_days', 0), reverse=True)
            
            if long_living_smells:
                f.write("=== SMELL PIÙ LONGEVI (Top 10) ===\n")
                for i, smell in enumerate(long_living_smells[:10], 1):
                    f.write(f"{i}. {smell.get('smell_name', 'N/A')} in {smell.get('filename', 'N/A')}\n")
                    f.write(f"   - Durata: {smell.get('lifespan_days', 0)} giorni\n")
                    f.write(f"   - Funzione: {smell.get('function_name', 'N/A')}\n")
                    f.write(f"   - Stato: {'Attivo' if smell.get('is_active') else 'Rimosso'}\n")
                    f.write(f"   - Modifiche: {smell.get('times_modified', 0)}\n")
                    f.write(f"   - Commit coinvolti: {smell.get('total_commits', 0)}\n")
                f.write("\n")
            
            # Smell più modificati
            frequently_modified = [
                smell for smell in smell_histories 
                if smell.get('times_modified', 0) > 0
            ]
            frequently_modified.sort(key=lambda x: x.get('times_modified', 0), reverse=True)
            
            if frequently_modified:
                f.write("=== SMELL PIÙ FREQUENTEMENTE MODIFICATI (Top 10) ===\n")
                for i, smell in enumerate(frequently_modified[:10], 1):
                    f.write(f"{i}. {smell.get('smell_name', 'N/A')} in {smell.get('filename', 'N/A')}\n")
                    f.write(f"   - Modifiche: {smell.get('times_modified', 0)}\n")
                    f.write(f"   - Funzione: {smell.get('function_name', 'N/A')}\n")
                    f.write(f"   - Stato: {'Attivo' if smell.get('is_active') else 'Rimosso'}\n")
                    if smell.get('lifespan_days'):
                        f.write(f"   - Durata: {smell.get('lifespan_days')} giorni\n")
                f.write("\n")
            
            # Statistiche per funzione
            functions_stats = {}
            for smell in smell_histories:
                func_name = smell.get('function_name', 'N/A')
                if func_name not in functions_stats:
                    functions_stats[func_name] = {
                        'total_smells': 0,
                        'active_smells': 0,
                        'files': set()
                    }
                functions_stats[func_name]['total_smells'] += 1
                if smell.get('is_active'):
                    functions_stats[func_name]['active_smells'] += 1
                functions_stats[func_name]['files'].add(smell.get('filename', 'N/A'))
            
            if functions_stats:
                problematic_functions = sorted(
                    [(func, stats) for func, stats in functions_stats.items()],
                    key=lambda x: x[1]['total_smells'],
                    reverse=True
                )[:10]
                
                f.write("=== FUNZIONI PIÙ PROBLEMATICHE (Top 10) ===\n")
                for i, (func_name, stats) in enumerate(problematic_functions, 1):
                    f.write(f"{i}. {func_name}\n")
                    f.write(f"   - Smell totali: {stats['total_smells']}\n")
                    f.write(f"   - Smell attivi: {stats['active_smells']}\n")
                    f.write(f"   - File coinvolti: {len(stats['files'])}\n")
                f.write("\n")
        
        logger.info(f"Report evoluzione smell salvato in {report_file}")
        
    except Exception as e:
        logger.error(f"Errore nella creazione del report di evoluzione smell: {e}")

def create_comparative_report(output_dir: str, combined_df: pd.DataFrame, repo_names: list):
    """
    Crea un report comparativo per più repository con informazioni sugli smell.
    
    Args:
        output_dir: Directory di output
        combined_df: DataFrame con le metriche di tutti i repository
        repo_names: Lista dei nomi dei repository analizzati
    """
    report_file = os.path.join(output_dir, "comparative_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("REPORT COMPARATIVO DEI REPOSITORY\n")
        f.write("================================\n\n")
        f.write(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Numero totale di repository analizzati: {len(repo_names)}\n")
        f.write(f"Numero totale di commit analizzati: {len(combined_df)}\n\n")
        
        f.write("STATISTICHE PER REPOSITORY\n")
        f.write("-------------------------\n\n")
        
        # Raggruppa per repository con statistiche sugli smell
        repo_stats = combined_df.groupby('repo_name').agg({
            'commit_hash': 'count',
            'author': lambda x: len(set(x)),
            'LOC_added': 'sum',
            'LOC_deleted': 'sum',
            'files_changed': 'sum',
            'commit_cyclomatic_complexity': 'mean',
            'project_cyclomatic_complexity': 'mean',
            'author_experience': 'mean',
            'time_since_last_commit': 'mean',
            'smell_density': 'mean',
            'num_warnings': 'mean',
            'is_pr': 'sum',
            'is_bug_fix': 'sum',
            'total_smells_found': 'sum',
            'smells_introduced': 'sum',
            'smells_removed': 'sum',
            'smells_modified': 'sum',
            'smells_persisted': 'sum'
        }).reset_index()
        
        # Rinomina le colonne
        repo_stats.columns = [
            'Repository', 'Commit', 'Autori', 'LOC Aggiunte', 
            'LOC Eliminate', 'File Modificati', 'CC Media Commit',
            'CC Media Progetto', 'Esperienza Autori', 'Tempo da Ultimo Commit',
            'Densità Smell', 'Warning Medi', 'Num PR', 'Num Bug Fix',
            'Smell Totali', 'Smell Introdotti', 'Smell Rimossi', 
            'Smell Modificati', 'Smell Persistenti'
        ]
        
        # Scrivi le statistiche per ogni repository
        for _, row in repo_stats.iterrows():
            f.write(f"Repository: {row['Repository']}\n")
            f.write(f"  - Commit: {row['Commit']}\n")
            f.write(f"  - Autori distinti: {row['Autori']}\n")
            f.write(f"  - Linee aggiunte: {row['LOC Aggiunte']}\n")
            f.write(f"  - Linee eliminate: {row['LOC Eliminate']}\n")
            f.write(f"  - File modificati: {row['File Modificati']}\n")
            f.write(f"  - Complessità ciclomatica media dei commit: {row['CC Media Commit']:.2f}\n")
            f.write(f"  - Complessità ciclomatica media del progetto: {row['CC Media Progetto']:.2f}\n")
            f.write(f"  - Esperienza media autori: {row['Esperienza Autori']:.2f} commit precedenti\n")
            f.write(f"  - Tempo medio dall'ultimo commit: {row['Tempo da Ultimo Commit']:.2f} ore\n")
            f.write(f"  - Densità di smell media: {row['Densità Smell']:.4f}\n")
            f.write(f"  - Warning medi per commit: {row['Warning Medi']:.2f}\n")
            f.write(f"  - Numero di Pull Request: {row['Num PR']}\n")
            f.write(f"  - Numero di Bug Fix: {row['Num Bug Fix']}\n")
            f.write(f"  - Smell totali trovati: {row['Smell Totali']}\n")
            f.write(f"  - Smell introdotti: {row['Smell Introdotti']}\n")
            f.write(f"  - Smell rimossi: {row['Smell Rimossi']}\n")
            f.write(f"  - Smell modificati: {row['Smell Modificati']}\n")
            f.write(f"  - Smell persistenti: {row['Smell Persistenti']}\n")
            
            # Calcola il bilancio degli smell
            net_smell_change = row['Smell Introdotti'] - row['Smell Rimossi']
            if net_smell_change > 0:
                f.write(f"  - Bilancio smell: +{net_smell_change} (peggioramento)\n")
            elif net_smell_change < 0:
                f.write(f"  - Bilancio smell: {net_smell_change} (miglioramento)\n")
            else:
                f.write(f"  - Bilancio smell: 0 (stabile)\n")
            f.write("\n")
        
        f.write("\nCONFRONTO GLOBALE\n")
        f.write("-----------------\n\n")
        
        # Repository con più commit
        f.write("Repository con più commit:\n")
        top_commits = repo_stats.sort_values('Commit', ascending=False)
        for i, (_, row) in enumerate(top_commits.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} ({row['Commit']} commit)\n")
        
        # Repository con più autori
        f.write("\nRepository con più autori:\n")
        top_authors = repo_stats.sort_values('Autori', ascending=False)
        for i, (_, row) in enumerate(top_authors.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} ({row['Autori']} autori)\n")
        
        # Repository con maggiore attività
        f.write("\nRepository con maggiore attività (LOC aggiunte+eliminate):\n")
        repo_stats['Attività'] = repo_stats['LOC Aggiunte'] + repo_stats['LOC Eliminate']
        top_activity = repo_stats.sort_values('Attività', ascending=False)
        for i, (_, row) in enumerate(top_activity.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} ({row['Attività']} LOC)\n")
        
        # Repository con complessità maggiore
        f.write("\nRepository con complessità maggiore:\n")
        top_complexity = repo_stats.sort_values('CC Media Progetto', ascending=False)
        for i, (_, row) in enumerate(top_complexity.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} (CC: {row['CC Media Progetto']:.2f})\n")
        
        # Repository con più smell
        f.write("\nRepository con più smell totali:\n")
        top_smells = repo_stats.sort_values('Smell Totali', ascending=False)
        for i, (_, row) in enumerate(top_smells.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} ({row['Smell Totali']} smell)\n")
        
        # Repository con densità di smell maggiore
        f.write("\nRepository con densità di smell maggiore:\n")
        top_smell_density = repo_stats.sort_values('Densità Smell', ascending=False)
        for i, (_, row) in enumerate(top_smell_density.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} (Densità: {row['Densità Smell']:.4f})\n")
        
        # Repository con miglior bilancio degli smell (più smell rimossi che introdotti)
        f.write("\nRepository con miglior gestione degli smell:\n")
        repo_stats['Bilancio Smell'] = repo_stats['Smell Rimossi'] - repo_stats['Smell Introdotti']
        top_smell_management = repo_stats.sort_values('Bilancio Smell', ascending=False)
        for i, (_, row) in enumerate(top_smell_management.head(3).iterrows(), 1):
            balance = row['Bilancio Smell']
            if balance > 0:
                status = f"(+{balance} smell rimossi)"
            elif balance < 0:
                status = f"({-balance} smell introdotti)"
            else:
                status = "(bilanciato)"
            f.write(f"  {i}. {row['Repository']} {status}\n")
        
        # Repository con maggiore esperienza degli autori
        f.write("\nRepository con maggiore esperienza degli autori:\n")
        top_experience = repo_stats.sort_values('Esperienza Autori', ascending=False)
        for i, (_, row) in enumerate(top_experience.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} (Esperienza: {row['Esperienza Autori']:.2f} commit precedenti)\n")
        
        # Repository con tempo minore tra commit
        f.write("\nRepository con sviluppo più intenso (minor tempo tra commit):\n")
        top_frequency = repo_stats.sort_values('Tempo da Ultimo Commit')
        for i, (_, row) in enumerate(top_frequency.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} (Tempo medio: {row['Tempo da Ultimo Commit']:.2f} ore)\n")
        
        # Repository con più bug fix
        f.write("\nRepository con più bug fix:\n")
        top_bugfix = repo_stats.sort_values('Num Bug Fix', ascending=False)
        for i, (_, row) in enumerate(top_bugfix.head(3).iterrows(), 1):
            percentage = row['Num Bug Fix'] / row['Commit'] * 100 if row['Commit'] > 0 else 0
            f.write(f"  {i}. {row['Repository']} ({row['Num Bug Fix']} bug fix, {percentage:.2f}% dei commit)\n")
        
        # Analisi correlazioni smell-bug fix
        f.write("\n=== ANALISI CORRELAZIONI ===\n")
        f.write("Correlazione tra smell e bug fix:\n")
        for _, row in repo_stats.iterrows():
            if row['Commit'] > 0:
                smell_rate = row['Smell Totali'] / row['Commit']
                bug_rate = row['Num Bug Fix'] / row['Commit']
                f.write(f"  {row['Repository']}: {smell_rate:.2f} smell/commit, {bug_rate:.2f} bug/commit\n")
        
        # Trend globali
        f.write("\n=== TREND GLOBALI ===\n")
        total_smell_introduced = repo_stats['Smell Introdotti'].sum()
        total_smell_removed = repo_stats['Smell Rimossi'].sum()
        global_balance = total_smell_removed - total_smell_introduced
        
        f.write(f"Smell introdotti totali: {total_smell_introduced}\n")
        f.write(f"Smell rimossi totali: {total_smell_removed}\n")
        f.write(f"Bilancio globale: {global_balance}\n")
        
        if global_balance > 0:
            f.write("Trend globale: Miglioramento della qualità del codice\n")
        elif global_balance < 0:
            f.write("Trend globale: Peggioramento della qualità del codice\n")
        else:
            f.write("Trend globale: Stabile\n")
        
        # Medie globali
        f.write(f"\nMedia smell per commit: {combined_df['total_smells_found'].mean():.2f}\n")
        f.write(f"Media densità smell: {combined_df['smell_density'].mean():.4f}\n")
        f.write(f"Percentuale commit con smell: {(combined_df['total_smells_found'] > 0).mean() * 100:.2f}%\n")
    
    logger.info(f"Report comparativo salvato in {report_file}")

def create_smell_csv_export(output_dir: str, smell_evolution_file: str):
    """
    Crea un export CSV dettagliato dell'evoluzione degli smell.
    
    Args:
        output_dir: Directory di output
        smell_evolution_file: File JSON con i dati di evoluzione degli smell
    """
    try:
        with open(smell_evolution_file, 'r') as f:
            smell_data = json.load(f)
        
        # Estrai i dati degli smell per il CSV
        csv_data = []
        smell_histories = smell_data.get('smell_histories', [])
        
        for smell in smell_histories:
            csv_data.append({
                'smell_id': smell.get('smell_id', ''),
                'filename': smell.get('filename', ''),
                'function_name': smell.get('function_name', ''),
                'smell_name': smell.get('smell_name', ''),
                'description': smell.get('description', ''),
                'is_active': smell.get('is_active', False),
                'current_line': smell.get('current_line', ''),
                'first_commit': smell.get('first_commit', ''),
                'first_date': smell.get('first_date', ''),
                'last_commit': smell.get('last_commit', ''),
                'last_date': smell.get('last_date', ''),
                'lifespan_days': smell.get('lifespan_days', ''),
                'times_modified': smell.get('times_modified', 0),
                'times_file_changed': smell.get('times_file_changed', 0),
                'total_commits': smell.get('total_commits', 0),
                'total_events': smell.get('total_events', 0)
            })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(output_dir, "smell_evolution.csv")
            df.to_csv(csv_file, index=False)
            logger.info(f"Export CSV smell evolution salvato in {csv_file}")
        
        # Crea anche un CSV con i riassunti dei file
        file_summaries = smell_data.get('file_summaries', {})
        if file_summaries:
            file_data = []
            for filename, summary in file_summaries.items():
                file_data.append({
                    'filename': filename,
                    'active_smells_count': summary.get('active_smells_count', 0),
                    'inactive_smells_count': summary.get('inactive_smells_count', 0),
                    'total_smells_ever': summary.get('total_smells_ever', 0),
                    'total_modifications': summary.get('total_modifications', 0),
                    'total_commits_with_changes': summary.get('total_commits_with_changes', 0)
                })
            
            if file_data:
                df_files = pd.DataFrame(file_data)
                file_csv = os.path.join(output_dir, "file_smell_summary.csv")
                df_files.to_csv(file_csv, index=False)
                logger.info(f"Export CSV riassunto file salvato in {file_csv}")
                
    except Exception as e:
        logger.error(f"Errore nella creazione dell'export CSV: {e}")

def create_comprehensive_analysis_report(output_dir: str, repo_name: str):
    """
    Crea un report comprensivo che combina tutte le analisi disponibili.
    
    Args:
        output_dir: Directory di output
        repo_name: Nome del repository
    """
    comprehensive_report_file = os.path.join(output_dir, "comprehensive_analysis.txt")
    
    try:
        with open(comprehensive_report_file, 'w') as f:
            f.write(f"ANALISI COMPRENSIVA - {repo_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Leggi i dati esistenti
            metrics_file = os.path.join(output_dir, "commit_metrics.csv")
            smell_evolution_file = os.path.join(output_dir, "smell_evolution.json")
            freq_file = os.path.join(output_dir, "file_frequencies.json")
            
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                f.write("=== PANORAMICA GENERALE ===\n")
                f.write(f"Periodo analizzato: {df['date'].min()} - {df['date'].max()}\n")
                f.write(f"Commit analizzati: {len(df)}\n")
                f.write(f"Autori coinvolti: {len(df['author'].unique())}\n")
                f.write(f"Attività totale: {df['LOC_added'].sum() + df['LOC_deleted'].sum()} LOC\n\n")
            
            if os.path.exists(smell_evolution_file):
                with open(smell_evolution_file, 'r') as smell_f:
                    smell_data = json.load(smell_f)
                
                f.write("=== SALUTE DEL CODICE ===\n")
                summary = smell_data.get('summary', {})
                f.write(f"Smell attivi: {summary.get('active_smells', 0)}\n")
                f.write(f"Smell risolti: {summary.get('removed_smells', 0)}\n")
                f.write(f"File problematici: {summary.get('files_with_smells', 0)}\n")
                
                # Calcola alcuni KPI
                if summary.get('total_smells_tracked', 0) > 0:
                    resolution_rate = (summary.get('removed_smells', 0) / summary.get('total_smells_tracked', 1)) * 100
                    f.write(f"Tasso di risoluzione smell: {resolution_rate:.2f}%\n")
                f.write("\n")
            
            if os.path.exists(freq_file):
                with open(freq_file, 'r') as freq_f:
                    freq_data = json.load(freq_f)
                
                f.write("=== FILE PIÙ MODIFICATI ===\n")
                edited_files = freq_data.get('edited_files', {})
                if edited_files:
                    top_edited = sorted(edited_files.items(), key=lambda x: x[1], reverse=True)[:5]
                    for i, (filename, count) in enumerate(top_edited, 1):
                        f.write(f"{i}. {filename} ({count} modifiche)\n")
                f.write("\n")
                
                f.write("=== FILE CON PIÙ BUG ===\n")
                bugged_files = freq_data.get('bugged_files', {})
                if bugged_files:
                    top_bugged = sorted(bugged_files.items(), key=lambda x: x[1], reverse=True)[:5]
                    for i, (filename, count) in enumerate(top_bugged, 1):
                        f.write(f"{i}. {filename} ({count} bug fix)\n")
                f.write("\n")
            
            # Raccomandazioni
            f.write("=== RACCOMANDAZIONI ===\n")
            f.write("1. Concentrare gli sforzi di refactoring sui file più problematici\n")
            f.write("2. Implementare code review più rigorose per i file con più bug\n")
            f.write("3. Considerare la suddivisione di file con alta complessità\n")
            f.write("4. Monitorare l'evoluzione degli smell nel tempo\n")
            f.write("5. Investire in test automatizzati per i file più critici\n")
            
        logger.info(f"Report comprensivo salvato in {comprehensive_report_file}")
        
    except Exception as e:
        logger.error(f"Errore nella creazione del report comprensivo: {e}")

def export_all_reports(output_dir: str, repo_name: str):
    """
    Esporta tutti i report disponibili per un repository.
    
    Args:
        output_dir: Directory di output
        repo_name: Nome del repository
    """
    try:
        # Crea il report comprensivo
        create_comprehensive_analysis_report(output_dir, repo_name)
        
        # Esporta i CSV degli smell se disponibili
        smell_evolution_file = os.path.join(output_dir, "smell_evolution.json")
        if os.path.exists(smell_evolution_file):
            create_smell_csv_export(output_dir, smell_evolution_file)
        
        logger.info(f"Tutti i report per {repo_name} sono stati esportati in {output_dir}")
        
    except Exception as e:
        logger.error(f"Errore nell'esportazione di tutti i report: {e}")