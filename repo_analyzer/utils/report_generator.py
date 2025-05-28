"""Generazione di report per l'analisi di repository."""

import os
import datetime
import logging
import pandas as pd

logger = logging.getLogger('repo_analyzer.report_generator')

def create_single_repo_report(output_dir: str, metrics_df: pd.DataFrame, repo_name: str):
    """
    Crea un report riassuntivo per un singolo repository.
    
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
            f.write("Statistiche medie:\n")
            f.write(f"- LOC aggiunte per commit: {metrics_df['LOC_added'].mean():.2f}\n")
            f.write(f"- LOC eliminate per commit: {metrics_df['LOC_deleted'].mean():.2f}\n")
            f.write(f"- File modificati per commit: {metrics_df['files_changed'].mean():.2f}\n")
            f.write(f"- Complessità ciclomatica media dei commit: {metrics_df['commit_cyclomatic_complexity'].mean():.2f}\n")
            f.write(f"- Complessità ciclomatica media del progetto: {metrics_df['project_cyclomatic_complexity'].mean():.2f}\n")
            f.write(f"- Esperienza media degli autori: {metrics_df['author_experience'].mean():.2f} commit precedenti\n")
            f.write(f"- Tempo medio dall'ultimo commit: {metrics_df['time_since_last_commit'].mean():.2f} ore\n")
            f.write(f"- Densità di smell media: {metrics_df['smell_density'].mean():.4f}\n")
            f.write(f"- Numero medio di warning: {metrics_df['num_warnings'].mean():.2f}\n")
            f.write(f"- Smell totali trovati: {metrics_df['total_smells_found'].sum()}\n")
            f.write(f"- Smell totali trovati per commit: {metrics_df['total_smells_found'].mean():.2f}\n")
            
            f.write(f"\nTipologie di commit:\n")
            f.write(f"- Numero medio di bug fix: {metrics_df['is_bug_fix'].sum()} ({metrics_df['is_bug_fix'].mean()*100:.2f}%)\n")
            f.write(f"- Commit di nuove funzionalità: {metrics_df['new_feature'].sum()} ({metrics_df['new_feature'].mean()*100:.2f}%)\n")
            f.write(f"- Commit di enhancement: {metrics_df['enhancement'].sum()} ({metrics_df['enhancement'].mean()*100:.2f}%)\n")
            f.write(f"- Commit di refactoring: {metrics_df['refactoring'].sum()} ({metrics_df['refactoring'].mean()*100:.2f}%)\n")
            f.write(f"- Commit di bug fixing: {metrics_df['bug_fixing'].sum()} ({metrics_df['bug_fixing'].mean()*100:.2f}%)\n")
            
            f.write(f"\nInformazioni generali:\n")
            f.write(f"- Numero di Pull Request: {metrics_df['is_pr'].sum()} ({metrics_df['is_pr'].mean()*100:.2f}%)\n")
            f.write(f"- Numero di autori distinti: {len(metrics_df['author'].unique())}\n")
            f.write(f"- Repository: {metrics_df['repo_name'].iloc[0] if len(metrics_df) > 0 else 'N/A'}\n")
            
            # Statistiche aggiuntive utili
            f.write(f"\nStatistiche aggiuntive:\n")
            f.write(f"- Commit totali analizzati: {len(metrics_df)}\n")
            f.write(f"- Media LOC nette per commit: {(metrics_df['LOC_added'] - metrics_df['LOC_deleted']).mean():.2f}\n")
            f.write(f"- Rapporto LOC eliminate/aggiunte: {(metrics_df['LOC_deleted'].sum() / max(metrics_df['LOC_added'].sum(), 1)):.2f}\n")
            f.write(f"- File medi con complessità modificata: {metrics_df['changed_files_complexity'].apply(len).mean():.2f}\n")
        else:
            f.write("Nessun dato disponibile per l'analisi.\n")
    
    logger.info(f"Report riassuntivo salvato in {report_file}")

def create_comparative_report(output_dir: str, combined_df: pd.DataFrame, repo_names: list):
    """
    Crea un report comparativo per più repository.
    
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
        
        # Raggruppa per repository
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
            'is_bug_fix': 'sum'
        }).reset_index()
        
        # Rinomina le colonne per la leggibilità
        repo_stats.columns = [
            'Repository', 'Commit', 'Autori', 'LOC Aggiunte', 
            'LOC Eliminate', 'File Modificati', 'CC Media Commit',
            'CC Media Progetto', 'Esperienza Autori', 'Tempo da Ultimo Commit',
            'Densità Smell', 'Warning Medi', 'Num PR', 'Num Bug Fix'
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
            f.write(f"  - Numero di Bug Fix: {row['Num Bug Fix']}\n\n")
        
        f.write("\nCONFRONTO GLOBALE\n")
        f.write("-----------------\n\n")
        
        # Ordina i repository per vari criteri
        f.write("Repository con più commit:\n")
        top_commits = repo_stats.sort_values('Commit', ascending=False)
        for i, (_, row) in enumerate(top_commits.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} ({row['Commit']} commit)\n")
        
        f.write("\nRepository con più autori:\n")
        top_authors = repo_stats.sort_values('Autori', ascending=False)
        for i, (_, row) in enumerate(top_authors.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} ({row['Autori']} autori)\n")
        
        f.write("\nRepository con maggiore attività (LOC aggiunte+eliminate):\n")
        repo_stats['Attività'] = repo_stats['LOC Aggiunte'] + repo_stats['LOC Eliminate']
        top_activity = repo_stats.sort_values('Attività', ascending=False)
        for i, (_, row) in enumerate(top_activity.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} ({row['Attività']} LOC)\n")
        
        f.write("\nRepository con complessità maggiore:\n")
        top_complexity = repo_stats.sort_values('CC Media Progetto', ascending=False)
        for i, (_, row) in enumerate(top_complexity.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} (CC: {row['CC Media Progetto']:.2f})\n")
        
        f.write("\nRepository con più code smells:\n")
        top_smells = repo_stats.sort_values('Densità Smell', ascending=False)
        for i, (_, row) in enumerate(top_smells.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} (Densità: {row['Densità Smell']:.4f})\n")
            
        f.write("\nRepository con maggiore esperienza degli autori:\n")
        top_experience = repo_stats.sort_values('Esperienza Autori', ascending=False)
        for i, (_, row) in enumerate(top_experience.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} (Esperienza: {row['Esperienza Autori']:.2f} commit precedenti)\n")
            
        f.write("\nRepository con tempo minore tra commit:\n")
        top_frequency = repo_stats.sort_values('Tempo da Ultimo Commit')
        for i, (_, row) in enumerate(top_frequency.head(3).iterrows(), 1):
            f.write(f"  {i}. {row['Repository']} (Tempo medio: {row['Tempo da Ultimo Commit']:.2f} ore)\n")
            
        f.write("\nRepository con più bug fix:\n")
        top_bugfix = repo_stats.sort_values('Num Bug Fix', ascending=False)
        for i, (_, row) in enumerate(top_bugfix.head(3).iterrows(), 1):
            percentage = row['Num Bug Fix'] / row['Commit'] * 100 if row['Commit'] > 0 else 0
            f.write(f"  {i}. {row['Repository']} ({row['Num Bug Fix']} bug fix, {percentage:.2f}% dei commit)\n")
    
    logger.info(f"Report comparativo salvato in {report_file}")