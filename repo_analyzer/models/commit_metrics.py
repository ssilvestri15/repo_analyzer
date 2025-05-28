"""Modello per le metriche di un commit."""

import datetime
from typing import List
from dataclasses import dataclass, field

@dataclass
class CommitMetrics:
    """Classe per memorizzare le metriche di un commit."""
    commit_hash: str
    author: str
    date: datetime.datetime
    message: str
    LOC_added: int = 0
    LOC_deleted: int = 0
    files_changed: int = 0
    commit_cyclomatic_complexity: int = 0  # Complessità ciclomatica dei file cambiati
    project_cyclomatic_complexity: int = 0  # Complessità ciclomatica totale
    author_experience: int = 0  # Numero di commit precedenti dell'autore
    time_since_last_commit: float = 0.0  # In ore
    total_smells_found: int = 0 #Numero di smell trovati
    smell_density: float = 0.0  # Smell per LOC
    num_warnings: int = 0  # Numero di warning nel codice modificato
    is_pr: bool = False  # Flag per Pull Request
    changed_files: List[str] = field(default_factory=list)
    changed_files_complexity: dict = field(default_factory=dict) # Complessità ciclomatica per file
    is_bug_fix: bool = False  # Flag per bug fix
    repo_name: str = ""  # Nome del repository
    new_feature: int = 0  # Flag for new feature commits
    bug_fixing: int = 0  # Flag for bug fixing commits
    enhancement: int = 0  # Flag for enhancement commits
    refactoring: int = 0  # Flag for refactoring commits