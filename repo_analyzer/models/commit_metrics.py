"""Modello aggiornato per le metriche di un commit con tracking degli smell."""

import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class CommitMetrics:
    """Classe per memorizzare le metriche di un commit con informazioni sugli smell."""
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
    total_smells_found: int = 0  # Numero di smell trovati
    smell_density: float = 0.0  # Smell per LOC
    num_warnings: int = 0  # Numero di warning nel codice modificato
    is_pr: bool = False  # Flag per Pull Request
    changed_files: List[str] = field(default_factory=list)
    changed_files_complexity: dict = field(default_factory=dict)  # Complessità ciclomatica per file
    is_bug_fix: bool = False  # Flag per bug fix
    repo_name: str = ""  # Nome del repository
    new_feature: int = 0  # Flag for new feature commits
    bug_fixing: int = 0  # Flag for bug fixing commits
    enhancement: int = 0  # Flag for enhancement commits
    refactoring: int = 0  # Flag for refactoring commits
    
    # Nuovi campi per il tracking degli smell
    smells_by_file: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Smell per file
    smells_by_type: Dict[str, int] = field(default_factory=dict)  # Conteggio smell per tipo
    smell_events: List[Dict[str, Any]] = field(default_factory=list)  # Eventi sugli smell
    
    # Statistiche sugli smell
    smells_introduced: int = 0  # Smell introdotti in questo commit
    smells_removed: int = 0  # Smell rimossi in questo commit
    smells_modified: int = 0  # Smell modificati in questo commit
    smells_persisted: int = 0  # Smell persistenti (invariati)
    
    # Informazioni sui file con smell
    files_with_smells: List[str] = field(default_factory=list)  # File che hanno smell
    files_smell_introduced: List[str] = field(default_factory=list)  # File con nuovi smell
    files_smell_removed: List[str] = field(default_factory=list)  # File con smell rimossi