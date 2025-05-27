"""Utilità per l'analisi del codice sorgente."""

import os
import re
import subprocess
import logging
import lizard
from typing import Dict
from codesmile import CodeSmileAnalyzer
import json

newfeature_keywords = {'new', 'feature', 'add', 'creat', 'introduc', 'implement'}
bugfixing_keywords = {'fix', 'repair', 'error', 'bug', 'issue', 'exception'}
enhancement_keywords = {'updat', 'modif', 'upgrad', 'export', 'remov', 'integrat', 'support', 'enhancement', 'replac', 'includ', 'expos', 'generat', 'migrat'}
refactoring_keywords = {'renam', 'reorganiz', 'refactor', 'clean', 'polish', 'mov', 'extract', 'reorder', 're-order', 'merg'}

logger = logging.getLogger('repo_analyzer.code_analysis')

class CodeAnalysisUtils:
    """Classe di utilità per l'analisi del codice."""
    
    @staticmethod
    def is_code_file(filename: str) -> bool:
        """
        Determina se un file è un file Python.
        
        Args:
            filename: Nome del file da verificare
        
        Returns:
            True se è un file Python, False altrimenti
        """
        _, ext = os.path.splitext(filename)
        return ext.lower() == '.py'
    
    @staticmethod
    def count_lines_of_code(file_path: str) -> int:
        """
        Conta le linee di codice in un file Python.
        
        Args:
            file_path: Percorso al file
        
        Returns:
            Numero di linee di codice
        """
        if not os.path.exists(file_path) or not CodeAnalysisUtils.is_code_file(file_path):
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines()]
                # Esclude le linee vuote e i commenti
                code_lines = [line for line in lines if line and not line.startswith(('#', '//', '/*', '*', '*/')) and '/**/' not in line]
                return len(code_lines)
        except Exception as e:
            logger.error(f"Errore nel conteggio LOC per {file_path}: {e}")
            return 0
    
    @staticmethod
    def calculate_cyclomatic_complexity(file_path: str) -> int:
        """
        Calcola la complessità ciclomatica di un file utilizzando lizard.
        
        Args:
            file_path: Percorso al file
        
        Returns:
            Complessità ciclomatica
        """
        if not os.path.exists(file_path) or not CodeAnalysisUtils.is_code_file(file_path):
            return {'complexity_media': 0.0, 'complexity_total': 0.0}
        
        try:
            analysis = lizard.analyze_file(file_path)
            # Somma la complessità ciclomatica di tutte le funzioni
            result = analysis.function_list
            if not result:
                return {'complexity_media': 0.0, 'complexity_total': 0.0}
            # Calcola la complessità ciclomatica media
            sum_complexity = sum(func.cyclomatic_complexity for func in result)
            media_complexity = sum_complexity / len(result)
            return {
                'complexity_media': media_complexity,
                'complexity_total': sum_complexity,
            }
        except Exception as e:
            logger.error(f"Errore nel calcolo della complessità ciclomatica per {file_path}: {e}")
            return {'complexity_media': 0.0, 'complexity_total': 0.0}
    
    @staticmethod
    def detect_code_smells(file_path: str) -> int:
        """
        TODO: Aggangiare CodeSmilel per rilevare i code smells.
        Rileva i code smells in un file Python.
        
        Args:
            file_path: Percorso al file
        
        Returns:
            Numero di code smells rilevati
        """
        analyzer = CodeSmileAnalyzer()
        result_json = analyzer.analyze(file_path)
        result = json.loads(result_json)
        print(f"Code smells detected in {file_path}: {result}")
        return 0
    
    @staticmethod
    def detect_warnings(file_path: str) -> int:
        """
        Rileva i warning nel codice Python utilizzando pylint.
        
        Args:
            file_path: Percorso al file Python
        
        Returns:
            Numero di warning rilevati
        """
        if not os.path.exists(file_path):
            return 0
        
        warnings_count = 0
        
        try:
            result = subprocess.run(
                ['pylint', '--disable=all', '--enable=warning', file_path],
                capture_output=True,
                text=True,
                check=False
            )
            # Conta le linee con warning
            warnings_count = result.stdout.count('warning')
        except FileNotFoundError:
            logger.warning("Pylint non installato, impossibile rilevare warning")
        except Exception as e:
            logger.error(f"Errore nell'esecuzione di pylint: {e}")
        
        return warnings_count
    
    @staticmethod
    def is_from_pull_request(commit_msg: str) -> bool:
        """
        Verifica se un commit proviene da una Pull Request basandosi sul messaggio.
        
        Args:
            commit_msg: Messaggio del commit
        
        Returns:
            True se il commit proviene da una PR, False altrimenti
        """
        pr_patterns = [
            r'Merge pull request #\d+',  # GitHub
            r'Merged in .* \(pull request #\d+\)',  # Bitbucket
            r'See merge request !\\d+',  # GitLab
        ]
        
        for pattern in pr_patterns:
            if re.search(pattern, commit_msg, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def is_bug_fix(commit_msg: str) -> bool:
        """
        Verifica se un commit è un bug fix basandosi sul messaggio.
        
        Args:
            commit_msg: Messaggio del commit
        
        Returns:
            True se il commit è un bug fix, False altrimenti
        """
        bug_keywords = ['fix', 'bug', 'issue', 'error', 'crash', 'resolve', 'address', 'solve']
        message_lower = commit_msg.lower()
        return any(keyword in message_lower for keyword in bug_keywords)
    
    @staticmethod
    def _classify_commit_message(commit_message: str) -> Dict[str, int]:
        """
        Classify commit message based on keywords to determine commit goals.

        Args:
            commit_message: The commit message to classify

        Returns:
            Dictionary with classification results
        """
        msg = commit_message.lower()
        goals = {
            "new_feature": int(any(k in msg for k in newfeature_keywords)),
            "bug_fixing": int(any(k in msg for k in bugfixing_keywords)),
            "enhancement": int(any(k in msg for k in enhancement_keywords)),
            "refactoring": int(any(k in msg for k in refactoring_keywords))
        }
        return goals