"""Modulo completo per il tracking dell'evoluzione degli smell."""

import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger('repo_analyzer.smell_tracker')

class SmellEventType(Enum):
    """Tipi di eventi che possono accadere agli smell."""
    SMELL_INTRODUCED = "introduced"  # Smell introdotto per la prima volta
    SMELL_MODIFIED = "modified"      # Smell modificato (linea cambiata ma smell rimane)
    SMELL_REMOVED = "removed"        # Smell rimosso
    SMELL_PERSISTED = "persisted"    # Smell rimane invariato
    FILE_MODIFIED = "file_modified"  # File modificato ma smell non cambiato

@dataclass
class SmellDetection:
    """Rappresenta una singola detection di smell."""
    filename: str
    function_name: str
    smell_name: str
    line: int
    description: str
    additional_info: str
    
    def get_smell_id(self) -> str:
        """Crea un ID univoco per lo smell basato su file, funzione e tipo."""
        return f"{self.filename}:{self.function_name}:{self.smell_name}"
    
    def get_location_id(self) -> str:
        """Crea un ID univoco per la posizione dello smell."""
        return f"{self.filename}:{self.line}:{self.smell_name}"

@dataclass
class SmellEvent:
    """Rappresenta un evento nella storia di uno smell."""
    commit_hash: str
    commit_date: datetime.datetime
    event_type: SmellEventType
    smell_detection: Optional[SmellDetection] = None
    previous_line: Optional[int] = None
    current_line: Optional[int] = None
    notes: str = ""

@dataclass
class SmellHistory:
    """Traccia la storia completa di uno smell."""
    smell_id: str
    filename: str
    function_name: str
    smell_name: str
    description: str
    
    # Prima e ultima occorrenza
    first_introduced_commit: str = ""
    first_introduced_date: Optional[datetime.datetime] = None
    last_seen_commit: str = ""
    last_seen_date: Optional[datetime.datetime] = None
    
    # Stato corrente
    is_active: bool = True
    current_line: Optional[int] = None
    
    # Storia degli eventi
    events: List[SmellEvent] = field(default_factory=list)
    
    # Statistiche
    times_modified: int = 0
    times_file_changed: int = 0
    commits_with_smell: Set[str] = field(default_factory=set)
    
    def add_event(self, event: SmellEvent):
        """Aggiunge un evento alla storia dello smell."""
        self.events.append(event)
        self.commits_with_smell.add(event.commit_hash)
        
        # Aggiorna i dati dell'ultima occorrenza
        self.last_seen_commit = event.commit_hash
        self.last_seen_date = event.commit_date
        
        # Aggiorna il primo commit se non impostato
        if not self.first_introduced_commit:
            self.first_introduced_commit = event.commit_hash
            self.first_introduced_date = event.commit_date
        
        # Aggiorna statistiche
        if event.event_type == SmellEventType.SMELL_MODIFIED:
            self.times_modified += 1
        elif event.event_type == SmellEventType.FILE_MODIFIED:
            self.times_file_changed += 1
        elif event.event_type == SmellEventType.SMELL_REMOVED:
            self.is_active = False
        
        # Aggiorna la linea corrente se disponibile
        if event.current_line is not None:
            self.current_line = event.current_line
    
    def get_lifespan_days(self) -> Optional[int]:
        """Calcola la durata di vita dello smell in giorni."""
        if self.first_introduced_date and self.last_seen_date:
            return (self.last_seen_date - self.first_introduced_date).days
        return None
    
    def get_summary(self) -> Dict:
        """Restituisce un riassunto della storia dello smell."""
        return {
            'smell_id': self.smell_id,
            'filename': self.filename,
            'function_name': self.function_name,
            'smell_name': self.smell_name,
            'description': self.description,
            'is_active': self.is_active,
            'current_line': self.current_line,
            'first_commit': self.first_introduced_commit,
            'first_date': self.first_introduced_date.isoformat() if self.first_introduced_date else None,
            'last_commit': self.last_seen_commit,
            'last_date': self.last_seen_date.isoformat() if self.last_seen_date else None,
            'lifespan_days': self.get_lifespan_days(),
            'times_modified': self.times_modified,
            'times_file_changed': self.times_file_changed,
            'total_commits': len(self.commits_with_smell),
            'total_events': len(self.events)
        }

@dataclass
class FileSmellTracker:
    """Traccia tutti gli smell in un file specifico."""
    filename: str
    
    # Smell attivi e inattivi
    active_smells: Dict[str, SmellHistory] = field(default_factory=dict)
    inactive_smells: Dict[str, SmellHistory] = field(default_factory=dict)
    
    # Statistiche del file
    total_modifications: int = 0
    commits_with_changes: Set[str] = field(default_factory=set)
    
    def add_smell_detection(self, commit_hash: str, commit_date: datetime.datetime, 
                           detection: SmellDetection, previous_detections: Optional[List[SmellDetection]] = None):
        """Aggiunge una detection di smell e determina il tipo di evento."""
        smell_id = detection.get_smell_id()
        
        # Determina il tipo di evento
        event_type = self._determine_event_type(detection, previous_detections)
        
        # Trova la linea precedente se lo smell esisteva già
        previous_line = None
        if smell_id in self.active_smells:
            previous_line = self.active_smells[smell_id].current_line
        elif smell_id in self.inactive_smells:
            previous_line = self.inactive_smells[smell_id].current_line
        
        # Crea l'evento
        event = SmellEvent(
            commit_hash=commit_hash,
            commit_date=commit_date,
            event_type=event_type,
            smell_detection=detection,
            previous_line=previous_line,
            current_line=detection.line
        )
        
        # Trova o crea la storia dello smell
        if smell_id in self.active_smells:
            smell_history = self.active_smells[smell_id]
        elif smell_id in self.inactive_smells:
            smell_history = self.inactive_smells[smell_id]
            # Riattiva lo smell se era stato rimosso
            if event_type != SmellEventType.SMELL_REMOVED:
                self.active_smells[smell_id] = smell_history
                del self.inactive_smells[smell_id]
                smell_history.is_active = True
        else:
            # Nuovo smell
            smell_history = SmellHistory(
                smell_id=smell_id,
                filename=detection.filename,
                function_name=detection.function_name,
                smell_name=detection.smell_name,
                description=detection.description,
                current_line=detection.line
            )
            self.active_smells[smell_id] = smell_history
        
        # Aggiunge l'evento alla storia
        smell_history.add_event(event)
        
        # Se lo smell è stato rimosso, spostalo negli inattivi
        if event_type == SmellEventType.SMELL_REMOVED and smell_id in self.active_smells:
            self.inactive_smells[smell_id] = self.active_smells[smell_id]
            del self.active_smells[smell_id]
    
    def mark_file_modified(self, commit_hash: str, commit_date: datetime.datetime):
        """Marca il file come modificato e aggiorna le statistiche."""
        self.total_modifications += 1
        self.commits_with_changes.add(commit_hash)
        
        # Aggiunge eventi FILE_MODIFIED per tutti gli smell attivi che persistono
        for smell_history in self.active_smells.values():
            # Controlla se questo smell ha già un evento per questo commit
            has_event_for_commit = any(event.commit_hash == commit_hash for event in smell_history.events)
            if not has_event_for_commit:
                event = SmellEvent(
                    commit_hash=commit_hash,
                    commit_date=commit_date,
                    event_type=SmellEventType.FILE_MODIFIED,
                    notes="File modified but smell unchanged"
                )
                smell_history.add_event(event)
    
    def check_removed_smells(self, commit_hash: str, commit_date: datetime.datetime, 
                           current_detections: List[SmellDetection]):
        """Controlla se alcuni smell sono stati rimossi in questo commit."""
        current_smell_ids = {d.get_smell_id() for d in current_detections}
        
        for smell_id, smell_history in list(self.active_smells.items()):
            if smell_id not in current_smell_ids:
                # Smell rimosso
                event = SmellEvent(
                    commit_hash=commit_hash,
                    commit_date=commit_date,
                    event_type=SmellEventType.SMELL_REMOVED,
                    previous_line=smell_history.current_line,
                    notes="Smell no longer detected in file"
                )
                smell_history.add_event(event)
                
                # Sposta negli inattivi
                self.inactive_smells[smell_id] = smell_history
                del self.active_smells[smell_id]
    
    def _determine_event_type(self, detection: SmellDetection, 
                            previous_detections: Optional[List[SmellDetection]] = None) -> SmellEventType:
        """Determina il tipo di evento basandosi sulle detection precedenti."""
        smell_id = detection.get_smell_id()
        
        # Se è la prima volta che vediamo questo smell
        if smell_id not in self.active_smells and smell_id not in self.inactive_smells:
            return SmellEventType.SMELL_INTRODUCED
        
        # Se lo smell esisteva già attivo
        if smell_id in self.active_smells:
            previous_smell = self.active_smells[smell_id]
            if previous_smell.current_line != detection.line:
                return SmellEventType.SMELL_MODIFIED
            else:
                return SmellEventType.SMELL_PERSISTED
        
        # Se lo smell era inattivo ed è stato rilevato di nuovo
        if smell_id in self.inactive_smells:
            return SmellEventType.SMELL_INTRODUCED  # Reintrodotto
        
        return SmellEventType.SMELL_INTRODUCED
    
    def get_summary(self) -> Dict:
        """Restituisce un riassunto degli smell nel file."""
        return {
            'filename': self.filename,
            'active_smells_count': len(self.active_smells),
            'inactive_smells_count': len(self.inactive_smells),
            'total_smells_ever': len(self.active_smells) + len(self.inactive_smells),
            'total_modifications': self.total_modifications,
            'total_commits_with_changes': len(self.commits_with_changes),
            'active_smells': [smell.get_summary() for smell in self.active_smells.values()],
            'inactive_smells': [smell.get_summary() for smell in self.inactive_smells.values()]
        }

@dataclass
class SmellEvolutionTracker:
    """Traccia l'evoluzione degli smell in tutto il progetto."""
    
    # Trackers per file
    file_trackers: Dict[str, FileSmellTracker] = field(default_factory=dict)
    
    # Cache delle detection precedenti per commit
    _previous_commit_detections: Dict[str, List[SmellDetection]] = field(default_factory=dict)
    
    def process_commit_smells(self, commit_hash: str, commit_date: datetime.datetime,
                            changed_files: List[str], smells_by_file: Dict[str, Dict]):
        """Processa gli smell rilevati in un commit."""
        
        # Processa ogni file modificato
        for filename in changed_files:
            if filename not in self.file_trackers:
                self.file_trackers[filename] = FileSmellTracker(filename)
            
            file_tracker = self.file_trackers[filename]
            
            # Marca il file come modificato
            file_tracker.mark_file_modified(commit_hash, commit_date)
            
            # Processa gli smell nel file
            if filename in smells_by_file:
                smell_data = smells_by_file[filename]
                current_detections = []
                
                # Converte le detection in oggetti SmellDetection
                for detection_data in smell_data.get('detections', []):
                    detection = SmellDetection(
                        filename=detection_data['filename'],
                        function_name=detection_data['function_name'],
                        smell_name=detection_data['smell_name'],
                        line=detection_data['line'],
                        description=detection_data['description'],
                        additional_info=detection_data['additional_info']
                    )
                    current_detections.append(detection)
                    
                    # Aggiunge la detection al tracker
                    previous_detections = self._previous_commit_detections.get(filename, [])
                    file_tracker.add_smell_detection(commit_hash, commit_date, detection, previous_detections)
                
                # Controlla se alcuni smell sono stati rimossi
                file_tracker.check_removed_smells(commit_hash, commit_date, current_detections)
                
                # Aggiorna la cache delle detection precedenti
                self._previous_commit_detections[filename] = current_detections
            else:
                # Nessuno smell rilevato, tutti gli smell attivi sono stati rimossi
                file_tracker.check_removed_smells(commit_hash, commit_date, [])
                self._previous_commit_detections[filename] = []
    
    def get_active_smells_summary(self) -> Dict:
        """Restituisce un riassunto degli smell attivi."""
        active_smells = []
        
        for file_tracker in self.file_trackers.values():
            for smell_history in file_tracker.active_smells.values():
                active_smells.append(smell_history.get_summary())
        
        return {
            'total_active_smells': len(active_smells),
            'smells_by_file': {filename: len(tracker.active_smells) 
                              for filename, tracker in self.file_trackers.items()},
            'active_smells': active_smells
        }
    
    def get_smell_evolution_report(self) -> Dict:
        """Genera un report completo sull'evoluzione degli smell."""
        total_smells = 0
        total_active = 0
        total_removed = 0
        smells_by_type = {}
        files_with_smells = 0
        
        smell_histories = []
        
        for file_tracker in self.file_trackers.values():
            if file_tracker.active_smells or file_tracker.inactive_smells:
                files_with_smells += 1
            
            # Processa smell attivi
            for smell_history in file_tracker.active_smells.values():
                total_smells += 1
                total_active += 1
                
                smell_type = smell_history.smell_name
                if smell_type not in smells_by_type:
                    smells_by_type[smell_type] = {'active': 0, 'removed': 0, 'total': 0}
                smells_by_type[smell_type]['active'] += 1
                smells_by_type[smell_type]['total'] += 1
                
                smell_histories.append(smell_history.get_summary())
            
            # Processa smell inattivi
            for smell_history in file_tracker.inactive_smells.values():
                total_smells += 1
                total_removed += 1
                
                smell_type = smell_history.smell_name
                if smell_type not in smells_by_type:
                    smells_by_type[smell_type] = {'active': 0, 'removed': 0, 'total': 0}
                smells_by_type[smell_type]['removed'] += 1
                smells_by_type[smell_type]['total'] += 1
                
                smell_histories.append(smell_history.get_summary())
        
        return {
            'summary': {
                'total_smells_tracked': total_smells,
                'active_smells': total_active,
                'removed_smells': total_removed,
                'files_with_smells': files_with_smells,
                'total_files_tracked': len(self.file_trackers)
            },
            'smells_by_type': smells_by_type,
            'smell_histories': smell_histories,
            'file_summaries': {filename: tracker.get_summary() 
                              for filename, tracker in self.file_trackers.items()}
        }
    
    def get_smell_trends_analysis(self) -> Dict:
        """Analizza i trend degli smell nel tempo."""
        trends = {
            'smell_introduction_trend': {},
            'smell_removal_trend': {},
            'most_problematic_files': [],
            'longest_living_smells': [],
            'most_frequently_modified_smells': [],
            'smell_hotspots': {}
        }
        
        # Analizza i file più problematici
        file_problems = []
        for filename, tracker in self.file_trackers.items():
            total_smells = len(tracker.active_smells) + len(tracker.inactive_smells)
            if total_smells > 0:
                file_problems.append({
                    'filename': filename,
                    'total_smells_ever': total_smells,
                    'active_smells': len(tracker.active_smells),
                    'inactive_smells': len(tracker.inactive_smells),
                    'total_modifications': tracker.total_modifications,
                    'smell_density': total_smells / max(tracker.total_modifications, 1)
                })
        
        trends['most_problematic_files'] = sorted(file_problems, 
                                                 key=lambda x: x['total_smells_ever'], 
                                                 reverse=True)[:10]
        
        # Analizza gli smell più longevi
        all_smells = []
        for tracker in self.file_trackers.values():
            all_smells.extend(list(tracker.active_smells.values()))
            all_smells.extend(list(tracker.inactive_smells.values()))
        
        long_living_smells = []
        for smell in all_smells:
            lifespan = smell.get_lifespan_days()
            if lifespan is not None and lifespan > 0:
                long_living_smells.append({
                    'smell_id': smell.smell_id,
                    'filename': smell.filename,
                    'smell_name': smell.smell_name,
                    'lifespan_days': lifespan,
                    'total_commits': len(smell.commits_with_smell),
                    'times_modified': smell.times_modified,
                    'is_active': smell.is_active
                })
        
        trends['longest_living_smells'] = sorted(long_living_smells, 
                                                key=lambda x: x['lifespan_days'], 
                                                reverse=True)[:10]
        
        # Analizza gli smell più frequentemente modificati
        frequently_modified = []
        for smell in all_smells:
            if smell.times_modified > 0:
                frequently_modified.append({
                    'smell_id': smell.smell_id,
                    'filename': smell.filename,
                    'smell_name': smell.smell_name,
                    'times_modified': smell.times_modified,
                    'times_file_changed': smell.times_file_changed,
                    'total_commits': len(smell.commits_with_smell),
                    'is_active': smell.is_active
                })
        
        trends['most_frequently_modified_smells'] = sorted(frequently_modified, 
                                                          key=lambda x: x['times_modified'], 
                                                          reverse=True)[:10]
        
        # Analizza gli hotspot degli smell (combinazione di frequenza e persistenza)
        hotspots = {}
        for tracker in self.file_trackers.values():
            for smell in list(tracker.active_smells.values()) + list(tracker.inactive_smells.values()):
                smell_type = smell.smell_name
                if smell_type not in hotspots:
                    hotspots[smell_type] = {
                        'total_occurrences': 0,
                        'active_occurrences': 0,
                        'files_affected': set(),
                        'average_lifespan': 0,
                        'total_modifications': 0
                    }
                
                hotspots[smell_type]['total_occurrences'] += 1
                if smell.is_active:
                    hotspots[smell_type]['active_occurrences'] += 1
                hotspots[smell_type]['files_affected'].add(smell.filename)
                hotspots[smell_type]['total_modifications'] += smell.times_modified
                
                lifespan = smell.get_lifespan_days()
                if lifespan:
                    current_avg = hotspots[smell_type]['average_lifespan']
                    count = hotspots[smell_type]['total_occurrences']
                    hotspots[smell_type]['average_lifespan'] = (current_avg * (count - 1) + lifespan) / count
        
        # Converte i set in liste per la serializzazione JSON
        for smell_type, data in hotspots.items():
            data['files_affected'] = list(data['files_affected'])
            data['files_affected_count'] = len(data['files_affected'])
        
        trends['smell_hotspots'] = dict(sorted(hotspots.items(), 
                                              key=lambda x: x[1]['total_occurrences'], 
                                              reverse=True))
        
        return trends
    
    def generate_smell_evolution_csv_data(self) -> List[Dict]:
        """Genera dati in formato CSV per l'evoluzione degli smell."""
        csv_data = []
        
        for tracker in self.file_trackers.values():
            all_smells = list(tracker.active_smells.values()) + list(tracker.inactive_smells.values())
            
            for smell in all_smells:
                for event in smell.events:
                    csv_data.append({
                        'smell_id': smell.smell_id,
                        'filename': smell.filename,
                        'function_name': smell.function_name,
                        'smell_name': smell.smell_name,
                        'commit_hash': event.commit_hash,
                        'commit_date': event.commit_date.isoformat(),
                        'event_type': event.event_type.value,
                        'line': event.current_line,
                        'previous_line': event.previous_line,
                        'notes': event.notes,
                        'is_smell_active': smell.is_active,
                        'smell_first_introduced': smell.first_introduced_commit,
                        'smell_lifespan_days': smell.get_lifespan_days(),
                        'times_modified': smell.times_modified
                    })
        
        return csv_data