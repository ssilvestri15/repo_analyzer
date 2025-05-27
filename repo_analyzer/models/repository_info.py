"""Modello per le informazioni di un repository."""

from dataclasses import dataclass

@dataclass
class RepositoryInfo:
    """Classe per memorizzare le informazioni di un repository."""
    url: str
    name: str = ""
    local_path: str = ""
    output_dir: str = ""
    temp_dir: str = ""
    
    def __post_init__(self):
        """Inizializza i campi derivati se non specificati."""
        if not self.name:
            # Estrai il nome dal URL
            self.name = self.url.rstrip('/').split('/')[-1]
            if self.name.endswith('.git'):
                self.name = self.name[:-4]