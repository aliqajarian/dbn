import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WordEntry:
    word: str
    category: str
    severity: int

class WordListManager:
    def __init__(self, word_lists_dir: str = None):
        if word_lists_dir is None:
            word_lists_dir = os.path.join(os.path.dirname(__file__), "word_lists")
        self.word_lists_dir = Path(word_lists_dir)
        self.word_lists: Dict[str, List[WordEntry]] = {}
        self.load_word_lists()

    def load_word_lists(self) -> None:
        """Load all word lists from the word_lists directory."""
        for file_path in self.word_lists_dir.glob("*.txt"):
            if file_path.name == "word_list_manager.py":
                continue
            self._load_word_list(file_path)

    def _load_word_list(self, file_path: Path) -> None:
        """Load a single word list file."""
        category_name = file_path.stem
        self.word_lists[category_name] = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    word, category, severity = line.split('|')
                    entry = WordEntry(
                        word=word.strip(),
                        category=category.strip(),
                        severity=int(severity.strip())
                    )
                    self.word_lists[category_name].append(entry)
                except ValueError:
                    print(f"Warning: Invalid line format in {file_path}: {line}")

    def get_words_by_category(self, category: str) -> List[WordEntry]:
        """Get all words from a specific category."""
        return self.word_lists.get(category, [])

    def get_words_by_severity(self, min_severity: int = 1, max_severity: int = 5) -> List[WordEntry]:
        """Get all words within a severity range."""
        result = []
        for entries in self.word_lists.values():
            result.extend([entry for entry in entries 
                         if min_severity <= entry.severity <= max_severity])
        return result

    def check_text(self, text: str) -> Dict[str, List[Tuple[str, int]]]:
        """
        Check text for matches in word lists.
        Returns a dictionary of category -> list of (word, severity) matches.
        """
        text = text.lower()
        matches = {}
        
        for category, entries in self.word_lists.items():
            category_matches = []
            for entry in entries:
                if entry.word.lower() in text:
                    category_matches.append((entry.word, entry.severity))
            if category_matches:
                matches[category] = category_matches
                
        return matches

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self.word_lists.keys())

    def get_severity_summary(self, text: str) -> Dict[str, int]:
        """
        Get a summary of severity scores by category for the given text.
        Returns a dictionary of category -> total severity score.
        """
        matches = self.check_text(text)
        summary = {}
        
        for category, word_matches in matches.items():
            total_severity = sum(severity for _, severity in word_matches)
            summary[category] = total_severity
            
        return summary 