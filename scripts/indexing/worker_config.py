from dataclasses import dataclass
from typing import List


@dataclass
class WorkerConfig:
    chunk_size: int
    chunk_overlap: int
    clean_html: bool
    lowercase: bool
    file_filters: List[str]