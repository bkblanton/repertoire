"""Keep readable reports above their supporting data directory."""
from pathlib import Path


def report_directory(result_path):
    directory = Path(result_path).parent
    return directory.parent if directory.name == 'data' else directory
