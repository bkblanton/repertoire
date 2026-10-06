"""Where outputs go: readable reports above their supporting data directory, and the data file format."""
import json
from pathlib import Path


def report_directory(result_path):
    directory = Path(result_path).parent
    return directory.parent if directory.name == 'data' else directory


def data_json(value):
    """Saved analysis JSON: compact, since the files are large and read by programs rather than people."""
    return json.dumps(value, separators=(',', ':'), allow_nan=False)
