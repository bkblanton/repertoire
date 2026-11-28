import json

import pytest
from helpers import run_fixture

from repertoire import coverage
from repertoire.character import analyze as character
from repertoire.engine import analyze as engine
from repertoire.openings import analyze as openings
from repertoire.preparation import analyze as preparation
from repertoire.ratings import analyze as ratings
from repertoire.report_insights import analyze as report_insights
from repertoire.vulnerabilities import analyze as vulnerabilities


@pytest.fixture(autouse=True)
def evaluation_store(tmp_path, monkeypatch):
    """Every test gets its own empty evaluation store and export path, never the local ones."""
    path = tmp_path / 'evals.sqlite'
    monkeypatch.setattr('repertoire.evals.STORE', path)
    monkeypatch.setattr('repertoire.evals.EXPORT', tmp_path / 'lichess_db_eval.jsonl.zst')
    return path


@pytest.fixture
def complete(tmp_path, monkeypatch):
    report, cache = run_fixture(tmp_path, monkeypatch, common_entry=True)
    path = tmp_path / 'white.json'
    for name, analyze in [
        ('vulnerabilities', vulnerabilities),
        ('preparation', preparation),
        ('character', character),
        ('ratings', ratings),
        ('openings', openings),
        ('insights', report_insights),
        ('engine', engine),
    ]:
        data = analyze(path, cache)
        path.with_suffix(f'.{name}.json').write_text(json.dumps(data))
    coverage.run(path, cache)
    return path, report
