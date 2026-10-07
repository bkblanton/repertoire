import json

import pytest
from helpers import run_fixture

from repertoire.character import analyze as character
from repertoire.openings import analyze as openings
from repertoire.preparation import analyze as preparation
from repertoire.ratings import analyze as ratings
from repertoire.report_insights import analyze as report_insights
from repertoire.vulnerabilities import analyze as vulnerabilities


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
    ]:
        data = analyze(path, cache)
        path.with_suffix(f'.{name}.json').write_text(json.dumps(data))
    return path, report
