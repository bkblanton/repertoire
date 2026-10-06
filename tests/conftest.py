import json

import pytest

from repertoire_score.character import analyze as character
from repertoire_score.openings import analyze as openings
from repertoire_score.preparation import analyze as preparation
from repertoire_score.ratings import analyze as ratings
from repertoire_score.report_insights import analyze as report_insights
from repertoire_score.vulnerabilities import analyze as vulnerabilities
from test_chapter_policies import run_fixture


@pytest.fixture
def complete(tmp_path, monkeypatch):
    report, cache = run_fixture(tmp_path, monkeypatch, common_entry=True)
    path = tmp_path / 'white.json'
    for name, analyze in [('vulnerabilities', vulnerabilities), ('preparation', preparation), ('character', character), ('ratings', ratings), ('openings', openings), ('insights', report_insights)]:
        data = analyze(path, cache)
        path.with_suffix(f'.{name}.json').write_text(json.dumps(data))
    return path, report
