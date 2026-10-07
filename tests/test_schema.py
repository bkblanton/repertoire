from repertoire_score import schema
from helpers import run_fixture


def missing(value, kind):
    return sorted(kind.__required_keys__ - set(value))


def test_saved_score_matches_the_documented_schema(tmp_path, monkeypatch):
    report, _ = run_fixture(tmp_path, monkeypatch, common_entry=True)
    assert missing(report, schema.ScoreResult) == []
    assert missing(report['overall'], schema.ScoreSummary) == []
    assert missing(report['overall']['posterior'], schema.Posterior) == []
    assert missing(report['manifest'], schema.ScoreManifest) == []
    for chapter in report['chapters']:
        assert missing(chapter, schema.Chapter) == []
        assert set(chapter['score']) <= set(schema.ChapterScore.__annotations__)
    for event in report['events']:
        assert missing(event, schema.StoppingEvent) == []
