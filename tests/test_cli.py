import importlib
import sys

import pytest

from repertoire_score import cli


def test_every_command_dispatches_to_a_stage_main():
    for name, module, _ in cli.COMMANDS:
        assert callable(importlib.import_module(f'repertoire_score.{module}').main), name


def test_dispatch_passes_arguments_and_restores_argv(monkeypatch):
    seen = {}
    monkeypatch.setattr('repertoire_score.render.main', lambda: seen.update(argv=list(sys.argv)))
    before = list(sys.argv)
    cli.main(['report', 'white.json', '--top', '3'])
    assert seen['argv'] == ['repertoire report', 'white.json', '--top', '3']
    assert sys.argv == before


def test_unknown_or_missing_command_lists_commands(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(['bogus'])
    assert error.value.code == 2
    assert 'rating-correlations' in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 0
    assert 'build' in capsys.readouterr().out


def test_stage_help_uses_the_subcommand_name(capsys):
    with pytest.raises(SystemExit):
        cli.main(['score', '--help'])
    assert capsys.readouterr().out.startswith('usage: repertoire score')


def test_interrupt_exits_cleanly_with_a_resume_hint(monkeypatch, capsys):
    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr('repertoire_score.fetch.main', interrupted)
    before = list(sys.argv)
    with pytest.raises(SystemExit) as error:
        cli.main(['fetch'])
    assert error.value.code == 130
    assert 'rerun the same command to continue' in capsys.readouterr().err
    assert sys.argv == before
