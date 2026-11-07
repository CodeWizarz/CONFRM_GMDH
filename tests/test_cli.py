import pytest

from gmdh_conformal.cli import main


def test_cli_lists_research_commands_without_a_synthetic_data_mode(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "run" in help_text
    assert "validate" in help_text
    assert "--synthetic" not in help_text
