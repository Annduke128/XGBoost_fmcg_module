"""Tests for CLI entrypoint."""

import subprocess
import sys


def test_cli_no_args_exits_1():
    result = subprocess.run(
        [sys.executable, "-m", "ml.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Usage" in result.stdout


def test_cli_unknown_command_exits_1():
    result = subprocess.run(
        [sys.executable, "-m", "ml.cli", "unknown"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Unknown command" in result.stdout
