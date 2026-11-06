"""Folding tool discovery for a fresh Conda environment and custom installations."""

import pytest

from rnamap import features


@pytest.fixture
def folding_installation(tmp_path, monkeypatch):
    for name in ("CONTRAFOLD", "ETERNAFOLD", "ETERNAFOLD_PARAMS", "ETERNAFOLD_PATH", "ETERNAFOLD_PARAMETERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(features.sys, "prefix", str(tmp_path / "python"))
    prefix = tmp_path / "conda"
    binaries = prefix / "bin"
    binaries.mkdir(parents=True)
    for name in ("contrafold", "eternafold"):
        binary = binaries / name
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
    params = prefix / "lib/eternafold-lib/parameters/EternaFoldParams.v1"
    params.parent.mkdir(parents=True)
    params.touch()
    monkeypatch.setenv("PATH", str(binaries))
    return tuple(str(p) for p in (binaries / "contrafold", binaries / "eternafold", params))


def test_discover_conda_tools_on_path(folding_installation):
    # No activation variables or matching Python prefix: locate parameters beside the executable.
    assert features.folding_tools() == folding_installation


@pytest.mark.parametrize("names", [("ETERNAFOLD", "ETERNAFOLD_PARAMS"), ("ETERNAFOLD_PATH", "ETERNAFOLD_PARAMETERS")])
def test_tool_environment_variables(folding_installation, monkeypatch, names):
    cf, ef, params = folding_installation
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("CONTRAFOLD", cf)
    monkeypatch.setenv(names[0], ef)
    monkeypatch.setenv(names[1], params)
    assert features.folding_tools() == folding_installation


def test_explicit_paths_override_environment(folding_installation, monkeypatch):
    for name in ("CONTRAFOLD", "ETERNAFOLD", "ETERNAFOLD_PARAMS", "ETERNAFOLD_PATH", "ETERNAFOLD_PARAMETERS"):
        monkeypatch.setenv(name, "/missing/custom/path")
    assert features.folding_tools(*folding_installation) == folding_installation


@pytest.mark.parametrize("override", ["contrafold_bin", "eternafold_bin", "eternafold_params"])
def test_invalid_override_is_reported(folding_installation, override):
    # A broken explicit override must not silently use the automatically discovered installation.
    with pytest.raises(SystemExit, match="Missing folding tools:.*conda env create"):
        features.folding_tools(**{override: "/missing/custom/path"})


def test_missing_tools_fail_before_featurizing(folding_installation, monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(SystemExit, match="CONTRAfold.*EternaFold.*EternaFold parameters"):
        features.featurize(["test"], ["ACGU"])
