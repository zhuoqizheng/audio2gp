from __future__ import annotations

from pathlib import Path

import pytest

from audio2gp import cli


def _args(tmp_path: Path, *extra: str):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    return cli.build_parser().parse_args([str(audio), "-o", str(tmp_path), *extra])


def test_defaults_cover_the_local_pipeline(tmp_path: Path) -> None:
    options = cli._options_from_args(_args(tmp_path))
    assert options.quantize == 16
    assert options.demucs_model == "htdemucs_ft"
    assert options.device == "auto"
    assert options.selected_stems == ("vocals", "drums", "bass", "other")
    assert options.guitar_tuning == "standard"
    assert options.bass_tuning == "standard"
    assert options.suggest_tuning is False
    assert options.detect_key is True
    assert options.detect_techniques is True
    assert options.tempo_from_drums is True
    assert options.cache_stems is True
    assert options.ghost_amplitude == 0.15
    assert options.max_polyphony == 6
    assert options.onset_threshold == 0.5
    assert options.frame_threshold == 0.3
    assert options.detect_chords is True
    assert options.export_gp5 is True
    assert options.output_format == "gp5_midi"
    assert options.time_signature == "4/4"
    assert options.manual_downbeat_s is None


def test_new_flags_are_plumbed_to_options(tmp_path: Path) -> None:
    options = cli._options_from_args(
        _args(
            tmp_path,
            "--stems",
            "other,vocals",
            "--guitar-tuning",
            "drop_d",
            "--bass-tuning",
            "drop_d",
            "--suggest-tuning",
            "--ghost-amplitude",
            "0.3",
            "--max-polyphony",
            "4",
            "--onset-threshold",
            "0.6",
            "--frame-threshold",
            "0.4",
            "--no-detect-key",
            "--no-techniques",
            "--no-tempo-from-drums",
            "--no-cache-stems",
            "--quantize",
            "12",
            "--no-align-downbeats",
            "--time-signature",
            "12/8",
            "--manual-downbeat",
            "1.25",
            "--no-vocals-track",
            "--no-keep-stems",
            "--no-chords",
            "--midi-only",
        )
    )
    assert options.selected_stems == ("vocals", "other")
    assert options.guitar_tuning == "drop_d"
    assert options.bass_tuning == "drop_d"
    assert options.suggest_tuning is True
    assert options.ghost_amplitude == 0.3
    assert options.max_polyphony == 4
    assert options.onset_threshold == 0.6
    assert options.frame_threshold == 0.4
    assert options.detect_key is False
    assert options.detect_techniques is False
    assert options.tempo_from_drums is False
    assert options.time_signature == "12/8"
    assert options.manual_downbeat_s == 1.25
    assert options.meter.bar_ticks == 5760
    assert options.cache_stems is False
    assert options.quantize == 12
    assert options.align_downbeats is False
    assert options.include_vocals is False
    assert options.keep_stems is False
    assert options.detect_chords is False
    assert options.export_gp5 is False
    assert options.output_format == "midi_only"


def test_bpm_override_and_zero(tmp_path: Path) -> None:
    assert cli._options_from_args(_args(tmp_path, "--bpm", "93")).bpm_override == 93.0
    assert cli._options_from_args(_args(tmp_path, "--bpm", "0")).bpm_override is None


def test_all_stems_alias_and_unknown_stem(tmp_path: Path) -> None:
    assert cli._options_from_args(_args(tmp_path, "--stems", "all")).selected_stems == (
        "vocals",
        "drums",
        "bass",
        "other",
    )
    with pytest.raises(ValueError, match="unknown --stems"):
        cli._options_from_args(_args(tmp_path, "--stems", "piano"))


def test_missing_input_exits_nonzero(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.wav"
    assert cli.main([str(missing)]) == 1
    assert "input audio does not exist" in capsys.readouterr().err


def test_main_propagates_pipeline_error(tmp_path: Path, monkeypatch, capsys) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake")

    def broken(*args, **kwargs):
        raise cli.PipelineError(
            "separate",
            "boom",
            job_id="job",
            work_dir=tmp_path / "work",
            out_dir=tmp_path / "out",
            log_path=tmp_path / "work" / "pipeline.log",
        )

    monkeypatch.setattr(cli, "run", broken)
    assert cli.main([str(audio), "-o", str(tmp_path)]) == 1
    assert "error [separate]: boom" in capsys.readouterr().err


def test_stable_module_aliases() -> None:
    from audio2gp import export_gp5, separation

    assert separation.separate is __import__(
        "audio2gp.separate", fromlist=["separate"]
    ).separate
    assert export_gp5.write_gp5 is __import__(
        "audio2gp.gp_writer", fromlist=["write_gp5"]
    ).write_gp5
