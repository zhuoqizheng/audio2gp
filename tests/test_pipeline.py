from __future__ import annotations

import json
import zipfile
from pathlib import Path

import guitarpro as gp
import mido
import pytest

import audio2gp.pipeline as pipeline
from audio2gp.config import STEM_NAMES, Options
from audio2gp.gp_writer import verify_gp5
from audio2gp.keydetect import KeyInfo
from audio2gp.rhythm import RhythmInfo
from audio2gp.types import ChordEvent, NoteEvent, PreprocessedAudio

BPM = 120.0


def _chord_events() -> list[ChordEvent]:
    return [
        ChordEvent(0, 3840, 0.0, 2.0, "C", 0, "major", 0.8),
        ChordEvent(3840, 7680, 2.0, 4.0, "G", 7, "major", 0.7),
    ]


def _melodic_events() -> dict[str, list[NoteEvent]]:
    return {
        "vocals": [NoteEvent(0.05, 0.40, 60, 0.8)],
        "bass": [NoteEvent(0.05, 0.40, 40, 0.8)],
        "other": [
            NoteEvent(0.05, 0.40, 64, 0.8),
            NoteEvent(0.55, 0.90, 67, 0.7),
        ],
    }


def _drum_events() -> list[NoteEvent]:
    return [NoteEvent(0.05, 0.10, 36, 0.8), NoteEvent(0.55, 0.60, 38, 0.7)]


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    melodic: dict[str, list[NoteEvent]] | None = None,
    drum_events: list[NoteEvent] | None = None,
    stage_warnings: list[str] | None = None,
) -> tuple[Path, dict[str, Path]]:
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake input")
    source_dir = tmp_path / "source_stems"
    source_dir.mkdir()
    stem_sources: dict[str, Path] = {}
    for name in STEM_NAMES:
        source = source_dir / f"{name}.wav"
        source.write_bytes(f"fake {name}".encode())
        stem_sources[name] = source
    melodic_events = _melodic_events() if melodic is None else melodic
    drums = _drum_events() if drum_events is None else drum_events
    extra_warnings = list(stage_warnings or [])

    def fake_prepare(input_path, work_dir, options, logger=None, progress_cb=None):
        if progress_cb:
            progress_cb(50.0, "fake preprocessing")
        return PreprocessedAudio(
            path=work_dir / "song.wav",
            duration_s=4.0,
            samplerate=44100,
            bpm=BPM,
            warnings=[],
        )

    def fake_rhythm(wav_path, logger=None, progress_cb=None, *, meter=None):
        if progress_cb:
            progress_cb(100.0, "fake rhythm")
        return RhythmInfo(
            bpm=BPM,
            beats_per_bar=4,
            beat_times_s=[0.0, 0.5, 1.0, 1.5],
            downbeat_times_s=[0.0, 2.0],
            method="fake",
        )

    def fake_refine(drums_path, reference_bpm, logger=None):
        return float(reference_bpm), []

    def fake_detect_key(paths, warnings=None):
        return KeyInfo(
            tonic=7,
            mode="major",
            name="G major",
            confidence=0.31,
            method="fake",
        )

    def fake_separate(audio, options, device, logger=None, progress_cb=None):
        if progress_cb:
            progress_cb(100.0, "fake separation")
        return dict(stem_sources), list(extra_warnings), "cpu"

    def fake_transcribe(stems, options, bpm, logger=None, progress_cb=None):
        if progress_cb:
            progress_cb(100.0, "fake transcription")
        return {name: list(melodic_events.get(name, [])) for name in ("vocals", "bass", "other")}, []

    def fake_drums(stem_path, bpm, quantize, logger=None, progress_cb=None):
        if progress_cb:
            progress_cb(100.0, "fake drums")
        return list(drums), []

    monkeypatch.setattr(pipeline.preprocess_mod, "prepare", fake_prepare)
    monkeypatch.setattr(pipeline.rhythm_mod, "analyze_rhythm", fake_rhythm)
    monkeypatch.setattr(pipeline.rhythm_mod, "refine_tempo_from_drums", fake_refine)
    monkeypatch.setattr(pipeline.keydetect_mod, "detect_key", fake_detect_key)
    monkeypatch.setattr(pipeline.separate_mod, "separate", fake_separate)
    monkeypatch.setattr(pipeline.transcribe_mod, "transcribe_stems", fake_transcribe)
    monkeypatch.setattr(pipeline.drums_mod, "transcribe_drums", fake_drums)
    monkeypatch.setattr(pipeline, "resolve_device", lambda requested: ("cpu", []))
    return input_path, stem_sources


def _options(tmp_path: Path, **kwargs) -> Options:
    params = dict(
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        demucs_model="htdemucs",
        quantize=16,
        device="cpu",
        keep_stems=True,
        # Most tests fake stem files with non-audio bytes; chord tests opt in.
        detect_chords=False,
    )
    params.update(kwargs)
    return Options(**params)


def _zip_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def _gp5_chord_names(path: Path) -> list[str]:
    song = gp.parse(str(path))
    names: list[str] = []
    for track in song.tracks:
        for measure in track.measures:
            for beat in measure.voices[0].beats:
                chord = getattr(beat.effect, "chord", None)
                if chord is not None:
                    names.append(chord.name)
    return names


def _midi_markers(path: Path) -> list[tuple[int, str]]:
    markers: list[tuple[int, str]] = []
    tick = 0
    for message in mido.MidiFile(str(path)).tracks[0]:
        tick += int(message.time)
        if message.type == "marker":
            markers.append((tick, message.text))
    return markers


def test_pipeline_writes_outputs_meta_zip_and_monotonic_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)
    progress: list[tuple[str, float, str]] = []
    result = pipeline.run(
        input_path,
        _options(tmp_path),
        progress_cb=lambda stage, pct, message: progress.append((stage, pct, message)),
    )

    assert result.status == "ok"
    assert result.device == "cpu"
    assert result.bpm == BPM
    assert result.note_counts == {"vocals": 1, "drums": 2, "bass": 1, "other": 2}
    assert result.out_dir.is_dir()
    assert result.work_dir.is_dir()
    for path in result.midi.values():
        assert path.is_file()
    assert {name: path.is_file() for name, path in result.midi.items()} == {
        name: True for name in STEM_NAMES
    }
    assert result.full_midi.is_file()
    assert result.gp5 is not None and result.gp5.is_file()
    assert result.zip_path.is_file()
    assert all(result.stems[name].is_file() for name in STEM_NAMES)

    assert verify_gp5(result.gp5, expected_measures=2, expected_tracks=4) == []
    song = gp.parse(str(result.gp5))
    assert [track.name for track in song.tracks] == ["Guitar", "Bass", "Drums", "Vocals"]
    assert result.meta["status"] == "ok"
    assert result.meta["tracks"] == ["Guitar", "Bass", "Drums", "Vocals"]
    assert result.meta["note_counts"] == result.note_counts
    assert result.meta["files"]["gp5"] == "song.gp5"
    assert set(result.meta["files"]["stems"]) == set(STEM_NAMES)

    names = _zip_names(result.zip_path)
    assert {"meta.json", "song.gp5", "midi/full.mid"} <= names
    assert {f"midi/{name}.mid" for name in STEM_NAMES} <= names
    assert {f"stems/{name}.wav" for name in STEM_NAMES} <= names

    assert progress
    percentages = [pct for _stage, pct, _message in progress]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100.0
    stages = [stage for stage, _pct, _message in progress]
    assert stages[0] == "preprocess"
    assert stages[-1] == "gp5"
    assert set(stages) == {
        "preprocess",
        "rhythm",
        "separate",
        "analyze",
        "transcribe",
        "post",
        "gp5",
    }
    assert result.meta["rhythm"]["method"] == "fake"
    assert result.meta["rhythm"]["beats_per_bar"] == 4
    assert result.meta["key"]["name"] == "G major"
    assert result.key["name"] == "G major"
    assert result.meta["audio"]["tempo_source"] == "drums-verified"


def test_high_vocal_notes_respect_max_fret_in_editable_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    melodic = _melodic_events()
    melodic["vocals"] = [NoteEvent(0.05, 0.40, 80, 0.8)]  # fret 16 in standard tuning
    input_path, _sources = _install_fakes(monkeypatch, tmp_path, melodic=melodic)
    result = pipeline.run(input_path, _options(tmp_path))

    assert result.status == "ok"
    assert not any("在线编辑状态" in warning for warning in result.meta["warnings"])
    state = json.loads((result.out_dir / "tracks.json").read_text(encoding="utf-8"))
    vocals = next(track for track in state["tracks"] if track["stem"] == "vocals")
    assert vocals["notes"]
    assert all(note["fret"] <= state["max_fret"] for note in vocals["notes"])


def test_manual_downbeat_works_for_non_4_4_meter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)

    def unexpected_rhythm(*args, **kwargs):  # manual alignment must not analyze
        raise AssertionError("automatic downbeat analysis should not run")

    monkeypatch.setattr(pipeline.rhythm_mod, "analyze_rhythm", unexpected_rhythm)
    result = pipeline.run(
        input_path,
        _options(
            tmp_path,
            time_signature="3/4",
            manual_downbeat_s=3.0,
        ),
    )
    assert result.status == "ok"
    assert result.meta["time_signature"] == "3/4"
    assert result.meta["bar_ticks"] == 2880
    assert result.meta["rhythm"]["mode"] == "manual"
    assert result.meta["rhythm"]["method"] == "manual"
    assert result.meta["rhythm"]["time_signature"] == "3/4"
    assert result.meta["rhythm"]["beats_per_bar"] == 3
    assert result.meta["rhythm"]["first_downbeat_s"] == 3.0
    assert result.meta["rhythm"]["grid_offset_s"] == 0.0

    song = gp.parse(str(result.gp5))
    assert len(song.measureHeaders) == result.meta["measure_count"]
    for header in song.measureHeaders:
        assert (
            header.timeSignature.numerator,
            header.timeSignature.denominator.value,
        ) == (3, 4)
    assert verify_gp5(result.gp5, result.meta["measure_count"], 4, time_signature="3/4") == []


def test_non_4_4_meter_runs_meter_aware_auto_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)

    def fake_rhythm(wav_path, logger=None, progress_cb=None, *, meter=None):
        assert meter is not None and meter.label == "6/8"
        return RhythmInfo(
            bpm=BPM,
            beats_per_bar=3,
            beat_times_s=[0.0, 0.5, 1.0, 1.5],
            downbeat_times_s=[0.0, 1.5],
            method="fake",
        )

    monkeypatch.setattr(pipeline.rhythm_mod, "analyze_rhythm", fake_rhythm)
    result = pipeline.run(input_path, _options(tmp_path, time_signature="6/8"))
    assert result.status == "ok"
    rhythm = result.meta["rhythm"]
    assert rhythm["mode"] == "auto"
    assert rhythm["method"] == "fake"
    assert rhythm["time_signature"] == "6/8"
    assert rhythm["first_downbeat_s"] == 0.0
    assert rhythm["auto_alignment_supported"] is True
    assert not any("只支持 4/4" in warning for warning in result.warnings)
    song = gp.parse(str(result.gp5))
    for header in song.measureHeaders:
        assert (
            header.timeSignature.numerator,
            header.timeSignature.denominator.value,
        ) == (6, 8)


def test_selected_stems_limit_midi_gp5_and_stem_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, sources = _install_fakes(monkeypatch, tmp_path)
    result = pipeline.run(
        input_path,
        _options(tmp_path, selected_stems=("vocals", "other")),
    )
    assert result.status == "ok"
    assert result.meta["selected_stems"] == ["vocals", "other"]
    assert set(result.stems) == {"vocals", "other"}
    assert set(result.midi) == {"vocals", "other"}
    assert set(result.note_counts) == {"vocals", "other"}
    assert result.meta["tracks"] == ["Guitar", "Vocals"]
    assert result.stems["vocals"] == result.out_dir / "stems" / "vocals.wav"
    assert result.stems["other"] == result.out_dir / "stems" / "other.wav"

    names = _zip_names(result.zip_path)
    assert {
        "meta.json",
        "song.gp5",
        "midi/full.mid",
        "midi/vocals.mid",
        "midi/other.mid",
        "stems/vocals.wav",
        "stems/other.wav",
    } <= names
    assert "midi/bass.mid" not in names
    assert "midi/drums.mid" not in names
    assert "stems/bass.wav" not in names
    assert "stems/drums.wav" not in names

    song = gp.parse(str(result.gp5))
    assert [track.name for track in song.tracks] == ["Guitar", "Vocals"]
    assert verify_gp5(result.gp5, expected_measures=2, expected_tracks=2) == []
    assert sources["bass"].is_file()


def test_drums_only_selection_produces_one_track(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)
    result = pipeline.run(
        input_path,
        _options(tmp_path, selected_stems=("drums",)),
    )
    assert result.status == "ok"
    assert result.meta["tracks"] == ["Drums"]
    assert set(result.midi) == {"drums"}
    assert set(result.stems) == {"drums"}
    assert result.note_counts == {"drums": 2}
    song = gp.parse(str(result.gp5))
    assert [track.name for track in song.tracks] == ["Drums"]


def test_empty_or_impossible_stem_selection_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one stem"):
        Options(selected_stems=())
    with pytest.raises(ValueError, match="no GP5 track"):
        Options(selected_stems=("vocals",), include_vocals=False)
    with pytest.raises(ValueError, match="no GP5 track"):
        Options(selected_stems=("other",), guitar_stem="vocals")
    with pytest.raises(ValueError, match="unknown values"):
        Options(selected_stems=("piano",))
    # MIDI-only mode does not require a stem that can build a GP5 track.
    midi_only = Options(
        selected_stems=("vocals",), include_vocals=False, export_gp5=False
    )
    assert midi_only.output_format == "midi_only"
    assert Options(selected_stems=("other",), include_vocals=False).output_format == "gp5_midi"


def _patch_chords(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_analyze(stems, bpm, **kwargs):
        calls.append({"stems": dict(stems), "bpm": bpm, **kwargs})
        progress_cb = kwargs.get("progress_cb")
        if progress_cb:
            progress_cb(100.0, "fake chords")
        return _chord_events()

    monkeypatch.setattr(pipeline.chords_mod, "analyze_chords", fake_analyze)
    return calls


def test_chords_are_written_to_json_gp5_midi_and_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)
    calls = _patch_chords(monkeypatch)
    result = pipeline.run(
        input_path,
        _options(tmp_path, detect_chords=True, selected_stems=("other", "bass")),
    )

    assert result.status == "ok"
    assert result.output_format == "gp5_midi"
    assert [event.label for event in result.chords] == ["C", "G"]
    assert result.chords_path == result.out_dir / "chords.json"
    assert set(calls[0]["stems"]) == {"other", "bass"}
    assert calls[0]["offset_s"] == 0.0
    assert calls[0]["duration_s"] == 4.0

    chords_doc = json.loads(result.chords_path.read_text(encoding="utf-8"))
    assert chords_doc["summary"]["progression"] == ["C", "G"]
    assert [segment["bar"] for segment in chords_doc["segments"]] == [1, 2]
    assert result.meta["chords"]["summary"]["unique_chords"] == ["C", "G"]
    assert result.meta["output_format"] == "gp5_midi"
    assert result.meta["files"]["chords"] == "chords.json"
    assert result.state["chords"][0]["label"] == "C"

    assert _gp5_chord_names(result.gp5) == ["C", "G"]
    assert _midi_markers(result.full_midi) == [(0, "C"), (3840, "G")]
    names = _zip_names(result.zip_path)
    assert {"chords.json", "tracks.json", "midi/full.mid", "song.gp5"} <= names


def test_midi_only_skips_gp5_tracks_and_editor_but_keeps_chords(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)
    _patch_chords(monkeypatch)
    result = pipeline.run(
        input_path,
        _options(
            tmp_path,
            export_gp5=False,
            detect_chords=True,
            selected_stems=("other", "bass"),
        ),
    )

    assert result.status == "ok"
    assert result.output_format == "midi_only"
    assert result.gp5 is None
    assert result.state == {}
    assert not (result.out_dir / "song.gp5").exists()
    assert not (result.out_dir / "tracks.json").exists()
    assert set(result.midi) == {"other", "bass"}
    assert result.full_midi.is_file()
    assert result.chords_path is not None and result.chords_path.is_file()
    assert _midi_markers(result.full_midi) == [(0, "C"), (3840, "G")]
    assert result.meta["output_format"] == "midi_only"
    assert result.meta["tracks"] == []
    assert "gp5" not in result.meta["files"]
    assert "tracks" not in result.meta["files"]
    assert result.meta["files"]["chords"] == "chords.json"
    assert "song.gp5" not in result.meta["files"].values()

    names = _zip_names(result.zip_path)
    assert {
        "meta.json",
        "chords.json",
        "midi/other.mid",
        "midi/bass.mid",
        "midi/full.mid",
    } <= names
    assert not any(name.endswith(".gp5") or name == "tracks.json" for name in names)


def test_midi_only_without_chords_omits_chord_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)
    result = pipeline.run(
        input_path,
        _options(tmp_path, export_gp5=False, detect_chords=False),
    )
    assert result.status == "ok"
    assert result.chords == []
    assert result.chords_path is None
    assert not (result.out_dir / "chords.json").exists()
    assert "chords" not in result.meta["files"]
    assert result.meta["chords"] == {}
    assert _midi_markers(result.full_midi) == []


def test_downbeat_alignment_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)
    result = pipeline.run(
        input_path,
        _options(tmp_path, align_downbeats=False),
    )
    assert result.status == "ok"
    assert result.meta["rhythm"]["enabled"] is False
    assert result.meta["rhythm"]["method"] == "off"
    assert result.meta["rhythm"]["grid_offset_s"] == 0.0


def test_keep_stems_false_leaves_stems_out_of_output_and_zip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, sources = _install_fakes(monkeypatch, tmp_path)
    result = pipeline.run(
        input_path,
        _options(tmp_path, keep_stems=False),
    )
    assert result.status == "ok"
    assert not (result.out_dir / "stems").exists()
    for name in STEM_NAMES:
        assert result.stems[name] == sources[name]
        assert result.stems[name].is_file()
    names = _zip_names(result.zip_path)
    assert not any(name.startswith("stems/") for name in names)
    assert "stems" not in result.meta["files"]
    assert result.gp5 is not None and result.gp5.is_file()


@pytest.mark.parametrize(
    ("include_vocals", "guitar_stem", "expected_tracks"),
    [
        (True, "other", ["Guitar", "Bass", "Drums", "Vocals"]),
        (False, "other", ["Guitar", "Bass", "Drums"]),
        (True, "vocals", ["Guitar", "Bass", "Drums"]),
    ],
)
def test_track_set_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    include_vocals: bool,
    guitar_stem: str,
    expected_tracks: list[str],
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)
    result = pipeline.run(
        input_path,
        _options(
            tmp_path,
            include_vocals=include_vocals,
            guitar_stem=guitar_stem,
        ),
    )
    assert result.meta["tracks"] == expected_tracks
    song = gp.parse(str(result.gp5))
    assert [track.name for track in song.tracks] == expected_tracks


def test_empty_stems_still_produce_empty_tracks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = {name: [] for name in ("vocals", "bass", "other")}
    input_path, _sources = _install_fakes(
        monkeypatch, tmp_path, melodic=empty, drum_events=[]
    )
    result = pipeline.run(input_path, _options(tmp_path))
    assert result.status == "ok"
    assert result.note_counts == {name: 0 for name in STEM_NAMES}
    assert result.gp5 is not None
    assert any("empty drum track" in warning for warning in result.warnings)
    song = gp.parse(str(result.gp5))
    for track in song.tracks:
        for measure in track.measures:
            assert all(
                beat.status == gp.BeatStatus.rest for beat in measure.voices[0].beats
            )


def test_gp5_failure_is_partial_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)

    def broken_gp5(*args, **kwargs):
        raise RuntimeError("writer exploded")

    monkeypatch.setattr(pipeline, "write_gp5", broken_gp5)
    result = pipeline.run(input_path, _options(tmp_path))

    assert result.status == "partial"
    assert result.gp5 is None
    assert result.full_midi.is_file()
    assert result.zip_path.is_file()
    assert any("GP5 generation failed" in warning for warning in result.warnings)
    assert result.meta["status"] == "partial"
    assert result.meta["gp5_error"] == "writer exploded"
    assert "gp5" not in result.meta["files"]
    assert not (result.out_dir / "song.gp5").exists()
    assert "midi/full.mid" in _zip_names(result.zip_path)


def test_stage_failure_writes_error_meta_and_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path, _sources = _install_fakes(monkeypatch, tmp_path)

    def broken_prepare(*args, **kwargs):
        raise ValueError("unreadable input")

    monkeypatch.setattr(pipeline.preprocess_mod, "prepare", broken_prepare)
    with pytest.raises(pipeline.PipelineError) as caught:
        pipeline.run(input_path, _options(tmp_path))
    exc = caught.value
    assert exc.stage == "preprocess"
    assert "unreadable input" in exc.message
    assert exc.work_dir.is_dir()
    assert exc.log_path.is_file()
    assert (exc.out_dir / "meta.json").is_file()
    import json

    meta = json.loads((exc.out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "error"
    assert meta["stage"] == "preprocess"
    assert "unreadable input" in meta["error"]


def test_missing_input_writes_error_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch, tmp_path)
    missing = tmp_path / "does-not-exist.wav"
    with pytest.raises(pipeline.PipelineError) as caught:
        pipeline.run(missing, _options(tmp_path))
    assert caught.value.stage == "preprocess"
    assert (caught.value.out_dir / "meta.json").is_file()
