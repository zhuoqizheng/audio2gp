from __future__ import annotations

import io
import json
from pathlib import Path

import mido
import pytest

from audio2gp import chords, drums, keydetect, pipeline, preprocess, rhythm, separate, transcribe
from audio2gp.config import Options
from audio2gp.midi_in import prepare


def _note(track: mido.MidiTrack, pitch: int, channel: int, start: int, length: int) -> None:
    track.append(mido.Message("note_on", note=pitch, velocity=100, channel=channel, time=start))
    track.append(mido.Message("note_off", note=pitch, velocity=0, channel=channel, time=length))


def _midi_bytes() -> bytes:
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(100), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    meta.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=480)
    )

    bass = mido.MidiTrack()
    mid.tracks.append(bass)
    bass.append(mido.MetaMessage("track_name", name="Bass DI", time=0))
    bass.append(mido.Message("program_change", program=33, channel=0, time=0))
    for index in range(2):
        _note(bass, 40 + index, 0, 0 if index == 0 else 480, 480)

    guitar = mido.MidiTrack()
    mid.tracks.append(guitar)
    guitar.append(mido.MetaMessage("track_name", name="Guitar L", time=0))
    guitar.append(mido.Message("program_change", program=25, channel=1, time=0))
    _note(guitar, 64, 1, 0, 960)

    vocal = mido.MidiTrack()
    mid.tracks.append(vocal)
    vocal.append(mido.MetaMessage("track_name", name="Vocal Take", time=0))
    _note(vocal, 72, 2, 0, 480)

    drums_track = mido.MidiTrack()
    mid.tracks.append(drums_track)
    drums_track.append(mido.MetaMessage("track_name", name="Percussion", time=0))
    _note(drums_track, 36, 9, 0, 240)
    _note(drums_track, 38, 9, 480, 240)

    buffer = io.BytesIO()
    mid.save(file=buffer)
    return buffer.getvalue()


def _write_midi(tmp_path: Path) -> Path:
    path = tmp_path / "song.mid"
    path.write_bytes(_midi_bytes())
    return path


def test_prepare_maps_tracks_and_warns_about_tempo_and_meter(tmp_path: Path) -> None:
    info = prepare(_write_midi(tmp_path), Options(time_signature="4/4"))
    assert info.bpm == pytest.approx(100.0)
    assert {name: len(notes) for name, notes in info.events.items()} == {
        "bass": 2,
        "other": 1,
        "vocals": 1,
        "drums": 2,
    }
    assert any("速度事件" in warning for warning in info.warnings)
    assert any("3/4" in warning and "4/4" in warning for warning in info.warnings)


def test_pipeline_midi_input_skips_audio_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("audio processing must not run for MIDI input")

    for module, name in [
        (preprocess, "prepare"),
        (separate, "separate"),
        (transcribe, "transcribe_stems"),
        (drums, "transcribe_drums"),
        (rhythm, "analyze_rhythm"),
        (keydetect, "detect_key"),
        (chords, "analyze_chords"),
    ]:
        monkeypatch.setattr(module, name, fail)

    midi_path = _write_midi(tmp_path)
    result = pipeline.run(
        midi_path,
        Options(output_dir=tmp_path / "output", work_dir=tmp_path / "work"),
    )
    assert result.status == "ok"
    assert result.device == "midi"
    assert result.bpm == pytest.approx(100.0)
    assert result.meta["input_kind"] == "midi"
    assert result.meta["rhythm"]["mode"] == "midi"
    assert result.meta["chords"] == {}
    assert result.note_counts == {"bass": 2, "other": 1, "vocals": 1, "drums": 2}
    assert (result.out_dir / "song.gp5").is_file()
    assert (result.out_dir / "midi" / "reference.mid").read_bytes() == midi_path.read_bytes()
    assert any("MIDI 输入" in warning for warning in result.warnings)


def test_midi_input_without_a_matching_selected_stem_fails_clearly(tmp_path: Path) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("program_change", program=33, channel=0, time=0))
    _note(track, 40, 0, 0, 480)
    midi_path = tmp_path / "bass-only.mid"
    mid.save(midi_path)
    with pytest.raises(pipeline.PipelineError) as caught:
        pipeline.run(
            midi_path,
            Options(
                output_dir=tmp_path / "output",
                work_dir=tmp_path / "work",
                selected_stems=("vocals",),
            ),
        )
    assert caught.value.stage == "separate"
    assert "勾选" in caught.value.message
    meta = json.loads(
        (tmp_path / "output" / caught.value.job_id / "meta.json").read_text()
    )
    assert meta["input_kind"] == "midi"
