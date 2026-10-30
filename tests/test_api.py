from __future__ import annotations

import io
import json
import logging
import shutil
import time
import wave
from pathlib import Path
from typing import Any, Callable

import guitarpro as gp
import mido
import pytest
from fastapi.testclient import TestClient

from audio2gp import score_ir
from audio2gp.api import create_app
from audio2gp.config import TUNING_BASS, TUNING_GUITAR
from audio2gp.types import FingeredNote, Note, TabTrack
from audio2gp.utils import read_json, write_json, zip_directory


def _tracks() -> list[TabTrack]:
    guitar = TabTrack(
        name="Guitar",
        stem="other",
        tuning=[TUNING_GUITAR[i] for i in range(1, 7)],
        notes=[
            FingeredNote(0, 960, 64, 90, 1, 0),
            FingeredNote(0, 960, 59, 90, 2, 0),
            FingeredNote(960, 1920, 67, 90, 1, 3),
        ],
        channel=0,
        instrument=27,
    )
    bass = TabTrack(
        name="Bass",
        stem="bass",
        tuning=[TUNING_BASS[i] for i in range(1, 5)],
        notes=[FingeredNote(0, 960, 28, 90, 4, 0)],
        channel=2,
        instrument=33,
    )
    drums = TabTrack(
        name="Drums",
        stem="drums",
        tuning=[],
        is_percussion=True,
        channel=9,
        instrument=0,
        drum_notes=[Note(0, 240, 36, 100, "drums")],
    )
    return [guitar, bass, drums]


def _make_runner(record_calls: list[dict[str, Any]] | None = None) -> Callable[..., Any]:
    def runner(input_path: Path, options: Any, progress_cb=None, job_id=None) -> None:
        if record_calls is not None:
            record_calls.append({"input": Path(input_path), "options": options, "job_id": job_id})
        out_dir = Path(options.output_dir) / str(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, pct, message in [
            ("preprocess", 2.0, "normalize"),
            ("separate", 20.0, "demucs"),
            ("transcribe", 60.0, "notes"),
            ("post", 85.0, "quantize"),
            ("gp5", 95.0, "write"),
        ]:
            if progress_cb:
                progress_cb(name, pct, message)
        state = score_ir.build_state(
            _tracks(),
            bpm=120.0,
            measure_count=1,
            job_id=str(job_id),
            quantize=16,
            layout="auto",
            max_fret=15,
            key={"name": "C major", "gp_key": "CMajor"},
            time_signature=options.time_signature,
        )
        committed = score_ir.commit_state(state, out_dir)
        # The initial pipeline state published for editing starts at revision
        # zero; commit_state is only used here to obtain real GP5/MIDI files.
        initial = committed["state"]
        initial["revision"] = 0
        initial.pop("edited_at", None)
        write_json(out_dir / "tracks.json", initial)
        stems = out_dir / "stems"
        stems.mkdir(parents=True, exist_ok=True)
        for stem in ("other", "bass", "drums", "vocals"):
            shutil.copyfile(input_path, stems / f"{stem}.wav")
        meta = read_json(out_dir / "meta.json")
        meta.update(
            {
                "status": "ok",
                "revision": 0,
                "audio": {"bpm": 120.0, "duration_s": 0.2},
                "device": "cpu",
                "selected_stems": ["other", "bass", "drums"],
                "note_counts": {"other": 3, "bass": 1, "drums": 1},
                "warnings": ["fake warning"],
                "key": {"name": "C major"},
                "measure_count": 1,
                "tracks": ["Guitar", "Bass", "Drums"],
                "time_signature": options.time_signature,
                "bar_ticks": options.meter.bar_ticks,
                "options": options.to_meta(),
                "rhythm": {
                    "time_signature": options.time_signature,
                    "mode": "manual" if options.manual_downbeat_s else "off",
                    "enabled": options.align_downbeats,
                    "first_downbeat_s": options.manual_downbeat_s,
                },
            }
        )
        meta["files"]["stems"] = {
            stem: f"stems/{stem}.wav" for stem in ("other", "bass", "drums", "vocals")
        }
        write_json(out_dir / "meta.json", meta)
        zip_directory(out_dir, out_dir / str(meta["files"]["zip"]))

    return runner


def _make_midi_only_runner() -> Callable[..., Any]:
    def runner(input_path: Path, options: Any, progress_cb=None, job_id=None) -> None:
        out_dir = Path(options.output_dir) / str(job_id)
        (out_dir / "midi").mkdir(parents=True, exist_ok=True)
        for name in ("other", "bass", "full"):
            (out_dir / "midi" / f"{name}.mid").write_bytes(b"MThd")
        chords = {
            "version": 1,
            "method": "fake",
            "bpm": 120.0,
            "segments": [
                {
                    "start_tick": 0,
                    "end_tick": 1920,
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "label": "C",
                    "root": 0,
                    "quality": "major",
                    "confidence": 0.8,
                    "bar": 1,
                    "beat": 1,
                }
            ],
            "summary": {
                "chord_count": 1,
                "unique_chords": ["C"],
                "quality_counts": {"major": 1},
                "progression": ["C"],
            },
        }
        write_json(out_dir / "chords.json", chords)
        zip_name = f"audio2gp_{job_id}.zip"
        meta = {
            "job_id": str(job_id),
            "status": "ok",
            "output_format": "midi_only",
            "options": {"detect_chords": True, "export_gp5": False},
            "selected_stems": ["other", "bass"],
            "tracks": [],
            "audio": {"bpm": 120.0, "duration_s": 1.0},
            "warnings": [],
            "chords": chords,
            "files": {
                "midi": {"other": "midi/other.mid", "bass": "midi/bass.mid"},
                "full_midi": "midi/full.mid",
                "chords": "chords.json",
                "meta": "meta.json",
                "zip": zip_name,
            },
        }
        write_json(out_dir / "meta.json", meta)
        zip_directory(out_dir, out_dir / zip_name)

    return runner


def _wav_bytes(seconds: float = 0.2, *, samples: int = 8000) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(samples)
        handle.writeframes(b"\x00\x00" * int(seconds * samples))
    return buffer.getvalue()


def _client(tmp_path: Path, **kwargs: Any) -> TestClient:
    app = create_app(
        tmp_path / "output",
        tmp_path / "work",
        runner=_make_runner(),
        **kwargs,
    )
    return TestClient(app)


def _create_job(
    client: TestClient,
    *,
    options: dict[str, Any] | None = None,
    filename: str = "song.wav",
    data: bytes | None = None,
) -> dict:
    response = client.post(
        "/api/jobs",
        files={"file": (filename, data if data is not None else _wav_bytes(), "audio/wav")},
        data={"options": json.dumps(options or {})},
    )
    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["id"]
    return payload


def _wait_done(client: TestClient, job_id: str, timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"done", "error"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return _client(tmp_path)


def test_static_ui_health_and_limits(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "audio2gp" in index.text
    assert "/static/app.js" in index.text

    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/vendor/alphatab/alphaTab.min.js").status_code == 200

    health = client.get("/api/health")
    assert health.json()["ok"] is True

    limits = client.get("/api/limits").json()
    assert limits["max_upload_mb"] >= 1
    assert 16 in limits["quantize"]
    assert "standard" in limits["guitar_tunings"]
    assert limits["output_formats"] == ["gp5_midi", "midi_only"]
    assert limits["chord_analysis"] is True
    assert [entry["value"] for entry in limits["time_signatures"]] == [
        "4/4",
        "3/4",
        "2/4",
        "6/8",
        "9/8",
        "12/8",
    ]
    assert limits["manual_downbeat"] is True
    assert limits["auto_downbeat_time_signatures"] == [
        "4/4",
        "3/4",
        "2/4",
        "6/8",
        "9/8",
        "12/8",
    ]
    assert limits["transcription_engines"] == ["basic_pitch", "external"]
    assert "exportFormatSelect" in index.text
    assert "detectChords" in index.text
    assert "chordPanel" in index.text
    assert "仅 MIDI" in index.text
    assert "和弦分析" in index.text
    assert "timeSignatureSelect" in index.text
    assert "manualDownbeat" in index.text
    assert "不做自动拍号识别" in index.text


def test_job_lifecycle_outputs_and_downloads(client: TestClient) -> None:
    created = _create_job(client)
    job = _wait_done(client, created["id"])
    assert job["status"] == "done", job
    assert job["progress"] == 100.0
    assert job["partial"] is False
    assert job["warnings"] == ["fake warning"]
    result = job["result"]
    assert result["can_edit"] is True
    assert result["bpm"] == 120.0
    assert result["revision"] == 0
    assert [track["name"] for track in result["tracks"]] == ["Guitar", "Bass", "Drums"]
    assert set(result["files"]["midi"]) == {"other", "bass", "drums"}
    assert set(result["audio"]) == {"other", "bass", "drums", "vocals"}

    state_response = client.get(result["state_url"])
    assert state_response.status_code == 200
    state = state_response.json()
    assert score_ir.get_track(state, "guitar")["notes"][0]["id"] == "n000001"

    score = client.get(result["score_url"])
    assert score.status_code == 200
    import io

    song = gp.parse(io.BytesIO(score.content))
    assert len(song.measureHeaders) == 1
    assert [track.name for track in song.tracks] == ["Guitar", "Bass", "Drums"]

    full_midi = client.get(result["files"]["full_midi"])
    assert full_midi.status_code == 200
    assert full_midi.content[:4] == b"MThd"

    stem = client.get(result["audio"]["bass"])
    assert stem.status_code == 200
    assert stem.headers["cache-control"] == "no-store"

    zip_response = client.get(result["files"]["zip"])
    assert zip_response.status_code == 200
    assert zip_response.content[:2] == b"PK"

    assert client.get(result["files"]["meta"]).status_code == 200
    assert client.get(result["files"]["tracks"]).status_code == 200


def test_midi_only_result_exposes_chords_without_score_editor(tmp_path: Path) -> None:
    app = create_app(
        tmp_path / "output",
        tmp_path / "work",
        runner=_make_midi_only_runner(),
    )
    client = TestClient(app)
    created = _create_job(
        client,
        options={
            "selected_stems": ["other", "bass"],
            "export_gp5": False,
            "detect_chords": True,
        },
    )
    job = _wait_done(client, created["id"])
    assert job["status"] == "done", job
    assert job["message"] == "完成（仅 MIDI）"
    result = job["result"]
    assert result["output_format"] == "midi_only"
    assert result["can_edit"] is False
    assert result["state_url"] is None
    assert result["score_url"] is None
    assert result["tracks"] == []
    assert result["chord_detection"] is True
    assert [segment["label"] for segment in result["chords"]] == ["C"]
    assert result["chord_summary"]["progression"] == ["C"]
    assert "gp5" not in result["files"]
    assert "tracks" not in result["files"]
    assert set(result["files"]["midi"]) == {"other", "bass"}
    assert client.get(result["files"]["chords"]).status_code == 200
    chords = client.get(result["files"]["chords"]).json()
    assert chords["segments"][0]["label"] == "C"
    assert client.get(result["files"]["zip"]).status_code == 200
    assert client.get(f"/api/jobs/{created['id']}/state").status_code == 404
    assert client.get(f"/api/jobs/{created['id']}/score.gp5").status_code == 404


def test_edit_delete_mute_and_revision_conflict(client: TestClient) -> None:
    created = _create_job(client)
    job = _wait_done(client, created["id"])
    job_id = created["id"]
    base = f"/api/jobs/{job_id}"
    state = client.get(f"{base}/state").json()
    assert state["revision"] == 0

    moved = client.patch(
        f"{base}/tracks/guitar/notes/n000003",
        json={"revision": 0, "string": 2},
    )
    assert moved.status_code == 200, moved.text
    payload = moved.json()
    assert payload["revision"] == 1
    note = score_ir.get_note(
        score_ir.get_track(payload["state"], "guitar"), "n000003"
    )
    assert (note["string"], note["fret"], note["pitch"]) == (2, 8, 67)
    assert payload["job"]["result"]["revision"] == 1

    stale = client.patch(
        f"{base}/tracks/guitar/notes/n000003",
        json={"revision": 0, "string": 1},
    )
    assert stale.status_code == 409
    assert "刷新" in stale.json()["detail"]

    deleted = client.delete(
        f"{base}/tracks/guitar/notes/n000002",
        params={"revision": 1},
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["revision"] == 2
    guitar_notes = score_ir.get_track(deleted.json()["state"], "guitar")["notes"]
    assert [note["id"] for note in guitar_notes] == ["n000001", "n000003"]

    muted = client.put(
        f"{base}/tracks/bass/mute",
        json={"revision": 2, "muted": True},
    )
    assert muted.status_code == 200, muted.text
    assert muted.json()["revision"] == 3
    assert score_ir.get_track(muted.json()["state"], "bass")["muted"] is True

    # Edits are visible in the exported GP5 and in the refreshed job result.
    score = client.get(f"{base}/score.gp5")
    assert score.status_code == 200
    refreshed = client.get(f"{base}").json()
    assert refreshed["result"]["revision"] == 3
    assert [track["muted"] for track in refreshed["result"]["tracks"]] == [
        False,
        True,
        False,
    ]
    downloads = refreshed["result"]["files"]
    assert downloads["gp5"].startswith(f"/api/jobs/{job_id}/")
    assert client.get(downloads["gp5"]).status_code == 200


def test_transpose_undo_and_flags(client: TestClient) -> None:
    created = _create_job(client)
    _wait_done(client, created["id"])
    base = f"/api/jobs/{created['id']}"
    original = client.get(f"{base}/state").json()
    original_pitches = [
        note["pitch"] for note in score_ir.get_track(original, "guitar")["notes"]
    ]

    transposed = client.post(
        f"{base}/tracks/guitar/transpose",
        json={"revision": 0, "semitones": 12},
    )
    assert transposed.status_code == 200, transposed.text
    assert transposed.json()["revision"] == 1
    assert transposed.json()["changed"] == 3
    assert transposed.json()["can_undo"] is True
    assert [
        note["pitch"]
        for note in score_ir.get_track(transposed.json()["state"], "guitar")["notes"]
    ] == [pitch + 12 for pitch in original_pitches]

    undone = client.post(f"{base}/undo", json={"revision": 1})
    assert undone.status_code == 200, undone.text
    assert undone.json()["revision"] == 2
    assert [
        note["pitch"] for note in score_ir.get_track(undone.json()["state"], "guitar")["notes"]
    ] == original_pitches
    assert undone.json()["can_undo"] is False
    assert client.post(f"{base}/undo", json={"revision": 2}).status_code == 422

    flagged = client.put(
        f"{base}/tracks/guitar/notes/n000003/flag",
        json={"flagged": True, "reason": "wrong octave?"},
    )
    assert flagged.status_code == 200, flagged.text
    assert flagged.json()["flag_count"] == 1
    flag = flagged.json()["flags"][0]
    assert (flag["note_id"], flag["reason"]) == ("n000003", "wrong octave?")
    assert flag["bar"] == 1
    flags_url = flagged.json()["files"]["flags"]
    assert client.get(flags_url).json()["flags"][0]["note_id"] == "n000003"
    refreshed = client.get(base).json()
    assert refreshed["result"]["flag_count"] == 1
    assert refreshed["result"]["flags"][0]["reason"] == "wrong octave?"

    unflagged = client.put(
        f"{base}/tracks/guitar/notes/n000003/flag",
        json={"flagged": False},
    )
    assert unflagged.status_code == 200
    assert unflagged.json()["flag_count"] == 0
    assert client.get(flags_url).json()["flags"] == []


def test_transpose_rejects_drums_and_bad_values(client: TestClient) -> None:
    created = _create_job(client)
    _wait_done(client, created["id"])
    base = f"/api/jobs/{created['id']}"
    assert client.post(
        f"{base}/tracks/drums/transpose",
        json={"revision": 0, "semitones": 12},
    ).status_code == 422
    assert client.post(
        f"{base}/tracks/guitar/transpose",
        json={"revision": 0, "semitones": 0},
    ).status_code == 422
    assert client.post(
        f"{base}/tracks/guitar/transpose",
        json={"revision": 0, "semitones": 25},
    ).status_code == 422


def test_invalid_edits_are_rejected(client: TestClient) -> None:
    created = _create_job(client)
    _wait_done(client, created["id"])
    base = f"/api/jobs/{created['id']}"
    missing = client.patch(
        f"{base}/tracks/guitar/notes/missing",
        json={"revision": 0, "fret": 1},
    )
    assert missing.status_code == 422
    collision = client.patch(
        f"{base}/tracks/guitar/notes/n000002",
        json={"revision": 0, "string": 1},
    )
    assert collision.status_code == 422
    no_change = client.patch(
        f"{base}/tracks/guitar/notes/n000001",
        json={"revision": 0},
    )
    assert no_change.status_code == 422
    drum_edit = client.patch(
        f"{base}/tracks/drums/notes/n000005",
        json={"revision": 0, "string": 1},
    )
    assert drum_edit.status_code == 422
    assert client.post(
        f"{base}/tracks/guitar/notes/n000001",
        json={"revision": 0, "fret": 1},
    ).status_code == 405


def test_option_and_upload_validation(tmp_path: Path) -> None:
    client = _client(tmp_path, max_upload_mb=1, max_duration_s=0.1)

    unknown = client.post(
        "/api/jobs",
        files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
        data={"options": json.dumps({"not_an_option": True})},
    )
    assert unknown.status_code == 422

    no_stems = client.post(
        "/api/jobs",
        files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
        data={"options": json.dumps({"selected_stems": []})},
    )
    assert no_stems.status_code == 422

    no_gp5_track = client.post(
        "/api/jobs",
        files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
        data={
            "options": json.dumps(
                {
                    "selected_stems": ["vocals"],
                    "guitar_stem": "other",
                    "include_vocals": False,
                }
            )
        },
    )
    assert no_gp5_track.status_code == 422

    bad_ext = client.post(
        "/api/jobs",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"options": "{}"},
    )
    assert bad_ext.status_code == 422

    empty = client.post(
        "/api/jobs",
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"options": "{}"},
    )
    assert empty.status_code == 422

    too_large = client.post(
        "/api/jobs",
        files={"file": ("big.wav", b"x" * (1024 * 1024 + 1), "audio/wav")},
        data={"options": "{}"},
    )
    assert too_large.status_code == 413

    too_long = client.post(
        "/api/jobs",
        files={"file": ("long.wav", _wav_bytes(1.0), "audio/wav")},
        data={"options": "{}"},
    )
    assert too_long.status_code == 413


def test_file_endpoint_rejects_unlisted_and_traversal_paths(client: TestClient) -> None:
    created = _create_job(client)
    _wait_done(client, created["id"])
    base = f"/api/jobs/{created['id']}/files"
    assert client.get(f"{base}/meta.json").status_code == 200
    assert client.get(f"{base}/pipeline.log").status_code == 404
    assert client.get(f"{base}/../meta.json").status_code == 404
    assert client.get(f"{base}/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get(f"{base}/midi/other.mid/extra").status_code == 404
    assert client.get(f"{base}/stems/other.wav").status_code == 200


def test_runner_receives_safe_option_values(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    app = create_app(
        tmp_path / "output",
        tmp_path / "work",
        runner=_make_runner(calls),
    )
    client = TestClient(app)
    created = _create_job(
        client,
        options={
            "guitar_tuning": "drop_d",
            "bass_tuning": "drop_d",
            "quantize": 12,
            "selected_stems": ["other", "bass"],
            "max_fret": 19,
            "bpm_override": 98.5,
            "transcription_engine": "external",
        },
    )
    job = _wait_done(client, created["id"])
    assert job["status"] == "done"
    assert len(calls) == 1
    call = calls[0]
    assert call["job_id"] == created["id"]
    assert call["options"].guitar_tuning == "drop_d"
    assert call["options"].bass_tuning == "drop_d"
    assert call["options"].quantize == 12
    assert call["options"].selected_stems == ("bass", "other")
    assert call["options"].max_fret == 19
    assert call["options"].bpm_override == 98.5
    assert call["options"].transcription_engine == "external"
    assert call["input"].is_file()
    assert call["input"].parent == tmp_path / "work" / created["id"]


def test_meter_and_manual_downbeat_are_plumbed_through(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    app = create_app(
        tmp_path / "output",
        tmp_path / "work",
        runner=_make_runner(calls),
    )
    client = TestClient(app)
    created = _create_job(
        client,
        options={
            "time_signature": "12/8",
            "manual_downbeat_s": 1.25,
            "selected_stems": ["other", "bass", "drums"],
        },
    )
    job = _wait_done(client, created["id"])
    assert job["status"] == "done", job
    assert calls[0]["options"].time_signature == "12/8"
    assert calls[0]["options"].manual_downbeat_s == 1.25
    result = job["result"]
    assert result["time_signature"] == "12/8"
    assert result["bar_ticks"] == 5760
    assert result["manual_downbeat_s"] == 1.25
    state = client.get(result["state_url"]).json()
    assert state["time_signature"] == "12/8"
    assert state["bar_ticks"] == 5760

    invalid = client.post(
        "/api/jobs",
        files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
        data={"options": json.dumps({"time_signature": "5/4"})},
    )
    assert invalid.status_code == 422

    bad_downbeat = client.post(
        "/api/jobs",
        files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
        data={"options": json.dumps({"manual_downbeat_s": -1})},
    )
    assert bad_downbeat.status_code == 422


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/state").status_code == 404
    assert client.get("/api/jobs/nope/score.gp5").status_code == 404
    assert client.get("/api/jobs/nope/audio/other").status_code == 404
    assert client.get("/api/jobs/nope/files/meta.json").status_code == 404


def test_health_does_not_import_torch(client: TestClient) -> None:
    # The API module must let the app start without a GPU/torch import at
    # module import time. The fake runner never touches the model stack.
    response = client.get("/api/health")
    assert response.json() == {"ok": True, "version": "0.2.0"}


def _midi_bytes() -> bytes:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.MetaMessage("track_name", name="Bass", time=0))
    track.append(mido.Message("program_change", program=33, channel=0, time=0))
    track.append(mido.Message("note_on", note=40, velocity=100, channel=0, time=0))
    track.append(mido.Message("note_off", note=40, velocity=0, channel=0, time=480))
    buffer = io.BytesIO()
    mid.save(file=buffer)
    return buffer.getvalue()


def test_midi_upload_uses_the_midi_pipeline(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "output", tmp_path / "work"))
    response = client.post(
        "/api/jobs",
        files={"file": ("song.mid", _midi_bytes(), "audio/midi")},
        data={"options": json.dumps({"selected_stems": ["bass"]})},
    )
    assert response.status_code == 202, response.text
    job = _wait_done(client, response.json()["id"])
    assert job["status"] == "done", job
    result = job["result"]
    assert result["input_kind"] == "midi"
    assert result["device"] == "midi"
    assert result["score_url"]
    assert result["note_counts"] == {"bass": 1}


def test_completed_jobs_survive_restart_and_bad_dirs_are_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "output"
    job_dir = output / "job-midi"
    job_dir.mkdir(parents=True)
    write_json(
        job_dir / "meta.json",
        {
            "job_id": "job-midi",
            "status": "ok",
            "created": "2026-01-01T00:00:00+00:00",
            "input": {"name": "song.mid"},
            "input_kind": "midi",
            "audio": {"bpm": 120.0, "duration_s": 2.0},
            "options": {},
            "warnings": [],
            "tracks": ["Bass"],
        },
    )
    broken = output / "broken"
    broken.mkdir()
    (broken / "meta.json").write_text("{not json", encoding="utf-8")
    interrupted = output / "interrupted"
    interrupted.mkdir()
    (interrupted / "chords.json").write_text("[]", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="audio2gp.api"):
        client = TestClient(create_app(output, tmp_path / "work"))
    assert "interrupted" not in caplog.text
    job = client.get("/api/jobs/job-midi")
    assert job.status_code == 200
    payload = job.json()
    assert payload["status"] == "done"
    assert payload["original_name"] == "song.mid"
    assert payload["result"]["input_kind"] == "midi"
    assert client.get("/api/jobs/broken").status_code == 404
