from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import guitarpro as gp
import pytest
import soundfile as sf
from gradio_client import Client, handle_file

from audio2gp.config import TICKS_PER_BAR
from audio2gp.gp_writer import parse_gp5

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_mix(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_test_audio.py"),
            str(path),
            "--seconds",
            "3.0",
            "--bpm",
            "100",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _wait_for_server(url: str, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"web app exited early:\n{log_path.read_text(errors='replace')[-8000:]}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    pytest.fail(f"web app did not become ready:\n{log_path.read_text(errors='replace')[-8000:]}")


def _status_name(job) -> str:
    code = job.status().code
    return str(getattr(code, "name", code))


def test_gradio_convert_and_single_job_queue(tmp_path: Path) -> None:
    mix = tmp_path / "mix.wav"
    _make_mix(mix)
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/"
    log_path = tmp_path / "server.log"
    env = os.environ.copy()
    env.update(
        {
            "AUDIO2GP_OUTPUT_DIR": str(tmp_path / "output"),
            "AUDIO2GP_WORK_DIR": str(tmp_path / "work"),
            "GRADIO_ANALYTICS_ENABLED": "False",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "audio2gp.web_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    client: Client | None = None
    try:
        _wait_for_server(base_url, process, log_path)
        client = Client(base_url, verbose=False)
        args = (
            handle_file(str(mix)),
            "htdemucs",
            16,
            "other",
            15,
            0,
            True,
            True,
            True,
            "auto",
            ["bass", "other"],
        )
        first = client.submit(*args, api_name="/convert")
        second = client.submit(*args, api_name="/convert")

        # Gradio does not guarantee which submission is served first; with a
        # concurrency limit of one, exactly one may be active while the other
        # waits in the queue.
        saw_serialized_jobs = False
        active = {"STARTING", "PROGRESS", "ITERATING"}
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            first_status = _status_name(first)
            second_status = _status_name(second)
            if (
                first_status in active
                and second_status == "IN_QUEUE"
                or second_status in active
                and first_status == "IN_QUEUE"
            ):
                saw_serialized_jobs = True
                break
            if first_status in {"FINISHED", "ERROR", "CANCELLED"} and second_status in {
                "FINISHED",
                "ERROR",
                "CANCELLED",
            }:
                break
            time.sleep(0.05)
        assert saw_serialized_jobs, (
            f"jobs did not serialize: first={_status_name(first)}, "
            f"second={_status_name(second)}"
        )

        result = first.result(timeout=180.0)
        second_result = second.result(timeout=180.0)
        assert "`ok`" in second_result[6]
        assert len(result) == 8

        # Only the selected source stems (bass + guitar/other) are produced.
        audio_paths = result[:4]
        assert audio_paths[0] is None and audio_paths[1] is None
        assert audio_paths[2] and audio_paths[3]
        for audio_path in audio_paths:
            if audio_path is None:
                continue
            path = Path(audio_path)
            assert path.is_file() and path.stat().st_size > 0
            assert sf.info(path).frames > 0

        file_paths = [Path(path) for path in result[4]]
        assert len(file_paths) == 4
        assert all(path.is_file() and path.stat().st_size > 0 for path in file_paths)
        suffixes = sorted(path.suffix for path in file_paths)
        assert suffixes.count(".mid") == 3
        assert suffixes.count(".gp5") == 1
        gp5_path = next(path for path in file_paths if path.suffix == ".gp5")

        zip_info = result[5]
        assert isinstance(zip_info, dict) and zip_info.get("value")
        zip_path = Path(zip_info["value"])
        assert zip_path.is_file() and zip_path.stat().st_size > 0
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
        assert {"meta.json", "song.gp5", "midi/full.mid"} <= names
        assert {"midi/bass.mid", "midi/other.mid"} <= names
        assert {"stems/bass.wav", "stems/other.wav"} <= names
        assert not {"midi/vocals.mid", "midi/drums.mid"} & names
        assert not {"stems/vocals.wav", "stems/drums.wav"} & names

        summary, log = result[6], result[7]
        assert "`ok`" in summary
        assert "GP5" in summary
        assert "小节对齐" in summary
        assert "分轨" in summary
        assert "贝斯" in summary and "吉他/其他" in summary
        assert "完成：ok" in log

        song = parse_gp5(gp5_path)
        assert [track.name for track in song.tracks] == ["Guitar", "Bass"]
        for track in song.tracks:
            for measure in track.measures:
                assert len(measure.voices) == 2
                assert (
                    sum(beat.duration.time for beat in measure.voices[0].beats)
                    == TICKS_PER_BAR
                )
                assert all(
                    beat.status != gp.BeatStatus.empty
                    for beat in measure.voices[0].beats
                )
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_handle.close()
