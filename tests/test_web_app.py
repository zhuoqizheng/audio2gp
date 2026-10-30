from __future__ import annotations

import threading
import time
from pathlib import Path


def test_second_job_is_rejected_while_lock_is_held(tmp_path: Path) -> None:
    import audio2gp.web_app as web

    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"not actually decoded while the lock is held")

    assert web.JOB_LOCK.acquire(blocking=False)
    try:
        states = list(
            web._run_job(
                str(input_path),
                "htdemucs",
                16,
                "other",
                15,
                0,
                True,
                True,
                True,
                "cpu",
                ["vocals", "drums", "bass", "other"],
            )
        )
    finally:
        web.JOB_LOCK.release()

    assert len(states) == 1
    summary, log = states[0][-2], states[0][-1]
    assert "已有任务正在运行" in summary
    assert "已有任务正在运行" in log
    assert web.JOB_LOCK.locked() is False


def test_disconnect_keeps_the_lock_until_the_worker_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    import audio2gp.web_app as web

    started = threading.Event()
    release = threading.Event()

    def fake_run(path, options, progress_cb=None):
        started.set()
        release.wait(5.0)
        return None

    monkeypatch.setattr(web, "run", fake_run)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"fake")

    states = web._run_job(
        str(input_path),
        "htdemucs",
        16,
        "other",
        15,
        0,
        True,
        True,
        True,
        "cpu",
        ["other"],
    )
    try:
        next(states)
        assert started.wait(2.0)
        assert web.JOB_LOCK.locked()

        states.close()  # a Gradio client closed the tab
        assert web.JOB_LOCK.locked(), "lock released while the job was still running"

        release.set()
        deadline = time.monotonic() + 5.0
        while web.JOB_LOCK.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert web.JOB_LOCK.locked() is False
    finally:
        release.set()


def test_main_refuses_a_non_loopback_host(capsys) -> None:
    import audio2gp.web_app as web

    assert web.main(["--host", "0.0.0.0"]) == 2
    assert "localhost-only" in capsys.readouterr().out
