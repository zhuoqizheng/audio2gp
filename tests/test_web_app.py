from __future__ import annotations

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
