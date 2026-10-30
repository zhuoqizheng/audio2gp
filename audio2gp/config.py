from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def loopback_host_error(host: str) -> str | None:
    """Return a refusal message unless host is a loopback address."""
    if host in LOOPBACK_HOSTS:
        return None
    return f"Refusing to bind {host!r}: audio2gp is localhost-only. Use 127.0.0.1."


# General MIDI programs used by the generated files.
PROGRAMS = {
    "guitar": 27,       # Electric Guitar (clean)
    "bass": 33,         # Electric Bass (finger)
    "vocals": 53,       # Voice Oohs
    "drums": 0,
}

# String 1 is the highest-pitched string in Guitar Pro. The names are the
# UI/CLI values; each preset maps string number -> open MIDI pitch.
TUNING_GUITAR = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}  # E4 B3 G3 D3 A2 E2
TUNING_BASS = {1: 43, 2: 38, 3: 33, 4: 28}                    # G2 D2 A1 E1
GUITAR_TUNINGS: dict[str, dict[int, int]] = {
    "standard": TUNING_GUITAR,
    "drop_d": {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 38},       # D2 A2 D3 G3 B3 E4
    "half_step_down": {1: 63, 2: 58, 3: 54, 4: 49, 5: 44, 6: 39},
}
BASS_TUNINGS: dict[str, dict[int, int]] = {
    "standard": TUNING_BASS,
    "drop_d": {1: 43, 2: 38, 3: 33, 4: 26},                     # D1 A1 D2 G2
}
GUITAR_TUNING_LABELS = {
    "standard": "标准 EADGBE",
    "drop_d": "Drop D",
    "half_step_down": "降半音",
}
BASS_TUNING_LABELS = {
    "standard": "标准 4 弦 EADG",
    "drop_d": "Drop D (DADG)",
}

STEM_NAMES = ("vocals", "drums", "bass", "other")
MELODIC_STEMS = ("vocals", "bass", "other")
# 12 = 1/8 triplet (bar / 12), in addition to binary grids.
QUANTIZE_VALUES = (8, 12, 16, 32)

TICKS_PER_QUARTER = 960
# 4/4 reference bar. Meter.bar_ticks is authoritative for the selected meter.
TICKS_PER_BAR = 3840
BEATS_PER_BAR = 4

# The first release of user-selected meters. They cover the common song
# signatures, including the compound meters used for shuffle/blues feels.
TIME_SIGNATURE_CHOICES = ("4/4", "3/4", "2/4", "6/8", "9/8", "12/8")
TIME_SIGNATURE_LABELS = {
    "4/4": "4/4（最常用）",
    "3/4": "3/4（三拍子 / 华尔兹）",
    "2/4": "2/4（进行曲 / 二拍子）",
    "6/8": "6/8（复合二拍，每拍三个八分）",
    "9/8": "9/8（复合三拍）",
    "12/8": "12/8（慢布鲁斯 / shuffle，四拍三连音感）",
}
_METER_DENOMINATORS = {2, 4, 8, 16}


@dataclass(frozen=True)
class Meter:
    """A user-selected time signature and its tick geometry.

    BPM is always quarter-note BPM, as in MIDI. The bar length therefore only
    depends on the numerator and denominator:

        bar_ticks = numerator * (4 * 960 / denominator)
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not 1 <= int(self.numerator) <= 16:
            raise ValueError("time signature numerator must be between 1 and 16")
        if int(self.denominator) not in _METER_DENOMINATORS:
            raise ValueError(
                "time signature denominator must be one of "
                + ", ".join(str(value) for value in sorted(_METER_DENOMINATORS))
            )

    @property
    def label(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def bar_ticks(self) -> int:
        return int(self.numerator) * (4 * TICKS_PER_QUARTER) // int(self.denominator)

    @property
    def bar_quarters(self) -> float:
        return float(self.numerator) * 4.0 / float(self.denominator)

    @property
    def compound(self) -> bool:
        # 6/8, 9/8, 12/8: each musical beat is a dotted quarter.
        return (
            self.denominator == 8
            and self.numerator >= 6
            and self.numerator % 3 == 0
        )

    @property
    def pulses_per_bar(self) -> int:
        """Musical beats per bar (compound meters group three eighths)."""
        return self.numerator // 3 if self.compound else self.numerator

    @property
    def pulse_ticks(self) -> int:
        return self.bar_ticks // self.pulses_per_bar

    @property
    def beams(self) -> tuple[int, int, int, int]:
        """Best-effort GP5 beam grouping values for the four stored slots."""
        group = 3 if self.compound else 2
        values = [group] * self.pulses_per_bar
        values = (values + [0, 0, 0, 0])[:4]
        return tuple(values)  # type: ignore[return-value]

    def bar_seconds(self, bpm: float) -> float:
        return (60.0 / float(bpm)) * self.bar_quarters


def parse_time_signature(value: str | Meter) -> Meter:
    if isinstance(value, Meter):
        return value
    text = str(value or "").strip()
    if "/" not in text:
        raise ValueError(
            "time_signature must look like 4/4, one of "
            + ", ".join(TIME_SIGNATURE_CHOICES)
        )
    numerator, denominator = text.split("/", 1)
    try:
        meter = Meter(int(numerator), int(denominator))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time_signature must be a supported signature: "
            + ", ".join(TIME_SIGNATURE_CHOICES)
        ) from exc
    if meter.label not in TIME_SIGNATURE_CHOICES:
        raise ValueError(
            "time_signature must be one of " + ", ".join(TIME_SIGNATURE_CHOICES)
        )
    return meter

# (duration ticks, GuitarPro Duration value, dotted, tuplet enters, tuplet times)
# Tuplets use Guitar Pro's 3-in-the-time-of-2 notation for the 1/8-triplet grid.
DURATION_SPECS = (
    (5760, 1, True, 1, 1),
    (3840, 1, False, 1, 1),
    (2880, 2, True, 1, 1),
    (1920, 2, False, 1, 1),
    (1440, 4, True, 1, 1),
    (1280, 2, False, 3, 2),
    (960, 4, False, 1, 1),
    (720, 8, True, 1, 1),
    (640, 4, False, 3, 2),
    (480, 8, False, 1, 1),
    (360, 16, True, 1, 1),
    (320, 8, False, 3, 2),
    (240, 16, False, 1, 1),
    (180, 32, True, 1, 1),
    (160, 16, False, 3, 2),
    (120, 32, False, 1, 1),
    (80, 32, False, 3, 2),
)
REPRESENTABLE_TICKS = tuple(spec[0] for spec in DURATION_SPECS)
VELOCITY_STEPS = (15, 31, 47, 63, 79, 95, 111, 127)

# Preferred percussion lanes (string numbers) in the six-lane GP drum staff.
DRUM_LANE_BY_PITCH = {
    36: 1,  # kick
    38: 2,  # snare
    42: 3,  # closed hi-hat
    46: 4,  # open hi-hat
    49: 5,  # crash
    41: 6, 43: 6, 45: 6, 47: 6, 48: 6, 50: 6,  # toms share a fallback lane
}


def grid_ticks(quantize: int) -> int:
    if quantize not in QUANTIZE_VALUES:
        raise ValueError(f"quantize must be one of {QUANTIZE_VALUES}")
    return TICKS_PER_BAR // quantize


def grid_seconds(bpm: float, quantize: int) -> float:
    # The grid names a note value, not a fraction of the bar, so it is
    # independent of the selected time signature.
    return (grid_ticks(quantize) / TICKS_PER_QUARTER) * (60.0 / float(bpm))


def bar_seconds(
    bpm: float,
    beats_per_bar: int = BEATS_PER_BAR,
    *,
    denominator: int = 4,
) -> float:
    """Seconds per bar for a numerator/denominator meter (quarter-note BPM)."""
    return (60.0 / float(bpm)) * (float(beats_per_bar) * 4.0 / float(denominator))


def ticks_to_seconds(tick: int, bpm: float) -> float:
    return (tick / TICKS_PER_QUARTER) * (60.0 / float(bpm))


@dataclass
class Options:
    output_dir: Path = Path("output")
    work_dir: Path = Path("work")
    demucs_model: str = "htdemucs_ft"
    # basic_pitch = built-in TFLite model; external = command in
    # AUDIO2GP_EXTERNAL_CMD (see audio2gp.engines).
    transcription_engine: str = "basic_pitch"
    quantize: int = 16
    guitar_stem: str = "other"
    keep_stems: bool = True
    max_fret: int = 15
    min_note_sec: float = 0.03
    device: str = "auto"
    include_vocals: bool = True
    bpm_override: float | None = None
    # Manual meter. This build deliberately has no automatic time-signature
    # detection; the user chooses the signature.
    time_signature: str = "4/4"
    # Optional manual first downbeat in source-audio seconds. When set it
    # replaces automatic downbeat analysis and works with any meter.
    manual_downbeat_s: float | None = None
    # Automatic downbeat analysis is meter-aware: madmom for 4/4, librosa
    # accent phase for the other user-selected signatures. A manual downbeat
    # always wins over the automatic result.
    align_downbeats: bool = True
    guitar_tuning: str = "standard"
    bass_tuning: str = "standard"
    # When enabled, every preset is fingered and the one that preserves the
    # most original pitches (then the lowest frets) wins. Off by default so a
    # chosen tuning is never silently changed.
    suggest_tuning: bool = False
    # AI-generated mixes often contain ghost octaves and dense spill from the
    # separator. These are deliberately conservative defaults.
    ghost_amplitude: float = 0.15
    max_polyphony: int = 6
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    detect_techniques: bool = True
    detect_key: bool = True
    # Chord labels are a draft analysis from the harmonic stems. They are
    # independent of whether a GP5 is exported.
    detect_chords: bool = True
    # False = MIDI-only. MIDI, stems, chords, meta, and ZIP are still written;
    # fingering, GP5 construction, and the browser score editor are skipped.
    export_gp5: bool = True
    tempo_from_drums: bool = True
    cache_stems: bool = True
    # Source stems to transcribe and deliver. Demucs still separates all four
    # internally; unselected stems are not transcribed, written to MIDI/GP5, or
    # copied to output/zip. This lets a guitar+vocal recording produce just two
    # tracks instead of four (mostly empty) ones.
    selected_stems: tuple[str, ...] = STEM_NAMES
    layout: str = "auto"  # auto | onset | ties
    host: str = "127.0.0.1"
    port: int = 7860
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.work_dir = Path(self.work_dir)
        if self.bpm_override is not None and self.bpm_override <= 0:
            self.bpm_override = None
        if self.manual_downbeat_s is not None and float(self.manual_downbeat_s) <= 0:
            self.manual_downbeat_s = None
        self.time_signature = parse_time_signature(self.time_signature).label
        raw_stems = self.selected_stems
        if isinstance(raw_stems, str):
            raw_stems = (raw_stems,)
        selected = {str(name) for name in (raw_stems or ())}
        unknown = selected - set(STEM_NAMES)
        if unknown:
            raise ValueError(
                "selected_stems contains unknown values: "
                + ", ".join(sorted(unknown))
            )
        self.selected_stems = tuple(name for name in STEM_NAMES if name in selected)
        self.validate()

    def gp5_track_names(self) -> tuple[str, ...]:
        """Tracks that the selected stems can produce, in GP5 order."""
        names: list[str] = []
        if self.guitar_stem in self.selected_stems:
            names.append("Guitar")
        if "bass" in self.selected_stems:
            names.append("Bass")
        if "drums" in self.selected_stems:
            names.append("Drums")
        if (
            self.include_vocals
            and "vocals" in self.selected_stems
            and self.guitar_stem != "vocals"
        ):
            names.append("Vocals")
        return tuple(names)

    def validate(self) -> None:
        if self.quantize not in QUANTIZE_VALUES:
            raise ValueError(f"quantize must be one of {QUANTIZE_VALUES}")
        if self.bpm_override is not None and not 20 <= float(self.bpm_override) <= 300:
            raise ValueError("bpm_override must be between 20 and 300")
        if self.guitar_stem not in {"other", "vocals"}:
            raise ValueError("guitar_stem must be 'other' or 'vocals'")
        if not 0 <= int(self.max_fret) <= 24:
            raise ValueError("max_fret must be between 0 and 24")
        if self.min_note_sec < 0:
            raise ValueError("min_note_sec must be >= 0")
        if self.guitar_tuning not in GUITAR_TUNINGS:
            raise ValueError(
                "guitar_tuning must be one of " + ", ".join(GUITAR_TUNINGS)
            )
        if self.bass_tuning not in BASS_TUNINGS:
            raise ValueError(
                "bass_tuning must be one of " + ", ".join(BASS_TUNINGS)
            )
        if not 0.0 <= float(self.ghost_amplitude) <= 1.0:
            raise ValueError("ghost_amplitude must be between 0 and 1")
        if not 1 <= int(self.max_polyphony) <= 12:
            raise ValueError("max_polyphony must be between 1 and 12")
        if not 0.0 <= float(self.onset_threshold) <= 1.0:
            raise ValueError("onset_threshold must be between 0 and 1")
        if not 0.0 <= float(self.frame_threshold) <= 1.0:
            raise ValueError("frame_threshold must be between 0 and 1")
        if self.layout not in {"auto", "onset", "ties"}:
            raise ValueError("layout must be auto, onset, or ties")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.demucs_model not in {"htdemucs_ft", "htdemucs"}:
            raise ValueError("demucs_model must be htdemucs_ft or htdemucs")
        if self.transcription_engine not in {"basic_pitch", "external"}:
            raise ValueError("transcription_engine must be basic_pitch or external")
        if not self.selected_stems:
            raise ValueError(
                "select at least one stem: " + ", ".join(STEM_NAMES)
            )
        if self.export_gp5 and not self.gp5_track_names():
            raise ValueError(
                "the selected stems produce no GP5 track; select the stem used "
                "by the guitar part, bass, drums, or vocals, or use MIDI-only mode"
            )

    @property
    def meter(self) -> Meter:
        return parse_time_signature(self.time_signature)

    @property
    def output_format(self) -> str:
        return "gp5_midi" if self.export_gp5 else "midi_only"

    def to_meta(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["work_dir"] = str(self.work_dir)
        return data


def resolve_device(requested: str) -> tuple[str, list[str]]:
    """Resolve auto/cuda/cpu to a usable device. Imports torch lazily."""
    warnings: list[str] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment error
        raise RuntimeError(
            "PyTorch is not importable. Run scripts/setup_env.sh and verify "
            "the user-site CUDA torch is visible inside .venv."
        ) from exc

    requested = requested or "auto"
    if requested == "cpu":
        return "cpu", warnings
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda", warnings
        warnings.append("CUDA was requested but is unavailable; using CPU (very slow).")
        return "cpu", warnings
    if torch.cuda.is_available():
        return "cuda", warnings
    warnings.append("CUDA is unavailable; using CPU (very slow).")
    return "cpu", warnings
