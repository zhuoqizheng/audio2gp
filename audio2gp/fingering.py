from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .config import TUNING_BASS, TUNING_GUITAR
from .types import FingeredNote, Note

GUITAR_MAX_STRINGS = 6
BASS_MAX_STRINGS = 4
GUITAR_MAX_SPAN = 5
BASS_MAX_SPAN = 4
# Folding is a last resort. Keep this much larger than ordinary position
# costs so an in-range position always wins when one exists.
FOLD_PENALTY = 1000.0


@dataclass
class _Candidate:
    pitch: int
    string: int
    fret: int
    folded: bool


@dataclass
class _Combo:
    notes: list[FingeredNote]
    cost: float
    center: float
    strings: frozenset[int]
    key: tuple[tuple[int, int, int], ...]
    folded: bool


def _tuning_pairs(tuning: dict[int, int]) -> list[tuple[int, int]]:
    return sorted(tuning.items(), key=lambda item: item[0])


def _candidates_for_pitch(
    pitch: int,
    tuning: dict[int, int],
    max_fret: int,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for variant, folded in ((pitch, False), (pitch - 12, True), (pitch + 12, True)):
        if not 0 <= variant <= 127:
            continue
        for string, open_pitch in _tuning_pairs(tuning):
            fret = variant - open_pitch
            if 0 <= fret <= max_fret:
                candidates.append(_Candidate(variant, string, fret, folded))
    # Prefer original octave and low frets; dedupe identical positions.
    seen: set[tuple[int, int, int]] = set()
    unique: list[_Candidate] = []
    for cand in sorted(
        candidates,
        key=lambda c: (c.folded, c.fret, c.string, abs(c.pitch - pitch), c.pitch),
    ):
        key = (cand.pitch, cand.string, cand.fret)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique


def _combo_from_candidates(
    chosen: list[_Candidate],
    notes: list[Note],
    folded: bool,
) -> _Combo:
    fretted = [c.fret for c in chosen if c.fret > 0]
    open_count = sum(1 for c in chosen if c.fret == 0)
    max_fret = max(fretted) if fretted else 0
    min_fret = min(fretted) if fretted else 0
    span = max_fret - min_fret
    avg_fret = sum(fretted) / len(fretted) if fretted else 0.0
    cost = (
        10.0 * max_fret
        + 5.0 * span
        + avg_fret
        + 2.0 * len(fretted)
        - 4.0 * open_count
        + (FOLD_PENALTY if folded else 0.0)
    )
    fingered: list[FingeredNote] = []
    for cand, note in zip(chosen, notes):
        fingered.append(
            FingeredNote(
                start_tick=note.start_tick,
                end_tick=note.end_tick,
                pitch=cand.pitch,
                velocity=note.velocity,
                string=cand.string,
                fret=cand.fret,
                slide=note.slide,
            )
        )
    key = tuple(sorted((n.pitch, n.string, n.fret) for n in fingered))
    centers = [n.fret for n in fingered if n.fret > 0]
    center = sum(centers) / len(centers) if centers else 0.0
    return _Combo(
        notes=fingered,
        cost=cost,
        center=center,
        strings=frozenset(c.string for c in chosen),
        key=key,
        folded=folded,
    )


def _make_layer(
    notes: list[Note],
    tuning: dict[int, int],
    max_fret: int,
    max_strings: int,
    max_span: int,
    limit: int,
    warnings: list[str],
) -> list[_Combo]:
    """Build legal fingerings for one onset, with folding/dropping fallback."""
    if not notes:
        return []
    # Prefer strong notes when more notes sound than there are strings.
    trimmed = sorted(notes, key=lambda n: (-n.velocity, n.pitch))
    if len(trimmed) > max_strings:
        dropped = trimmed[max_strings:]
        warnings.append(f"dropped {len(dropped)} simultaneous note(s): more than {max_strings} strings")
        trimmed = trimmed[:max_strings]
    # Dedupe duplicate pitches at this onset.
    by_pitch: dict[int, Note] = {}
    for note in trimmed:
        old = by_pitch.get(note.pitch)
        if old is None or note.velocity > old.velocity:
            by_pitch[note.pitch] = note
    ordered = sorted(by_pitch.values(), key=lambda n: (-n.velocity, n.pitch))

    candidate_lists = [
        _candidates_for_pitch(note.pitch, tuning, max_fret) for note in ordered
    ]
    # Search in the most constrained order for speed.
    order = sorted(range(len(ordered)), key=lambda i: len(candidate_lists[i]))
    ordered = [ordered[i] for i in order]
    candidate_lists = [candidate_lists[i] for i in order]

    results: list[_Combo] = []
    chosen: list[_Candidate] = []
    attempts = 0
    max_attempts = max(4000, limit * 80)

    def dfs(index: int, used: frozenset[int], frets: list[int], folded: bool) -> None:
        nonlocal attempts
        attempts += 1
        if attempts > max_attempts:
            return
        if index == len(ordered):
            fretted = [f for f in frets if f > 0]
            span = (max(fretted) - min(fretted)) if fretted else 0
            if span > max_span:
                return
            results.append(_combo_from_candidates(list(chosen), ordered, folded))
            if len(results) > limit * 8:
                results.sort(key=lambda c: (c.cost, c.key))
                del results[limit * 2 :]
            return
        for cand in candidate_lists[index]:
            if cand.string in used:
                continue
            if cand.fret > 0 and frets:
                fretted = [f for f in frets if f > 0] + [cand.fret]
                if max(fretted) - min(fretted) > max_span:
                    continue
            chosen.append(cand)
            dfs(
                index + 1,
                used | {cand.string},
                frets + [cand.fret],
                folded or cand.folded,
            )
            chosen.pop()

    dfs(0, frozenset(), [], False)
    if not results:
        # Last resort: drop the weakest notes until a legal subset exists.
        current = list(ordered)
        while current and not results:
            current = current[:-1]  # ordered by velocity; drop weakest tail
            warnings.append("fingering could not place the full chord; dropped the weakest note")
            candidate_lists = [
                _candidates_for_pitch(n.pitch, tuning, max_fret) for n in current
            ]
            results = []
            chosen = []
            attempts = 0
            ordered_local = current

            def dfs_subset(index: int, used: frozenset[int], frets: list[int], folded: bool) -> None:
                nonlocal attempts
                attempts += 1
                if attempts > max_attempts:
                    return
                if index == len(ordered_local):
                    fretted = [f for f in frets if f > 0]
                    span = (max(fretted) - min(fretted)) if fretted else 0
                    if span <= max_span:
                        results.append(_combo_from_candidates(list(chosen), ordered_local, folded))
                    return
                for cand in candidate_lists[index]:
                    if cand.string in used:
                        continue
                    if cand.fret > 0 and frets:
                        fretted = [f for f in frets if f > 0] + [cand.fret]
                        if max(fretted) - min(fretted) > max_span:
                            continue
                    chosen.append(cand)
                    dfs_subset(
                        index + 1,
                        used | {cand.string},
                        frets + [cand.fret],
                        folded or cand.folded,
                    )
                    chosen.pop()

            dfs_subset(0, frozenset(), [], False)
    if not results:
        warnings.append("no playable fingering for this onset; note(s) skipped")
        return []
    results.sort(key=lambda c: (c.cost, c.key))
    # Dedupe by shape and keep the best candidates.
    unique: list[_Combo] = []
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    for combo in results:
        if combo.key not in seen:
            seen.add(combo.key)
            unique.append(combo)
        if len(unique) >= limit:
            break
    return unique


def _transition(prev: _Combo, curr: _Combo) -> float:
    move = abs(prev.center - curr.center)
    changed_strings = len(curr.strings - prev.strings)
    same_pitch = {n.pitch for n in prev.notes} & {n.pitch for n in curr.notes}
    prev_pos = {(n.pitch, n.string, n.fret) for n in prev.notes}
    curr_pos = {(n.pitch, n.string, n.fret) for n in curr.notes}
    changed_position = len(same_pitch) - len(
        {p for p in same_pitch if (p, *next((x[1:] for x in prev_pos if x[0] == p), (0, 0))) in curr_pos}
    )
    return (
        3.0 * move
        + 0.5 * changed_strings
        + 6.0 * max(0.0, move - 4.0)
        + 4.0 * max(0, changed_position)
    )


def _trim_string_conflicts(notes: list[FingeredNote], warnings: list[str]) -> list[FingeredNote]:
    by_string: dict[int, list[FingeredNote]] = defaultdict(list)
    for note in notes:
        by_string[note.string].append(note)
    kept: list[FingeredNote] = []
    trimmed = 0
    for string_notes in by_string.values():
        string_notes.sort(key=lambda n: (n.start_tick, n.end_tick))
        chosen: list[FingeredNote] = []
        for note in string_notes:
            if chosen and note.start_tick < chosen[-1].end_tick:
                prev = chosen[-1]
                if note.start_tick > prev.start_tick:
                    prev.end_tick = note.start_tick
                    trimmed += 1
                    chosen.append(note)
                elif (note.velocity, note.end_tick) >= (prev.velocity, prev.end_tick):
                    chosen[-1] = note
                    trimmed += 1
            else:
                chosen.append(note)
        kept.extend(n for n in chosen if n.end_tick > n.start_tick)
    if trimmed:
        warnings.append(f"trimmed {trimmed} overlapping note(s) on the same string")
    kept.sort(key=lambda n: (n.start_tick, n.string, n.pitch))
    return kept


def _arrange(
    notes: list[Note],
    tuning: dict[int, int],
    max_fret: int,
    max_strings: int,
    max_span: int,
    limit: int,
    monophonic: bool,
) -> tuple[list[FingeredNote], list[str]]:
    warnings: list[str] = []
    by_start: dict[int, list[Note]] = defaultdict(list)
    for note in notes:
        if note.end_tick > note.start_tick and 0 <= note.pitch <= 127:
            by_start[note.start_tick].append(note)
    layers: list[list[_Combo]] = []
    for start in sorted(by_start):
        onset_notes = by_start[start]
        if monophonic:
            # Bass: lowest; vocals: strongest. This also prevents accidental diads.
            onset_notes = [min(onset_notes, key=lambda n: (n.pitch, -n.velocity))]
        combos = _make_layer(
            onset_notes,
            tuning,
            max_fret,
            max_strings,
            max_span,
            limit,
            warnings,
        )
        if combos:
            layers.append(combos)
    if not layers:
        return [], warnings

    dp: list[float] = [combo.cost for combo in layers[0]]
    back: list[list[int]] = []
    for layer_index in range(1, len(layers)):
        prev_layer = layers[layer_index - 1]
        curr_layer = layers[layer_index]
        new_dp: list[float] = []
        new_back: list[int] = []
        for curr in curr_layer:
            best_cost = float("inf")
            best_prev = 0
            for prev_index, prev in enumerate(prev_layer):
                cost = dp[prev_index] + curr.cost + _transition(prev, curr)
                if cost < best_cost:
                    best_cost = cost
                    best_prev = prev_index
            new_dp.append(best_cost)
            new_back.append(best_prev)
        dp = new_dp
        back.append(new_back)

    best_index = min(range(len(dp)), key=lambda i: dp[i])
    chosen_indices = [best_index]
    for layer_index in range(len(layers) - 1, 0, -1):
        best_index = back[layer_index - 1][chosen_indices[-1]]
        chosen_indices.append(best_index)
    chosen_indices.reverse()
    result: list[FingeredNote] = []
    folded_count = 0
    for layer, index in zip(layers, chosen_indices):
        combo = layer[index]
        if combo.folded:
            folded_count += 1
        result.extend(combo.notes)
    if folded_count:
        warnings.append(
            f"octave-folded {folded_count} onset(s) to fit the fretboard/range"
        )
    result = _trim_string_conflicts(result, warnings)
    return result, warnings


def arrange_guitar(
    notes: list[Note],
    max_fret: int = 15,
    monophonic: bool = False,
    tuning: dict[int, int] | None = None,
) -> tuple[list[FingeredNote], list[str]]:
    return _arrange(
        notes,
        tuning or TUNING_GUITAR,
        max_fret=max_fret,
        max_strings=1 if monophonic else GUITAR_MAX_STRINGS,
        max_span=GUITAR_MAX_SPAN,
        limit=64,
        monophonic=monophonic,
    )


def arrange_bass(
    notes: list[Note],
    max_fret: int = 15,
    tuning: dict[int, int] | None = None,
) -> tuple[list[FingeredNote], list[str]]:
    return _arrange(
        notes,
        tuning or TUNING_BASS,
        max_fret=max_fret,
        max_strings=1,
        max_span=BASS_MAX_SPAN,
        limit=16,
        monophonic=True,
    )


def arrange_vocals(notes: list[Note]) -> tuple[list[FingeredNote], list[str]]:
    return arrange_guitar(notes, max_fret=24, monophonic=True)


def arrangement_score(notes: list[Note], arranged: list[FingeredNote]) -> float:
    """Rank a candidate tuning mostly by how many original pitches survive.

    Coverage dominates; comfort only breaks ties between tunings that can play
    the same material. Folded octaves therefore never win, while Drop D wins
    for a low-D riff that would otherwise be folded or skipped.
    """
    if not notes:
        return 0.0
    wanted = {(int(note.start_tick), int(note.pitch)) for note in notes}
    hits = sum(
        1
        for note in arranged
        if (int(note.start_tick), int(note.pitch)) in wanted
    )
    coverage = hits / len(notes)
    frets = [note.fret for note in arranged if note.fret > 0]
    avg_fret = sum(frets) / len(frets) if frets else 0.0
    max_fret = max(frets) if frets else 0
    return 1000.0 * coverage - 0.5 * avg_fret - 0.1 * max_fret


def suggest_tuning(
    notes: list[Note],
    tunings: dict[str, dict[int, int]],
    *,
    max_fret: int = 15,
    kind: str = "guitar",
) -> tuple[str, list[dict[str, float | str]], list[FingeredNote], list[str]]:
    """Try every preset and return the best name, scores, result and warnings."""
    best_name = next(iter(tunings))
    best_score = float("-inf")
    best_notes: list[FingeredNote] = []
    best_warnings: list[str] = []
    scores: list[dict[str, float | str]] = []
    for name, tuning in tunings.items():
        if kind == "bass":
            arranged, warnings = arrange_bass(
                notes, max_fret=max_fret, tuning=tuning
            )
        else:
            arranged, warnings = arrange_guitar(
                notes, max_fret=max_fret, tuning=tuning
            )
        score = arrangement_score(notes, arranged)
        scores.append({"name": name, "score": round(float(score), 3)})
        if score > best_score:
            best_name = name
            best_score = score
            best_notes = arranged
            best_warnings = list(warnings)
    scores.sort(key=lambda item: float(item["score"]), reverse=True)
    return best_name, scores, best_notes, best_warnings
