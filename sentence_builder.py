#!/usr/bin/env python
"""
sentence_builder.py — Anti-flicker sentence accumulation for BSL demo.

Replaces naive `sentence.append(label)` with:
  1. Stability buffer: only emit after K consecutive same-label spots
  2. Cooldown timer: minimum gap between emissions (prevents burst repeats)
  3. Consecutive dedup: never emit the same word twice in a row
  4. Confidence gate: only consider predictions above threshold
"""
import time
import collections


class SentenceBuilder:
    """
    Accumulates recognized signs into a stable sentence with anti-flicker logic.

    Parameters
    ----------
    stability_count : int
        Number of consecutive same-label spotted events required before
        the word is emitted.  Default 2 — two back-to-back "good" spots
        needed to add "good" to the sentence.
    cooldown_sec : float
        Minimum seconds between sentence additions.  Prevents burst of
        the same word when the signer holds a pose.
    confidence_threshold : float
        Minimum confidence to consider a spotted label at all.
    max_history : int
        Maximum sentence length (oldest words are kept; display can
        truncate from the left).
    """

    def __init__(
        self,
        stability_count: int = 2,
        cooldown_sec: float = 1.0,
        confidence_threshold: float = 0.55,
        max_history: int = 50,
    ):
        self.stability_count = stability_count
        self.cooldown_sec = cooldown_sec
        self.confidence_threshold = confidence_threshold
        self.max_history = max_history

        # Internal state
        self._sentence: list[str] = []
        self._run_label: str = ""        # current consecutive label
        self._run_count: int = 0         # how many consecutive same labels
        self._last_emit_time: float = 0  # timestamp of last emission
        self._total_spots: int = 0
        self._total_emitted: int = 0
        self._total_rejected: int = 0

    # ── Public API ────────────────────────────────────────────

    def feed(self, label: str, confidence: float) -> str | None:
        """
        Feed a new spotted prediction.

        Returns the label string if it was emitted to the sentence,
        or None if rejected (low confidence, unstable, cooldown, dup).
        """
        self._total_spots += 1

        # Gate 1: confidence
        if confidence < self.confidence_threshold:
            self._run_label = ""
            self._run_count = 0
            self._total_rejected += 1
            return None

        # Gate 2: stability (consecutive same label)
        if label == self._run_label:
            self._run_count += 1
        else:
            self._run_label = label
            self._run_count = 1

        if self._run_count < self.stability_count:
            return None  # not stable yet

        # Gate 3: cooldown timer
        now = time.time()
        if (now - self._last_emit_time) < self.cooldown_sec:
            return None

        # Gate 4: consecutive dedup (don't repeat last word)
        if self._sentence and self._sentence[-1] == label:
            return None

        # ── EMIT ──
        self._sentence.append(label)
        if len(self._sentence) > self.max_history:
            self._sentence = self._sentence[-self.max_history:]
        self._last_emit_time = now
        self._total_emitted += 1
        # Reset run to prevent immediate re-emit
        self._run_count = 0
        return label

    @property
    def sentence(self) -> list[str]:
        """Current sentence as list of words."""
        return list(self._sentence)

    @property
    def text(self) -> str:
        """Current sentence as space-separated string."""
        return " ".join(self._sentence)

    def clear(self):
        """Reset everything."""
        self._sentence.clear()
        self._run_label = ""
        self._run_count = 0
        self._last_emit_time = 0
        self._total_spots = 0
        self._total_emitted = 0
        self._total_rejected = 0

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        return {
            "total_spots": self._total_spots,
            "total_emitted": self._total_emitted,
            "total_rejected": self._total_rejected,
            "sentence_length": len(self._sentence),
            "filter_rate": (
                f"{100*(1 - self._total_emitted/max(1,self._total_spots)):.0f}%"
            ),
        }

    def __len__(self):
        return len(self._sentence)

    def __repr__(self):
        return (
            f"SentenceBuilder(words={len(self._sentence)}, "
            f"emitted={self._total_emitted}/{self._total_spots})"
        )
