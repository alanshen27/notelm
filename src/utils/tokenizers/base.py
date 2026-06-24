from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from utils.midi_timing import TIMESTEP_MS


@dataclass
class TokenizerConfig:
    min_pitch: int = 21
    max_pitch: int = 108
    velocity_bins: int = 16
    use_program: bool = False


class BaseMidiTokenizer(ABC):
    """Common interface for MIDI → token id sequences."""

    name: str = "base"

    def __init__(self, config: TokenizerConfig | None = None):
        self.cfg = config or TokenizerConfig()
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._build_vocab()

    @property
    def timestep_ms(self) -> int:
        return TIMESTEP_MS

    def _add(self, token: str) -> None:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    @abstractmethod
    def _build_vocab(self) -> None:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def velocity_to_bin(self, velocity: int) -> int:
        bin_size = 128 / self.cfg.velocity_bins
        return min(int(velocity / bin_size), self.cfg.velocity_bins - 1)

    def bin_to_velocity(self, bin_idx: int) -> int:
        return min(127, int((bin_idx + 0.5) * 128 / self.cfg.velocity_bins))

    def _collect_notes(self, midi_path: str | Path) -> list[pretty_midi.Note]:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        notes: list[pretty_midi.Note] = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if self.cfg.min_pitch <= note.pitch <= self.cfg.max_pitch:
                    notes.append(note)
        notes.sort(key=lambda n: (n.start, n.pitch))
        return notes

    def ids_from_tokens(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id["UNK"]
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode_tokens(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(i, "UNK") for i in ids]

    @abstractmethod
    def encode_midi(self, midi_path: str | Path) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def tokens_to_midi(self, ids: list[int], output_path: str | Path) -> Path:
        raise NotImplementedError
