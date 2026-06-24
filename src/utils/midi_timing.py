"""Global MIDI quantization grid (shared by all tokenizers)."""

# Milliseconds per piano-roll column / timestep token.
TIMESTEP_MS = 20

# Derived helpers (do not duplicate timestep math in tokenizers).
TIMESTEP_SEC = TIMESTEP_MS / 1000.0
MAX_TIME_SHIFT_STEPS = 100  # 2 s at 20 ms when TIMESTEP_MS == 20


def seconds_to_timesteps(seconds: float) -> int:
    return round(seconds * 1000 / TIMESTEP_MS)


def timesteps_to_seconds(steps: int) -> float:
    return steps * TIMESTEP_SEC
