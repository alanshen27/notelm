"""Gradio UI for querying / generating from a trained LSTM checkpoint."""

from pathlib import Path

import gradio as gr

from inference import _search_roots, generate_midi, get_device, list_checkpoints

SRC = Path(__file__).resolve().parent


def _checkpoint_choices() -> list[str]:
    return list_checkpoints()


def _default_checkpoint() -> str:
    ckpts = _checkpoint_choices()
    return ckpts[0] if ckpts else ""


def refresh_checkpoints():
    ckpts = _checkpoint_choices()
    roots = "\n".join(f"  • {r}" for r in _search_roots())
    if ckpts:
        msg = f"Found {len(ckpts)} checkpoint(s).\nSearching:\n{roots}"
        return gr.Dropdown(choices=ckpts, value=ckpts[0]), msg
    msg = (
        "No checkpoints found.\n"
        f"Searching:\n{roots}\n\n"
        "On your server run:\n"
        "  find /notelm -name '*.pt'\n\n"
        "Then paste the full path below, or copy the file to src/checkpoints/."
    )
    return gr.Dropdown(choices=[], value=None), msg


def run_generation(
    checkpoint: str,
    custom_path: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed_midi,
    context_len: int,
):
    path = (custom_path or "").strip() or (checkpoint or "").strip()
    if not path:
        raise gr.Error(
            "No checkpoint found. Click Refresh, paste a path, or train first."
        )

    seed_path = None
    if seed_midi is not None:
        seed_path = seed_midi if isinstance(seed_midi, str) else getattr(seed_midi, "name", None)

    midi_path, preview, tokens = generate_midi(
        path,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k) if top_k > 0 else 0,
        seed_midi=seed_path,
        context_len=int(context_len),
    )

    status = (
        f"Generated {len(tokens)} tokens on {get_device()}.\n"
        f"Checkpoint: {path}"
    )
    return midi_path, preview, status


with gr.Blocks(title="notelm — LSTM MIDI") as demo:
    gr.Markdown(
        """
        # notelm — LSTM query UI
        Load a checkpoint, optionally seed from a MIDI file, and generate new piano MIDI.
        """
    )

    search_status = gr.Textbox(
        label="Checkpoint search",
        lines=4,
        interactive=False,
    )

    with gr.Row():
        checkpoint = gr.Dropdown(
            choices=_checkpoint_choices(),
            value=_default_checkpoint() or None,
            label="Checkpoint (auto-detected)",
            allow_custom_value=True,
        )
        refresh_btn = gr.Button("Refresh", scale=0)

    custom_path = gr.Textbox(
        label="Or paste checkpoint path",
        placeholder="/notelm/src/checkpoints/epoch-1/20260525-142432.pt",
    )

    with gr.Row():
        max_new_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max new tokens")
        temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
        top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k (0 = off)")
        context_len = gr.Slider(32, 1024, value=256, step=32, label="Seed context length")

    seed_midi = gr.File(label="Optional seed MIDI (.midi)", file_types=[".midi", ".mid"])

    generate_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        midi_out = gr.File(label="Download MIDI")
        status = gr.Textbox(label="Status", lines=3)

    token_preview = gr.Textbox(label="Token preview", lines=6)

    demo.load(refresh_checkpoints, outputs=[checkpoint, search_status])
    refresh_btn.click(refresh_checkpoints, outputs=[checkpoint, search_status])
    generate_btn.click(
        run_generation,
        inputs=[
            checkpoint,
            custom_path,
            max_new_tokens,
            temperature,
            top_k,
            seed_midi,
            context_len,
        ],
        outputs=[midi_out, token_preview, status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
