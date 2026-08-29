#!/usr/bin/env python3
"""Download extra MIDI corpora into data/. Stdlib plus optional huggingface_hub.

  python3 scripts/fetch_datasets.py emopia pop1k7 adl asap
  python3 scripts/fetch_datasets.py extra          # emopia pop1k7 adl asap
  python3 scripts/fetch_datasets.py ccby           # giantmidi atepp pdmx
  python3 scripts/fetch_datasets.py gigamidi --max-files 50000
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ZENODO_EMOPIA = "https://zenodo.org/api/records/5090631/files/EMOPIA_1.0.zip/content"
ZENODO_POP1K7 = "https://zenodo.org/api/records/13167761/files/Pop1K7.zip/content"
ADL_GIT = "https://github.com/lucasnfe/adl-piano-midi.git"
ASAP_GIT = "https://github.com/fosfrancesco/asap-dataset.git"


def log(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, _, val = raw.partition("=")
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = val.strip().strip("'\"")
    # huggingface.co is often blocked in mainland China.
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        log("HF_ENDPOINT unset — using https://hf-mirror.com")


def _download(url: str, dest: Path, *, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "notelm-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _install_extract(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    nested = [p for p in tmp.iterdir() if p.is_dir()]
    # Zip often has a single top-level folder.
    src = nested[0] if len(nested) == 1 and not any(tmp.glob("*.mid")) else tmp
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def fetch_emopia() -> None:
    dest = DATA / "EMOPIA"
    if _count_midi(dest) > 0:
        log(f"EMOPIA already present ({_count_midi(dest)} MIDI files)")
        return
    zpath = DATA / "EMOPIA_1.0.zip"
    _download(ZENODO_EMOPIA, zpath)
    tmp = DATA / "_emopia_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    _extract_zip(zpath, tmp)
    _install_extract(tmp, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    zpath.unlink(missing_ok=True)
    log(f"EMOPIA ready ({_count_midi(dest)} MIDI files)")


def fetch_pop1k7() -> None:
    dest = DATA / "Pop1K7"
    if _count_midi(dest) > 0:
        log(f"Pop1K7 already present ({_count_midi(dest)} MIDI files)")
        return
    zpath = DATA / "Pop1K7.zip"
    _download(ZENODO_POP1K7, zpath)
    tmp = DATA / "_pop1k7_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    log("Extracting Pop1K7 (this zip is ~300 MB)...")
    _extract_zip(zpath, tmp)
    _install_extract(tmp, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    zpath.unlink(missing_ok=True)
    log(f"Pop1K7 ready ({_count_midi(dest)} MIDI files)")


def _count_midi(root: Path) -> int:
    if not root.is_dir():
        return 0
    n = 0
    for ext in ("*.mid", "*.midi"):
        n += sum(1 for _ in root.rglob(ext))
    return n


def _git_clone(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True,
    )


def fetch_adl() -> None:
    dest = DATA / "adl-piano-midi"
    if _count_midi(dest) > 0:
        log(f"ADL Piano MIDI already present ({_count_midi(dest)} files)")
        return
    # GitHub gitignores midi/train and midi/test. The actual ~11k pieces are
    # midi/adl-piano-midi.zip inside the clone — we have to unzip them.
    if not dest.exists():
        log("Cloning lucasnfe/adl-piano-midi (zip of ~11k piano MIDIs)...")
        _git_clone(ADL_GIT, dest)
    zips = sorted(p for p in dest.rglob("*.zip") if p.is_file())
    if not zips:
        raise SystemExit(
            "ADL clone has no MIDI and no zip. midi/train is gitignored; "
            "expected midi/adl-piano-midi.zip."
        )
    for zpath in zips:
        log(f"Extracting {zpath.name} ({zpath.stat().st_size / 1e6:.1f} MB)...")
        _extract_zip(zpath, dest)
    n = _count_midi(dest)
    if n == 0:
        raise SystemExit("ADL zip extracted but no .mid files found")
    log(f"ADL ready ({n} MIDI files)")


def fetch_asap() -> None:
    dest = DATA / "asap-dataset"
    if _count_midi(dest) > 0:
        log(f"ASAP already present ({_count_midi(dest)} MIDI files)")
        return
    log("Cloning fosfrancesco/asap-dataset...")
    _git_clone(ASAP_GIT, dest)
    log(f"ASAP ready ({_count_midi(dest)} MIDI files)")


def _midi_bytes_from_row(row: dict) -> bytes | None:
    for key in (
        "music",  # Metacreation/GigaMIDI official field
        "midi",
        "midi_bytes",
        "bytes",
        "content",
        "midi_data",
        "file",
        "midi_file",
    ):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (bytes, bytearray, memoryview)):
            raw = bytes(val)
            if raw[:4] == b"MThd":
                return raw
        if isinstance(val, dict) and "bytes" in val:
            raw = bytes(val["bytes"])
            if raw[:4] == b"MThd":
                return raw
    return None


def _is_piano_row(row: dict) -> bool:
    for key in ("is_drum", "drum"):
        if row.get(key) in (True, 1, "True"):
            return False
    for key in (
        "loop_instrument_type",
        "instrument_group",
        "instrument_group__expressive_",
    ):
        val = row.get(key)
        if not val:
            continue
        items = val if isinstance(val, (list, tuple)) else [val]
        if any("piano" in str(x).lower() for x in items):
            return True
    progs: list[int] = []
    for key in (
        "program",
        "instrument_program",
        "inst_program",
        "MIDI_program_number",
        "MIDI_program_number__expressive_",
    ):
        val = row.get(key)
        if val is None:
            continue
        vals = val if isinstance(val, (list, tuple)) else [val]
        for x in vals:
            try:
                progs.append(int(x))
            except (TypeError, ValueError):
                continue
    if progs:
        return any(0 <= p <= 7 for p in progs)
    return True


def fetch_gigamidi(*, max_files: int) -> None:
    dest = DATA / "GigaMIDI"
    dest.mkdir(parents=True, exist_ok=True)
    existing = _count_midi(dest)
    if existing >= max_files:
        log(f"GigaMIDI already has {existing} files (>= {max_files})")
        return
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "GigaMIDI fetch needs the Hugging Face datasets package:\n"
            "  uv pip install datasets huggingface_hub\n"
            "The Hub copy is gated (research/education). Create a token at\n"
            "https://huggingface.co/settings/tokens , accept the dataset terms at\n"
            "https://huggingface.co/datasets/Metacreation/GigaMIDI , then:\n"
            "  export HF_TOKEN=hf_...\n"
            "  python3 scripts/fetch_datasets.py gigamidi",
            file=sys.stderr,
        )
        raise SystemExit(1)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "GigaMIDI is gated. Add HF_TOKEN to .env (gitignored) after accepting\n"
            "terms at https://huggingface.co/datasets/Metacreation/GigaMIDI",
            file=sys.stderr,
        )
        raise SystemExit(1)
    log(
        f"Streaming Metacreation/GigaMIDI (piano-ish tracks, cap {max_files}; "
        f"have {existing})..."
    )
    kwargs = dict(split="train", streaming=True, token=token)
    try:
        ds = load_dataset("Metacreation/GigaMIDI", "v2.0.0", **kwargs)
    except Exception:
        try:
            ds = load_dataset("Metacreation/GigaMIDI", **kwargs)
        except Exception as exc:
            print(
                f"Could not stream GigaMIDI ({exc}).\n"
                "Accept the gated terms at "
                "https://huggingface.co/datasets/Metacreation/GigaMIDI "
                "then retry.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

    written = existing
    scanned = 0
    piano_without_bytes = 0
    for row in ds:
        if written >= max_files:
            break
        scanned += 1
        if not isinstance(row, dict):
            row = dict(row)
        if not _is_piano_row(row):
            continue
        raw = _midi_bytes_from_row(row)
        if not raw:
            piano_without_bytes += 1
            if piano_without_bytes >= 50:
                raise SystemExit(
                    "GigaMIDI piano rows have no MIDI bytes in `music` "
                    "(gated access not granted?). Accept terms on the Hub, then retry."
                )
            continue
        name = row.get("md5") or row.get("id") or f"{written:08d}"
        path = dest / f"{name}.mid"
        if path.exists():
            continue
        path.write_bytes(raw)
        written += 1
        if written % 200 == 0:
            log(f"  GigaMIDI wrote {written}/{max_files}")
    n = _count_midi(dest)
    if n == 0:
        raise SystemExit("GigaMIDI fetch wrote 0 MIDI files")
    log(f"GigaMIDI ready ({n} MIDI files, scanned {scanned} rows)")


GIANTMIDI_DRIVE = "https://drive.google.com/drive/folders/1Stz3CAvMoplo79LR5I3onMWRelCugBYS"
ATEPP_HF = "anusfoil/atepp-midi"
ATEPP_ZENODO = "7182820"
PDMX_MID = "https://zenodo.org/api/records/15571083/files/mid.tar.gz/content"


def _ensure_pkg(mod: str, pip_name: str | None = None) -> None:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name or mod],
            check=True,
        )


def _promote_midis(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    n = _count_midi(src)
    if n == 0:
        return
    if dest.exists() and _count_midi(dest) == 0:
        shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        return
    for ext in ("*.mid", "*.midi"):
        for p in src.rglob(ext):
            if p.name.startswith("._") or "__MACOSX" in p.parts:
                continue
            out = dest / p.name
            if out.exists():
                out = dest / f"{p.parent.name}_{p.name}"
            if not out.exists():
                shutil.copy2(p, out)


def fetch_giantmidi(*, max_files: int = 0) -> None:
    dest = DATA / "GiantMIDI-Piano"
    if _count_midi(dest) >= 1000:
        log(f"GiantMIDI-Piano already present ({_count_midi(dest)} MIDI files)")
        return
    _ensure_pkg("gdown")
    import gdown

    tmp = DATA / "_giantmidi_dl"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    log("Downloading GiantMIDI-Piano from Google Drive (CC BY 4.0, ~193 MB)...")
    kw = {"output": str(tmp), "quiet": False}
    import inspect
    if "remaining_ok" in inspect.signature(gdown.download_folder).parameters:
        kw["remaining_ok"] = True
    gdown.download_folder(GIANTMIDI_DRIVE, **kw)
    zips = list(tmp.rglob("*.zip"))
    extract = DATA / "_giantmidi_extract"
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir()
    if zips:
        for zpath in zips:
            log(f"Extracting {zpath.name}...")
            _extract_zip(zpath, extract)
        _promote_midis(extract, dest)
    else:
        _promote_midis(tmp, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(extract, ignore_errors=True)
    n = _count_midi(dest)
    if n == 0:
        raise SystemExit("GiantMIDI-Piano download had no MIDI files")
    log(f"GiantMIDI-Piano ready ({n} MIDI files)")


def fetch_atepp(*, max_files: int = 0) -> None:
    dest = DATA / "ATEPP"
    if _count_midi(dest) >= 1000:
        log(f"ATEPP already present ({_count_midi(dest)} MIDI files)")
        return
    dest.mkdir(parents=True, exist_ok=True)
    log("Downloading ATEPP (CC BY / CC0 piano transcriptions)...")
    got = False
    try:
        _ensure_pkg("huggingface_hub")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=ATEPP_HF,
            repo_type="dataset",
            local_dir=str(dest),
        )
        got = _count_midi(dest) > 0
    except Exception as exc:
        log(f"Hugging Face ATEPP failed ({exc})")
    if not got:
        log(f"Trying Zenodo {ATEPP_ZENODO}")
        import json

        meta_path = DATA / "_atepp_record.json"
        _download(f"https://zenodo.org/api/records/{ATEPP_ZENODO}", meta_path, timeout=120)
        rec = json.loads(meta_path.read_text())
        meta_path.unlink(missing_ok=True)
        files = rec.get("files") or []
        files = sorted(files, key=lambda f: int(f.get("size") or 0), reverse=True)
        if not files:
            raise SystemExit("ATEPP Zenodo record has no files")
        url = files[0]["links"]["self"]
        if not url.endswith("/content"):
            url = url.rstrip("/") + "/content"
        zpath = DATA / files[0]["key"]
        _download(url, zpath, timeout=3600)
        tmp = DATA / "_atepp_extract"
        if tmp.exists():
            shutil.rmtree(tmp)
        if zpath.suffix == ".zip":
            _extract_zip(zpath, tmp)
        else:
            tmp.mkdir()
            shutil.unpack_archive(str(zpath), str(tmp))
        _promote_midis(tmp, dest)
        shutil.rmtree(tmp, ignore_errors=True)
        zpath.unlink(missing_ok=True)
    n = _count_midi(dest)
    if n == 0:
        raise SystemExit("ATEPP download had no MIDI files")
    log(f"ATEPP ready ({n} MIDI files)")


def fetch_pdmx(*, max_files: int = 0) -> None:
    dest = DATA / "PDMX"
    if _count_midi(dest) >= 1000:
        log(f"PDMX already present ({_count_midi(dest)} MIDI files)")
        return
    zpath = DATA / "pdmx-mid.tar.gz"
    log("Downloading PDMX MIDI only (CC BY 4.0, ~214 MB; skip PDF/MXL)...")
    _download(PDMX_MID, zpath, timeout=3600)
    dest.mkdir(parents=True, exist_ok=True)
    log("Extracting PDMX mid.tar.gz...")
    # RunPod volume is root-owned; archive UIDs 1032 would otherwise fail tar.
    subprocess.run(
        ["tar", "--no-same-owner", "-xzf", str(zpath), "-C", str(dest)],
        check=True,
    )
    zpath.unlink(missing_ok=True)
    n = _count_midi(dest)
    if n == 0:
        raise SystemExit("PDMX extract had no MIDI files")
    log(f"PDMX ready ({n} MIDI files)")


FETCHERS = {
    "emopia": lambda **_: fetch_emopia(),
    "pop1k7": lambda **_: fetch_pop1k7(),
    "adl": lambda **_: fetch_adl(),
    "asap": lambda **_: fetch_asap(),
    "gigamidi": lambda max_files, **_: fetch_gigamidi(max_files=max_files),
    "giantmidi": lambda **_: fetch_giantmidi(),
    "atepp": lambda **_: fetch_atepp(),
    "pdmx": lambda **_: fetch_pdmx(),
}

EXTRA = ("emopia", "pop1k7", "adl", "asap")
CCBY = ("giantmidi", "atepp", "pdmx")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch MIDI corpora into data/")
    p.add_argument(
        "names",
        nargs="+",
        help="emopia pop1k7 adl asap giantmidi atepp pdmx gigamidi extra ccby all",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=50_000,
        help="GigaMIDI piano-ish file cap (default 50000; Hub has 2.1M files)",
    )
    args = p.parse_args()
    _load_dotenv()
    DATA.mkdir(parents=True, exist_ok=True)

    wanted: list[str] = []
    for n in args.names:
        key = n.strip().lower()
        if key == "extra":
            wanted.extend(EXTRA)
        elif key == "ccby":
            wanted.extend(CCBY)
        elif key == "all":
            wanted.extend(EXTRA)
            wanted.extend(CCBY)
            wanted.append("gigamidi")
        elif key in FETCHERS:
            wanted.append(key)
        else:
            p.error(f"unknown corpus {n!r}")

    seen: set[str] = set()
    for name in wanted:
        if name in seen:
            continue
        seen.add(name)
        FETCHERS[name](max_files=args.max_files)


if __name__ == "__main__":
    main()
