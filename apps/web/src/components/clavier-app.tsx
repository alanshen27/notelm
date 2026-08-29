"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type PointerEvent } from "react";
import Link from "next/link";
import { routes } from "@/lib/routes";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { continueNotes, fetchCheckpoints, type Checkpoint, type SynthNote } from "@/lib/api";
import {
  checkpointLabel,
  cowriterCheckpoints,
  matchCheckpoint,
  preferCheckpoint,
} from "@/lib/checkpoints";
import { PRESETS, SynthEngine } from "@/lib/synth";
import { cn } from "@/lib/utils";

/** 88-key piano: A0–C8 */
const PITCH_TOP = 108;
const PITCH_BOTTOM = 21;
const CELL_W = 22;
const CELL_H = 18;
const PITCH_W = 64;
const RULER_H = 36;
const PEDAL_H = 28;
const PAD_BARS = 8;
const MIN_BARS = 8;
const ROWS = PITCH_TOP - PITCH_BOTTOM + 1;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BLACK = new Set([1, 3, 6, 8, 10]);
const PROG_BARS = 4;

const PROGRESSIONS: { id: string; label: string; chords: number[][] }[] = [
  { id: "pop", label: "I–V–vi–IV", chords: [[60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60]] },
  { id: "sad", label: "vi–IV–I–V", chords: [[57, 60, 64], [53, 57, 60], [60, 64, 67], [55, 59, 62]] },
  { id: "jazz", label: "I–vi–ii–V", chords: [[60, 64, 67], [57, 60, 64], [50, 53, 57], [55, 59, 62]] },
];

type RollNote = {
  id: number;
  pitch: number;
  cell: number;
  cells: number;
  velocity: number;
  source: "user" | "ai";
};

type CellRange = { start: number; end: number };

let noteId = 1;

function pitchName(p: number) {
  return `${NOTE_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
}

function isBlack(p: number) {
  return BLACK.has(p % 12);
}

function asNum(v: number | readonly number[]) {
  return typeof v === "number" ? v : v[0];
}

function normRange(a: number, b: number): CellRange {
  const start = Math.max(0, Math.min(a, b));
  const end = Math.max(Math.max(a, b), start + 1);
  return { start, end };
}

function colFromEvent(e: { clientX: number }, el: HTMLElement) {
  const x = e.clientX - el.getBoundingClientRect().left;
  return Math.max(0, Math.floor(x / CELL_W));
}

function overlaps(n: RollNote, range: CellRange) {
  return n.cell < range.end && n.cell + n.cells > range.start;
}

function mergeRanges(ranges: CellRange[]): CellRange[] {
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const out: CellRange[] = [];
  for (const r of sorted) {
    const last = out[out.length - 1];
    if (last && r.start <= last.end) last.end = Math.max(last.end, r.end);
    else out.push({ ...r });
  }
  return out;
}

function applySustain(notes: RollNote[], pedals: CellRange[], spb: number): SynthNote[] {
  return notes.map((n) => {
    const off = n.cell + n.cells;
    let release = off;
    for (const p of pedals) {
      if (off > p.start && off <= p.end) release = Math.max(release, p.end);
    }
    return {
      pitch: n.pitch,
      start: n.cell * spb,
      duration: Math.max(n.cells, release - n.cell) * spb,
      velocity: n.velocity,
    };
  });
}

type ClavierAppProps = {
  dev?: boolean;
  ckptQuery?: string;
};

export function ClavierApp({ dev = false, ckptQuery = "" }: ClavierAppProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const rollRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef(new SynthEngine());
  const playingRef = useRef(false);
  const loopRef = useRef<CellRange | null>(null);
  const playAgainRef = useRef<(() => void) | null>(null);

  const [tempo, setTempo] = useState(100);
  const [velocity, setVelocity] = useState(100);
  const [notes, setNotes] = useState<RollNote[]>([]);
  const [pedals, setPedals] = useState<CellRange[]>([]);
  const [viewW, setViewW] = useState(1280);
  const [selection, setSelection] = useState<CellRange | null>(null);
  const [loop, setLoop] = useState<CellRange | null>(null);
  const [heldKey, setHeldKey] = useState<number | null>(null);
  const [draft, setDraft] = useState<
    | { kind: "draw"; pitch: number; start: number; end: number }
    | { kind: "select"; start: number; end: number }
    | { kind: "loop"; start: number; end: number }
    | { kind: "pedal"; start: number; end: number }
    | { kind: "drop"; start: number }
    | null
  >(null);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [playPos, setPlayPos] = useState(0);
  const [playOrigin, setPlayOrigin] = useState(0);
  const [tool, setTool] = useState<"draw" | "select">("select");
  const [error, setError] = useState<string | null>(null);

  const spb = 60 / tempo / 4;

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);
  useEffect(() => {
    loopRef.current = loop;
  }, [loop]);

  useEffect(() => {
    engineRef.current.setParams(PRESETS["Neon Keys"]);
  }, []);

  useEffect(() => {
    fetchCheckpoints()
      .then((data) => {
        const raw = data.checkpoints || [];
        const list = dev ? raw : cowriterCheckpoints(raw);
        setCheckpoints(list);
        setCheckpoint((cur) => {
          const fromQuery = matchCheckpoint(raw, ckptQuery);
          if (fromQuery) return fromQuery;
          if (cur && raw.some((c) => c.path === cur)) return cur;
          return dev ? list[0]?.path ?? "" : preferCheckpoint(raw);
        });
      })
      .catch((err: Error) => setError(err.message));
  }, [dev, ckptQuery]);

  useEffect(() => {
    engineRef.current.onPos = (s) => setPlayPos(s);
    engineRef.current.onEnded = () => {
      if (loopRef.current && playingRef.current) {
        playAgainRef.current?.();
        return;
      }
      setPlaying(false);
      setPlayPos(0);
    };
  }, []);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewW(el.clientWidth));
    ro.observe(el);
    setViewW(el.clientWidth);
    const c4Row = PITCH_TOP - 60;
    el.scrollTop = Math.max(0, c4Row * CELL_H - el.clientHeight * 0.45);
    return () => ro.disconnect();
  }, []);

  const contentEnd = notes.reduce((m, n) => Math.max(m, n.cell + n.cells), 0);
  const minCols = Math.max(MIN_BARS * 16, Math.ceil(Math.max(0, viewW - PITCH_W) / CELL_W));
  const totalCols = Math.max(minCols, contentEnd + PAD_BARS * 16);
  const rollW = totalCols * CELL_W;
  const rollH = ROWS * CELL_H;
  const liveSel = draft?.kind === "select" ? normRange(draft.start, draft.end) : selection;
  const liveLoop = draft?.kind === "loop" ? normRange(draft.start, draft.end) : loop;
  const livePedal = draft?.kind === "pedal" ? normRange(draft.start, draft.end) : null;

  const timed = useMemo(
    () => applySustain(notes, pedals, spb),
    [notes, pedals, spb]
  );

  const stampProgression = useCallback((id: string, atCell: number) => {
    const spec = PROGRESSIONS.find((p) => p.id === id);
    if (!spec) return;
    const cellsPer = 16;
    const stamped: RollNote[] = [];
    spec.chords.forEach((chord, i) => {
      chord.forEach((pitch) => {
        stamped.push({
          id: noteId++,
          pitch,
          cell: atCell + i * cellsPer,
          cells: cellsPer,
          velocity,
          source: "user",
        });
      });
    });
    setNotes((prev) => [...prev, ...stamped]);
    setError(null);
  }, [velocity]);

  const fillRange = async (range: CellRange) => {
    const left = notes.filter((n) => n.cell + n.cells <= range.start);
    if (!left.length) {
      setError(
        notes.some((n) => n.cell >= range.end)
          ? "Need notes on the left of the gap — pick a bidirectional model (canon) once it is trained, or draw something before the range."
          : "Draw or drop something before this range."
      );
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const seedEnd = Math.max(...left.map((n) => n.cell + n.cells));
      const seedNotes = notes.map((n) => ({
        pitch: n.pitch,
        start: n.cell * spb,
        duration: n.cells * spb,
        velocity: n.velocity,
      }));
      const span = Math.max(range.end - seedEnd, range.end - range.start);
      const out = await continueNotes({
        notes: seedNotes,
        checkpoint: checkpoint || undefined,
        max_new_tokens: Math.min(1200, Math.max(240, span * 10)),
        temperature: 1,
        emotion: "none",
        instrument: "piano",
        tempo,
        range_start: range.start * spb,
        range_end: range.end * spb,
      });
      const filled: RollNote[] = [];
      for (const n of out.notes) {
        let cell = Math.round(n.start / spb);
        let cells = Math.max(1, Math.round(n.duration / spb));
        if (cell + cells <= range.start || cell >= range.end) continue;
        if (cell < range.start) {
          cells -= range.start - cell;
          cell = range.start;
        }
        if (cell + cells > range.end) cells = range.end - cell;
        if (cells < 1) continue;
        filled.push({
          id: noteId++,
          pitch: n.pitch,
          cell,
          cells,
          velocity: n.velocity ?? 100,
          source: "ai",
        });
      }
      setNotes((prev) => [...prev.filter((n) => !overlaps(n, range)), ...filled]);
      if (!filled.length) setError("Empty fill — select a longer gap or try again.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const continueAfter = () => {
    if (!notes.length) {
      setError("Draw or drop a progression first.");
      return;
    }
    const start = Math.max(...notes.map((n) => n.cell + n.cells));
    const range = { start, end: start + 64 };
    setSelection(range);
    void fillRange(range);
  };

  const startPlayback = async () => {
    if (!timed.length) {
      setError("Nothing to play — draw notes or click a progression.");
      return;
    }
    try {
      let seq = timed;
      let origin = 0;
      const lp = loopRef.current;
      if (lp) {
        const t0 = lp.start * spb;
        const t1 = lp.end * spb;
        seq = timed
          .filter((n) => n.start + n.duration > t0 && n.start < t1)
          .map((n) => ({
            ...n,
            start: Math.max(0, n.start - t0),
            duration:
              Math.min(n.start + n.duration, t1) - Math.max(n.start, t0),
          }))
          .filter((n) => n.duration > 0.01);
        origin = t0;
        if (!seq.length) {
          setError("Loop is empty.");
          return;
        }
      }
      setPlayOrigin(origin);
      setPlaying(true);
      playingRef.current = true;
      setError(null);
      await engineRef.current.play(seq);
    } catch (err) {
      setPlaying(false);
      playingRef.current = false;
      setError(err instanceof Error ? err.message : "Playback failed");
    }
  };

  useEffect(() => {
    playAgainRef.current = () => {
      void startPlayback();
    };
  });

  const togglePlay = async () => {
    if (playing) {
      playingRef.current = false;
      engineRef.current.stop();
      setPlaying(false);
      setPlayPos(0);
      return;
    }
    await startPlayback();
  };

  const exportWav = async () => {
    if (!timed.length) return;
    setBusy(true);
    try {
      const blob = await engineRef.current.renderWav(timed);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `clavier-${Date.now()}.wav`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === "Escape") {
        setSelection(null);
        setDraft(null);
      }
      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        void togglePlay();
      }
      if ((e.key === "Backspace" || e.key === "Delete") && selection) {
        e.preventDefault();
        setNotes((prev) => prev.filter((n) => !overlaps(n, selection)));
      }
      if (e.key >= "1" && e.key <= "8" && !e.metaKey && !e.ctrlKey) {
        const vel = [16, 32, 48, 64, 80, 96, 112, 127][Number(e.key) - 1];
        setVelocity(vel);
        if (selection) {
          setNotes((prev) =>
            prev.map((n) => (overlaps(n, selection) ? { ...n, velocity: vel } : n))
          );
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const finishSelect = (a: number, b: number) => {
    if (Math.abs(b - a) <= 1) {
      const bar = Math.floor(a / 16) * 16;
      setSelection({ start: bar, end: bar + 16 });
    } else {
      setSelection(normRange(a, b));
    }
    setDraft(null);
  };

  const onRollPointerDown = async (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || draft?.kind === "drop") return;
    const el = e.currentTarget;
    const col = colFromEvent(e, el);
    const drawing = tool === "draw" && !e.shiftKey;

    if (!drawing) {
      el.setPointerCapture(e.pointerId);
      setDraft({ kind: "select", start: col, end: col + 1 });
      return;
    }

    const row = Math.floor((e.clientY - el.getBoundingClientRect().top) / CELL_H);
    const pitch = PITCH_TOP - row;
    if (pitch < PITCH_BOTTOM || pitch > PITCH_TOP) return;

    const hit = notes.find(
      (n) => n.pitch === pitch && col >= n.cell && col < n.cell + n.cells
    );
    if (hit) {
      setNotes((prev) => prev.filter((n) => n.id !== hit.id));
      return;
    }

    el.setPointerCapture(e.pointerId);
    setDraft({ kind: "draw", pitch, start: col, end: col + 1 });
    try {
      await engineRef.current.noteOn(pitch, velocity);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Synth failed");
    }
  };

  const onRollPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const col = colFromEvent(e, e.currentTarget);
    if (draft?.kind === "draw") setDraft({ ...draft, end: col + 1 });
    if (draft?.kind === "select") setDraft({ ...draft, end: col + 1 });
  };

  const release = (el: HTMLElement, id: number) => {
    try {
      el.releasePointerCapture(id);
    } catch {
      /* already released */
    }
  };

  const onRollPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    if (draft?.kind === "select") {
      finishSelect(draft.start, draft.end);
      release(e.currentTarget, e.pointerId);
      return;
    }
    if (draft?.kind !== "draw") return;
    engineRef.current.noteOff(draft.pitch);
    const range = normRange(draft.start, draft.end);
    setNotes((prev) => [
      ...prev,
      {
        id: noteId++,
        pitch: draft.pitch,
        cell: range.start,
        cells: range.end - range.start,
        velocity,
        source: "user",
      },
    ]);
    setDraft(null);
    release(e.currentTarget, e.pointerId);
  };

  const onRulerPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const col = colFromEvent(e, e.currentTarget);
    e.currentTarget.setPointerCapture(e.pointerId);
    setDraft({ kind: "loop", start: col, end: col + 1 });
  };

  const onRulerPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (draft?.kind !== "loop") return;
    setDraft({ ...draft, end: colFromEvent(e, e.currentTarget) + 1 });
  };

  const onRulerPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    if (draft?.kind !== "loop") return;
    const a = draft.start;
    const b = draft.end;
    const col = Math.min(a, b);
    if (Math.abs(b - a) <= 1) {
      if (loop && col >= loop.start && col < loop.end) {
        /* click inside loop: keep */
      } else {
        setLoop(null);
      }
    } else {
      setLoop(normRange(a, b));
    }
    setDraft(null);
    release(e.currentTarget, e.pointerId);
  };

  const onPedalPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const col = colFromEvent(e, e.currentTarget);
    const hit = pedals.find((p) => col >= p.start && col < p.end);
    if (hit && !e.shiftKey) {
      setPedals((prev) => prev.filter((p) => p !== hit));
      return;
    }
    e.currentTarget.setPointerCapture(e.pointerId);
    setDraft({ kind: "pedal", start: col, end: col + 1 });
  };

  const onPedalPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (draft?.kind !== "pedal") return;
    setDraft({ ...draft, end: colFromEvent(e, e.currentTarget) + 1 });
  };

  const onPedalPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    if (draft?.kind !== "pedal") return;
    setPedals((prev) => mergeRanges([...prev, normRange(draft.start, draft.end)]));
    setDraft(null);
    release(e.currentTarget, e.pointerId);
  };

  const onDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    const el = rollRef.current;
    if (!el) return;
    setDraft({ kind: "drop", start: colFromEvent(e, el) });
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    const raw =
      e.dataTransfer.getData("text/plain") || e.dataTransfer.getData("text/progression");
    const id = raw.replace(/^ab-prog:/, "");
    const el = rollRef.current;
    const at = el ? colFromEvent(e, el) : 0;
    setDraft(null);
    stampProgression(id, at);
  };

  const audition = async (pitch: number, on: boolean) => {
    setHeldKey(on ? pitch : null);
    try {
      if (on) await engineRef.current.noteOn(pitch, velocity);
      else engineRef.current.noteOff(pitch);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Synth failed");
    }
  };

  const setVel = (vel: number) => {
    setVelocity(vel);
    if (selection) {
      setNotes((prev) =>
        prev.map((n) => (overlaps(n, selection) ? { ...n, velocity: vel } : n))
      );
    }
  };

  const drawPreview =
    draft?.kind === "draw"
      ? { pitch: draft.pitch, ...normRange(draft.start, draft.end) }
      : null;
  const dropPreview =
    draft?.kind === "drop" ? { start: draft.start, end: draft.start + PROG_BARS * 16 } : null;

  const modelSelect =
    checkpoints.length > 0 ? (
      <Select
        value={checkpoint || null}
        onValueChange={(v) => {
          if (!v) return;
          setCheckpoint(v);
          if (!dev || typeof window === "undefined") return;
          const picked = checkpoints.find((c) => c.path === v);
          const stem = String(picked?.name || "").replace(/\.pt$/i, "");
          const url = new URL(window.location.href);
          if (stem) url.searchParams.set("ckpt", stem);
          else url.searchParams.delete("ckpt");
          window.history.replaceState(null, "", url);
        }}
        items={Object.fromEntries(
          checkpoints.map((c) => [c.path, checkpointLabel(c, { verbose: dev })])
        )}
      >
        <SelectTrigger className={dev ? "w-[220px]" : "w-[168px]"} size="sm">
          <SelectValue placeholder="Model">
            {(value: string | null) => {
              const c = checkpoints.find((x) => x.path === value);
              return c ? checkpointLabel(c, { verbose: dev }) : "Model";
            }}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {checkpoints.map((c) => (
            <SelectItem key={c.path} value={c.path}>
              {checkpointLabel(c, { verbose: dev })}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    ) : null;

  const gap =
    liveSel &&
    notes.some((n) => n.cell + n.cells <= liveSel.start) &&
    notes.some((n) => n.cell >= liveSel.end);

  return (
    <div className="flex h-full flex-col bg-background text-foreground">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b px-3">
        <Link href={dev ? routes.app : routes.home} className="text-[15px] font-semibold tracking-tight">
          clavier
        </Link>
        {dev ? (
          <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
            dev
          </span>
        ) : null}
        <Separator orientation="vertical" className="h-4" />
        <div className="flex items-center gap-2">
          <span className="w-8 text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
            {tempo}
          </span>
          <Slider
            className="w-20"
            min={60}
            max={160}
            value={[tempo]}
            onValueChange={(v) => setTempo(Math.round(asNum(v)))}
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
            vel {velocity}
          </span>
          <Slider
            className="w-20"
            min={1}
            max={127}
            value={[velocity]}
            onValueChange={(v) => setVel(Math.round(asNum(v)))}
          />
        </div>
        <Separator orientation="vertical" className="h-4" />
        <div className="flex items-center gap-0.5 rounded-md border p-0.5">
          <Button
            variant={tool === "select" ? "secondary" : "ghost"}
            size="xs"
            onClick={() => setTool("select")}
          >
            Select
          </Button>
          <Button
            variant={tool === "draw" ? "secondary" : "ghost"}
            size="xs"
            onClick={() => setTool("draw")}
          >
            Draw
          </Button>
        </div>
        <Separator orientation="vertical" className="h-4" />
        <div className="flex min-w-0 items-center gap-1.5">
          {PROGRESSIONS.map((p) => (
            <button
              key={p.id}
              type="button"
              draggable
              onClick={() => stampProgression(p.id, selection?.start ?? contentEnd)}
              onDragStart={(e) => {
                e.dataTransfer.setData("text/plain", `ab-prog:${p.id}`);
                e.dataTransfer.effectAllowed = "copy";
              }}
              onDragEnd={() => setDraft(null)}
              className="flex h-7 cursor-grab items-center gap-1.5 rounded-md border border-border bg-muted/40 px-2 text-xs active:cursor-grabbing"
              title="Click to drop, or drag onto the roll"
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          {error ? (
            <span className="max-w-[240px] truncate text-xs text-destructive">{error}</span>
          ) : null}
          {modelSelect}
          <Button variant="outline" size="sm" onClick={togglePlay}>
            {playing ? "Stop" : "Play"}
          </Button>
          <Button size="sm" disabled={busy} onClick={continueAfter}>
            {busy ? "Filling…" : "Continue"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={busy || !timed.length}
            onClick={exportWav}
          >
            Export
          </Button>
        </div>
      </header>

      {liveSel ? (
        <div className="flex h-10 shrink-0 items-center gap-2 border-b px-3 text-xs">
          <span className="text-muted-foreground">
            select {Math.floor(liveSel.start / 16) + 1}–{Math.max(Math.ceil(liveSel.end / 16), 1)}
            {gap ? " · gap" : ""}
          </span>
          <Button size="sm" disabled={busy} onClick={() => fillRange(liveSel)}>
            {busy ? "Filling…" : gap ? "Fill gap" : "Fill with AI"}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setSelection(null)}>
            Clear
          </Button>
        </div>
      ) : null}

      <div
        ref={wrapRef}
        className="roll-wrap min-h-0 flex-1"
        onDragOver={onDragOver}
        onDragLeave={() => {
          if (draft?.kind === "drop") setDraft(null);
        }}
        onDrop={onDrop}
      >
        <div
          className="ab-board"
          style={{
            gridTemplateColumns: `${PITCH_W}px ${rollW}px`,
            gridTemplateRows: `${RULER_H}px ${rollH}px ${PEDAL_H}px`,
          }}
        >
          <div className="ab-corner" />
          <div
            className="ab-ruler"
            style={{ width: rollW, height: RULER_H }}
            onPointerDown={onRulerPointerDown}
            onPointerMove={onRulerPointerMove}
            onPointerUp={onRulerPointerUp}
            onPointerCancel={onRulerPointerUp}
          >
            {Array.from({ length: Math.ceil(totalCols / 16) }, (_, bar) => (
              <div
                key={bar}
                className="ab-ruler-bar"
                style={{ left: bar * 16 * CELL_W, width: 16 * CELL_W }}
              >
                {bar + 1}
              </div>
            ))}
            {liveLoop && (
              <div
                className="ab-loop"
                style={{
                  left: liveLoop.start * CELL_W,
                  width: Math.max(CELL_W, (liveLoop.end - liveLoop.start) * CELL_W),
                }}
              />
            )}
          </div>

          <div className="ab-keys" style={{ width: PITCH_W }}>
            {Array.from({ length: ROWS }, (_, r) => {
              const p = PITCH_TOP - r;
              const black = isBlack(p);
              return (
                <div
                  key={p}
                  className={cn(
                    "ab-key",
                    black ? "black" : "white",
                    p % 12 === 0 && "c",
                    heldKey === p && "held"
                  )}
                  style={{ height: CELL_H }}
                  onPointerDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    void audition(p, true);
                  }}
                  onPointerUp={() => void audition(p, false)}
                  onPointerLeave={() => {
                    if (heldKey === p) void audition(p, false);
                  }}
                >
                  {!black && p % 12 === 0 ? pitchName(p) : ""}
                </div>
              );
            })}
          </div>

          <div
            ref={rollRef}
            className={cn("roll", tool === "select" && "selecting")}
            style={{ width: rollW, height: rollH }}
            onPointerDown={onRollPointerDown}
            onPointerMove={onRollPointerMove}
            onPointerUp={onRollPointerUp}
            onPointerCancel={onRollPointerUp}
          >
            {Array.from({ length: ROWS }, (_, r) => {
              const p = PITCH_TOP - r;
              return (
                <div
                  key={p}
                  className={cn("roll-row absolute left-0", isBlack(p) && "black")}
                  style={{ top: r * CELL_H, height: CELL_H, width: rollW }}
                />
              );
            })}
            {Array.from({ length: totalCols + 1 }, (_, c) => (
              <div
                key={c}
                className={cn("roll-grid", c % 16 === 0 && "bar", c % 4 === 0 && "beat")}
                style={{ left: c * CELL_W, height: rollH }}
              />
            ))}
            {notes.map((n) => (
              <div
                key={n.id}
                className={cn("roll-note", n.source === "ai" ? "gen" : "seed")}
                title={`${pitchName(n.pitch)} · v${n.velocity}`}
                style={{
                  left: n.cell * CELL_W + 1,
                  top: (PITCH_TOP - n.pitch) * CELL_H + 1,
                  width: n.cells * CELL_W - 2,
                  height: CELL_H - 2,
                  opacity: 0.35 + 0.65 * (n.velocity / 127),
                }}
              />
            ))}
            {drawPreview && (
              <div
                className="roll-note seed opacity-70"
                style={{
                  left: drawPreview.start * CELL_W + 1,
                  top: (PITCH_TOP - drawPreview.pitch) * CELL_H + 1,
                  width: (drawPreview.end - drawPreview.start) * CELL_W - 2,
                  height: CELL_H - 2,
                }}
              />
            )}
            {dropPreview && (
              <div
                className="roll-drop"
                style={{
                  left: dropPreview.start * CELL_W,
                  width: (dropPreview.end - dropPreview.start) * CELL_W,
                  height: rollH,
                }}
              />
            )}
            {liveSel && (
              <div
                className="roll-sel"
                style={{
                  left: liveSel.start * CELL_W,
                  width: Math.max(CELL_W, (liveSel.end - liveSel.start) * CELL_W),
                  height: rollH,
                }}
              />
            )}
            {playing && (
              <div
                className="roll-playhead"
                style={{
                  left: ((playOrigin + playPos) / spb) * CELL_W,
                  height: rollH,
                }}
              />
            )}
          </div>

          <div className="ab-pedal-label">Ped</div>
          <div
            className="ab-pedal"
            style={{ width: rollW, height: PEDAL_H }}
            onPointerDown={onPedalPointerDown}
            onPointerMove={onPedalPointerMove}
            onPointerUp={onPedalPointerUp}
            onPointerCancel={onPedalPointerUp}
          >
            {pedals.map((p, i) => (
              <div
                key={`${p.start}-${p.end}-${i}`}
                className="ab-pedal-seg"
                style={{
                  left: p.start * CELL_W,
                  width: Math.max(4, (p.end - p.start) * CELL_W),
                }}
              />
            ))}
            {livePedal && (
              <div
                className="ab-pedal-seg"
                style={{
                  left: livePedal.start * CELL_W,
                  width: Math.max(4, (livePedal.end - livePedal.start) * CELL_W),
                  opacity: 0.7,
                }}
              />
            )}
          </div>
        </div>
        {notes.length === 0 && !draft && (
          <p className="ab-empty">
            Scroll the 88-key piano. Select on the roll, loop on the bar ruler.
            Pedal lane at the bottom · velocity in the top bar.
          </p>
        )}
      </div>
    </div>
  );
}
