"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Button, buttonVariants } from "@/components/ui/button";
import { generatePhrase, type SynthNote } from "@/lib/api";
import { apiUrl, routes } from "@/lib/routes";
import { PRESETS, SynthEngine } from "@/lib/synth";
import { cn } from "@/lib/utils";

const MOODS = [
  { id: "none", label: "Any" },
  { id: "Q1", label: "Bright" },
  { id: "Q2", label: "Tense" },
  { id: "Q3", label: "Dark" },
  { id: "Q4", label: "Calm" },
] as const;

export function Playground() {
  const engineRef = useRef(new SynthEngine());
  const [mood, setMood] = useState<(typeof MOODS)[number]["id"]>("none");
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<SynthNote[]>([]);
  const [midiUrl, setMidiUrl] = useState<string | null>(null);

  useEffect(() => {
    engineRef.current.setParams(PRESETS["Neon Keys"]);
    engineRef.current.onEnded = () => setPlaying(false);
    return () => {
      engineRef.current.stop();
    };
  }, []);

  const generate = useCallback(async () => {
    setBusy(true);
    setError(null);
    engineRef.current.stop();
    setPlaying(false);
    try {
      const out = await generatePhrase({
        emotion: mood,
        max_new_tokens: 256,
        temperature: 1.05,
      });
      const next = out.notes || [];
      if (!next.length) throw new Error("The model returned silence. Try again.");
      setNotes(next);
      setMidiUrl(out.midi_url ? apiUrl(out.midi_url) : null);
      await engineRef.current.play(next);
      setPlaying(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not generate.");
    } finally {
      setBusy(false);
    }
  }, [mood]);

  const play = useCallback(async () => {
    if (!notes.length) return;
    try {
      await engineRef.current.play(notes);
      setPlaying(true);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not play.");
    }
  }, [notes]);

  const stop = useCallback(() => {
    engineRef.current.stop();
    setPlaying(false);
  }, []);

  return (
    <section className="mx-auto flex min-h-[70vh] max-w-[720px] flex-col items-center px-6 py-16 text-center sm:py-20">
      <p className="mb-3 text-xs font-semibold tracking-[0.18em] text-muted-foreground uppercase">
        Playground
      </p>
      <h1 className="text-4xl font-bold tracking-[-0.04em] sm:text-5xl">
        One click. A phrase.
      </h1>
      <p className="mt-4 max-w-md text-muted-foreground">
        No grid. No sketch. Prelude writes something you can hear right now.
      </p>

      <div className="mt-7 flex flex-wrap justify-center gap-2">
        {MOODS.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => setMood(m.id)}
            className={cn(
              "rounded-full border px-3 py-1 text-xs font-medium tracking-wide transition-colors",
              mood === m.id
                ? "border-foreground/40 bg-foreground text-background"
                : "border-white/10 text-muted-foreground hover:border-white/25 hover:text-foreground"
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      <button
        type="button"
        onClick={generate}
        disabled={busy}
        aria-label={busy ? "Generating" : "Generate a phrase"}
        className={cn(
          "group relative mt-12 flex size-44 items-center justify-center rounded-full sm:size-52",
          "outline-none focus-visible:ring-2 focus-visible:ring-white/40",
          "disabled:cursor-wait"
        )}
      >
        <span
          aria-hidden
          className={cn(
            "absolute inset-[-18%] rounded-full bg-[radial-gradient(circle,oklch(1_0_0/0.18),transparent_70%)] blur-xl transition-opacity",
            busy ? "animate-pulse opacity-100" : "opacity-70 group-hover:opacity-100"
          )}
        />
        <span
          aria-hidden
          className={cn(
            "absolute inset-0 rounded-full border border-white/12 bg-black/40 shadow-[0_0_80px_oklch(1_0_0/0.08)]",
            busy && "animate-pulse"
          )}
        />
        <img
          src="/logo.png?v=7"
          alt=""
          width={208}
          height={208}
          className={cn(
            "relative size-[78%] object-contain transition-transform duration-500",
            busy ? "scale-95 opacity-80" : "group-hover:scale-105"
          )}
        />
      </button>

      <p className="mt-6 text-sm text-muted-foreground">
        {busy ? "Writing…" : notes.length ? "Again, whenever you want." : "Click the mark."}
      </p>

      {error && (
        <p className="mt-4 max-w-md text-sm text-destructive" role="alert">
          {error}
        </p>
      )}

      {!!notes.length && (
        <div className="mt-10 w-full">
          <PhraseRoll notes={notes} />
          <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
            {playing ? (
              <Button onClick={stop}>Stop</Button>
            ) : (
              <Button onClick={play}>Play</Button>
            )}
            <Button variant="outline" onClick={generate} disabled={busy}>
              Again
            </Button>
            {midiUrl && (
              <a
                href={midiUrl}
                className={buttonVariants({ variant: "ghost" })}
                download="notate.midi"
              >
                MIDI
              </a>
            )}
          </div>
        </div>
      )}

      <p className="mt-14 text-sm text-muted-foreground">
        Want to sketch the first bars yourself?{" "}
        <Link href={routes.app} className="text-foreground underline-offset-4 hover:underline">
          Open clavier
        </Link>
      </p>
    </section>
  );
}

function PhraseRoll({ notes }: { notes: SynthNote[] }) {
  const { viewBox, rects } = useMemo(() => layoutNotes(notes), [notes]);
  return (
    <div className="overflow-hidden rounded-2xl ring-1 ring-white/10">
      <svg className="block w-full" viewBox={viewBox} role="img">
        <title>generated phrase</title>
        <rect width="100%" height="100%" fill="#0c0c0c" />
        {rects.map((r, i) => (
          <rect
            key={i}
            x={r.x}
            y={r.y}
            width={r.w}
            height={r.h}
            rx="1.6"
            fill="#c45c26"
            opacity={0.55 + (r.velocity / 127) * 0.45}
          />
        ))}
      </svg>
    </div>
  );
}

function layoutNotes(notes: SynthNote[]) {
  const minP = Math.min(...notes.map((n) => n.pitch));
  const maxP = Math.max(...notes.map((n) => n.pitch));
  const end = Math.max(...notes.map((n) => n.start + n.duration), 1);
  const pad = 2;
  const lo = minP - pad;
  const hi = maxP + pad;
  const rows = Math.max(hi - lo, 8);
  const W = 280;
  const H = Math.min(120, Math.max(64, rows * 4.2));
  const rects = notes.map((n) => ({
    x: (n.start / end) * (W - 8) + 4,
    y: ((hi - n.pitch) / rows) * (H - 8) + 3,
    w: Math.max(2.4, (n.duration / end) * (W - 8)),
    h: Math.max(3.2, H / rows - 0.6),
    velocity: n.velocity,
  }));
  return { viewBox: `0 0 ${W} ${H}`, rects };
}
