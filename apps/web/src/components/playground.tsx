"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { generatePhrase, type SynthNote } from "@/lib/api";
import { routes } from "@/lib/routes";
import { PRESETS, SynthEngine } from "@/lib/synth";
import { LivingBlob } from "@/components/living-blob";
import { cn } from "@/lib/utils";

const MOODS = [
  { id: "none", label: "Any", chip: "border-foreground/40 bg-foreground text-background" },
  { id: "Q1", label: "Bright", chip: "border-[#d4a017] bg-[#f0c14b] text-[#3d2e00]" },
  { id: "Q2", label: "Tense", chip: "border-[#b33622] bg-[#d6452e] text-white" },
  { id: "Q3", label: "Dark", chip: "border-[#2a1d5c] bg-[#3a2a78] text-white" },
  { id: "Q4", label: "Calm", chip: "border-[#2b8a84] bg-[#3aa8a0] text-white" },
] as const;

export function Playground() {
  const engineRef = useRef(new SynthEngine());
  const [mood, setMood] = useState<(typeof MOODS)[number]["id"]>("none");
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [heard, setHeard] = useState(false);

  useEffect(() => {
    const engine = engineRef.current;
    engine.setParams(PRESETS["Neon Keys"]);
    engine.onEnded = () => setPlaying(false);
    return () => {
      engine.stop();
    };
  }, []);

  const generate = useCallback(async () => {
    engineRef.current.unlock();
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
      const next: SynthNote[] = out.notes || [];
      if (!next.length) throw new Error("The model returned silence. Try again.");
      await engineRef.current.play(next);
      setPlaying(true);
      setHeard(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not generate.");
    } finally {
      setBusy(false);
    }
  }, [mood]);

  return (
    <section className="mx-auto flex min-h-[70vh] max-w-[720px] flex-col items-center px-6 py-16 text-center sm:py-20">
      <p className="mb-3 text-xs font-semibold tracking-[0.18em] text-muted-foreground uppercase">
        Playground
      </p>
      <h1 className="text-4xl font-bold tracking-[-0.04em] sm:text-5xl">
        One click. A phrase.
      </h1>
      <p className="mt-4 max-w-md text-muted-foreground">
        No grid. No score. Tap the mark and just listen.
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
                ? m.chip
                : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground"
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="mt-10">
        <LivingBlob
          mood={mood}
          busy={busy}
          playing={playing}
          getLevel={() => engineRef.current.getLevel()}
          disabled={busy}
          onClick={generate}
          label={busy ? "Generating" : "Generate a phrase"}
        />
      </div>

      <p className="mt-2 text-sm text-muted-foreground">
        {busy ? "Writing…" : playing ? "Playing." : heard ? "Again, whenever you want." : "Click the mark."}
      </p>

      {error && (
        <p className="mt-4 max-w-md text-sm text-destructive" role="alert">
          {error}
        </p>
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
