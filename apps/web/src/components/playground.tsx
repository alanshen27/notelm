"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { fetchCheckpoints, generatePhrase, type SynthNote } from "@/lib/api";
import {
  PLAYGROUND_VOICES,
  playgroundVoices,
  voiceIsReady,
  type PlaygroundVoice,
} from "@/lib/checkpoints";
import { routes } from "@/lib/routes";
import { PRESETS, SynthEngine } from "@/lib/synth";
import { LivingBlob } from "@/components/living-blob";
import { cn } from "@/lib/utils";

const DEFAULT_VOICES: PlaygroundVoice[] = PLAYGROUND_VOICES.map((v) => ({
  stem: v.stem,
  label: v.label,
  path: "",
}));

const MOODS = [
  { id: "none", label: "Any", swatch: "bg-neutral-800" },
  { id: "Q1", label: "Bright", swatch: "bg-[#f0c14b]" },
  { id: "Q2", label: "Tense", swatch: "bg-[#d6452e]" },
  { id: "Q3", label: "Dark", swatch: "bg-[#3a2a78]" },
  { id: "Q4", label: "Calm", swatch: "bg-[#3aa8a0]" },
] as const;

const RING = 40;

const STAND_IN: SynthNote[] = [
  { pitch: 60, start: 0, duration: 0.45, velocity: 100 },
  { pitch: 64, start: 0, duration: 0.45, velocity: 92 },
  { pitch: 67, start: 0, duration: 0.45, velocity: 92 },
  { pitch: 72, start: 0.5, duration: 0.4, velocity: 104 },
  { pitch: 67, start: 1.0, duration: 0.35, velocity: 90 },
  { pitch: 64, start: 1.4, duration: 0.55, velocity: 88 },
  { pitch: 60, start: 2.05, duration: 0.9, velocity: 96 },
];

export function Playground() {
  const engineRef = useRef(new SynthEngine());
  const [mood, setMood] = useState<(typeof MOODS)[number]["id"]>("none");
  const [voices, setVoices] = useState<PlaygroundVoice[]>(DEFAULT_VOICES);
  const [voice, setVoice] = useState("prelude");
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

  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get("model")?.toLowerCase();
    if (q) setVoice(q);
    fetchCheckpoints()
      .then((data) => {
        const rows = playgroundVoices(data.checkpoints || []);
        setVoices(rows);
        setVoice((cur) => {
          const next = q || cur;
          const hit = rows.find((v) => v.stem === next);
          if (hit && voiceIsReady(hit)) return hit.stem;
          return rows.find(voiceIsReady)?.stem ?? "prelude";
        });
      })
      .catch(() => {
        /* keep the catalog; Prelude still generates via the API default */
      });
  }, []);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const pulse = () => {
      const el = document.getElementById("playground-audio");
      if (el) el.dataset.level = engineRef.current.getLevel().toFixed(3);
      raf = requestAnimationFrame(pulse);
    };
    raf = requestAnimationFrame(pulse);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const generate = useCallback(async () => {
    const engine = engineRef.current;
    engine.unlock();
    engine.stop();
    engine.tick();
    setBusy(true);
    setError(null);
    setPlaying(false);
    let next: SynthNote[] = [];
    try {
      const picked = voices.find((v) => v.stem === voice);
      const out = await generatePhrase({
        emotion: mood,
        checkpoint: picked?.path || undefined,
        max_new_tokens: 256,
        temperature: 1.05,
      });
      next = out.notes || [];
      if (!next.length) throw new Error("The model returned silence. Try again.");
    } catch (err) {
      next = STAND_IN;
      setError(
        err instanceof Error
          ? `${err.message} Playing a stand-in so you can still hear something.`
          : "Writer offline. Playing a stand-in."
      );
    }
    try {
      if (engine.ctx?.state === "suspended") await engine.ctx.resume();
      await engine.play(next, { simple: true, snapStart: true });
      setPlaying(true);
      setHeard(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not play.");
    } finally {
      setBusy(false);
    }
  }, [mood, voice, voices]);

  const selected = MOODS.find((m) => m.id === mood) ?? MOODS[0];
  const selectedVoice = voices.find((v) => v.stem === voice) ?? voices[0];
  const line = `${selectedVoice?.label ?? "Prelude"} · ${selected.label}`;

  const pickVoice = (next: PlaygroundVoice) => {
    if (!voiceIsReady(next)) return;
    engineRef.current.unlock();
    setVoice(next.stem);
    const url = new URL(window.location.href);
    url.searchParams.set("model", next.stem);
    window.history.replaceState(null, "", url);
  };

  return (
    <section className="mx-auto flex min-h-[70vh] max-w-[720px] flex-col items-center px-6 py-16 text-center sm:py-20">
      <p className="mb-3 text-xs font-semibold tracking-[0.18em] text-muted-foreground uppercase">
        Playground
      </p>
      <h1 className="text-4xl font-bold tracking-[-0.04em] sm:text-5xl">
        One click. A phrase.
      </h1>
      <p className="mt-4 max-w-md text-muted-foreground">
        Pick a model and a mood, then tap the mark. It writes a short line and plays it.
      </p>

      <div
        role="radiogroup"
        aria-label="Model"
        className="mt-8 flex flex-wrap justify-center gap-2"
      >
        {voices.map((v) => {
          const on = voice === v.stem;
          const ready = voiceIsReady(v);
          return (
            <button
              key={v.stem}
              type="button"
              role="radio"
              aria-checked={on}
              aria-label={ready ? v.label : `${v.label}, coming soon`}
              disabled={!ready}
              onClick={() => pickVoice(v)}
              className={cn(
                "rounded-full border px-3.5 py-1.5 text-xs font-medium tracking-wide transition-colors",
                on
                  ? "border-foreground bg-foreground text-background"
                  : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
                !ready && "cursor-not-allowed opacity-40 hover:border-border hover:text-muted-foreground"
              )}
            >
              {v.label}
              {!ready && <span className="ml-1.5 font-normal">soon</span>}
            </button>
          );
        })}
      </div>

      <div
        role="radiogroup"
        aria-label="Mood"
        className="relative mx-auto mt-10 size-[24.5rem] sm:size-[28rem]"
      >
        <div
          aria-hidden
          className="pointer-events-none absolute left-1/2 top-1/2 size-[80%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-border"
        />

        {MOODS.map((m, i) => {
          const rad = ((-90 + (i * 360) / MOODS.length) * Math.PI) / 180;
          const active = mood === m.id;
          return (
            <button
              key={m.id}
              type="button"
              role="radio"
              aria-checked={active}
              aria-label={m.label}
              onClick={() => {
                engineRef.current.unlock();
                setMood(m.id);
              }}
              className="absolute z-10 size-14 -translate-x-1/2 -translate-y-1/2 overflow-visible"
              style={{
                left: `${50 + RING * Math.cos(rad)}%`,
                top: `${50 + RING * Math.sin(rad)}%`,
              }}
            >
              <span
                className={cn(
                  "absolute left-1/2 top-1/2 size-9 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 shadow-sm transition-transform sm:size-10",
                  m.swatch,
                  active
                    ? "scale-110 border-foreground"
                    : "border-white opacity-80 hover:scale-105 hover:opacity-100"
                )}
              />
              <span
                className={cn(
                  "pointer-events-none absolute left-1/2 top-1/2 whitespace-nowrap text-[10px] font-medium tracking-wide sm:text-xs",
                  active ? "text-foreground" : "text-muted-foreground"
                )}
                style={{
                  transform: `translate(-50%, -50%) translate(${Math.cos(rad) * 28}px, ${Math.sin(rad) * 28}px)`,
                }}
              >
                {m.label}
              </span>
            </button>
          );
        })}

        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
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
      </div>

      <p
        id="playground-audio"
        className="mt-1 text-sm text-muted-foreground"
        data-busy={busy ? "1" : "0"}
        data-playing={playing ? "1" : "0"}
      >
        {busy
          ? `Writing · ${line}`
          : playing
            ? `Playing · ${line}`
            : heard
              ? `${line}. Again, whenever you want.`
              : `${line}. Click the mark.`}
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
