"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

type Props = {
  busy?: boolean;
  playing?: boolean;
  getLevel?: () => number;
  disabled?: boolean;
  onClick?: () => void;
  label: string;
};

export function LivingBlob({
  busy = false,
  playing = false,
  getLevel,
  disabled,
  onClick,
  label,
}: Props) {
  const wrapRef = useRef<HTMLButtonElement>(null);
  const stateRef = useRef({ playing, getLevel });
  stateRef.current = { playing, getLevel };

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    let raf = 0;
    const tick = () => {
      const { playing: isPlaying, getLevel: levelOf } = stateRef.current;
      if (isPlaying) {
        const n = 1 + (levelOf?.() ?? 0) * 0.28;
        el.style.setProperty("--blob-kick", String(n));
      } else {
        el.style.setProperty("--blob-kick", "1");
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <button
      ref={wrapRef}
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={cn(
        "blob-orb relative size-48 overflow-visible rounded-full sm:size-56",
        "outline-none focus-visible:ring-2 focus-visible:ring-foreground/25",
        "disabled:cursor-wait",
        busy && "blob-busy",
        playing && "blob-playing"
      )}
    >
      <span className="blob-halo" aria-hidden />
      <img src="/logo.png?v=7" alt="" className="blob-layer blob-layer-b" />
      <img src="/logo.png?v=7" alt="" className="blob-layer blob-layer-a" />
    </button>
  );
}
