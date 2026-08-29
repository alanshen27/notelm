"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

type Mood = "none" | "Q1" | "Q2" | "Q3" | "Q4";

type Props = {
  mood?: Mood;
  busy?: boolean;
  playing?: boolean;
  getLevel?: () => number;
  disabled?: boolean;
  onClick?: () => void;
  label: string;
};

export function LivingBlob({
  mood = "none",
  busy = false,
  playing = false,
  getLevel,
  disabled,
  onClick,
  label,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const stateRef = useRef({ playing, busy, getLevel });
  stateRef.current = { playing, busy, getLevel };

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    void video.play().catch(() => {
      /* autoplay can wait for the click */
    });
    let raf = 0;
    const tick = () => {
      const { playing: isPlaying, busy: isBusy, getLevel: levelOf } = stateRef.current;
      const kick = isPlaying ? 1.35 + (levelOf?.() ?? 0) * 1.4 : isBusy ? 1.7 : 1;
      video.playbackRate = kick;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <button
      type="button"
      onClick={() => {
        void videoRef.current?.play();
        onClick?.();
      }}
      disabled={disabled}
      aria-label={label}
      className={cn(
        "blob-mood relative size-48 overflow-hidden rounded-full sm:size-56",
        "outline-none focus-visible:ring-2 focus-visible:ring-foreground/25",
        "disabled:cursor-wait",
        `blob-mood-${mood}`
      )}
    >
      <video
        ref={videoRef}
        className="size-full object-cover"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        poster="/logo.png?v=7"
      >
        <source src="/blob.webm" type="video/webm" />
        <source src="/blob.mp4" type="video/mp4" />
      </video>
      <span className="blob-wash" aria-hidden />
    </button>
  );
}
