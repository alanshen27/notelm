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
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef({ busy, playing, getLevel });
  stateRef.current = { busy, playing, getLevel };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = "/logo.png?v=7";
    let raf = 0;
    let reduced = false;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const syncMotion = () => {
      reduced = mq.matches;
    };
    syncMotion();
    mq.addEventListener?.("change", syncMotion);

    const size = 360;
    canvas.width = size;
    canvas.height = size;

    const draw = (now: number) => {
      const { busy: isBusy, playing: isPlaying, getLevel: levelOf } = stateRef.current;
      const t = now / 1000;
      const level = isPlaying ? levelOf?.() ?? 0 : 0;
      const energy = reduced
        ? 0.08
        : isPlaying
          ? 0.85 + level * 1.8
          : isBusy
            ? 1.05
            : 0.55;
      const speed = reduced ? 0.18 : 0.85 + energy * 1.25;

      ctx.clearRect(0, 0, size, size);
      ctx.save();
      ctx.translate(size / 2, size / 2);

      const glow = 0.1 + energy * 0.18;
      const grd = ctx.createRadialGradient(0, 0, 20, 0, 0, size * 0.46);
      grd.addColorStop(0, `rgba(20,20,18,${0.08 + energy * 0.12})`);
      grd.addColorStop(1, "rgba(20,20,18,0)");
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.48, 0, Math.PI * 2);
      ctx.fill();

      const copies = reduced ? 1 : 6;
      for (let i = 0; i < copies; i++) {
        const phase = t * speed + i * 1.15;
        const dx = Math.cos(phase) * (18 + energy * 28);
        const dy = Math.sin(phase * 1.35 + i) * (16 + energy * 24);
        const scale =
          0.7 +
          0.12 * Math.sin(t * (1.2 + energy) + i) +
          energy * 0.08 +
          glow * 0.12;
        const rot = 0.12 * Math.sin(t * 0.7 + i * 0.4) + energy * 0.08 * Math.sin(t * 2.2 + i);
        ctx.save();
        ctx.globalAlpha = reduced ? 1 : 0.22 + i * 0.14;
        ctx.translate(dx, dy);
        ctx.rotate(rot);
        ctx.scale(scale, scale * (1 + 0.04 * Math.sin(phase)));
        if (img.complete && img.naturalWidth) {
          ctx.drawImage(img, -size * 0.36, -size * 0.36, size * 0.72, size * 0.72);
        }
        ctx.restore();
      }

      ctx.restore();
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      mq.removeEventListener?.("change", syncMotion);
    };
  }, []);

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={cn(
        "group relative flex size-48 items-center justify-center rounded-full sm:size-56",
        "outline-none focus-visible:ring-2 focus-visible:ring-foreground/25",
        "disabled:cursor-wait"
      )}
    >
      <canvas
        ref={canvasRef}
        className="size-full"
        width={360}
        height={360}
      />
    </button>
  );
}
