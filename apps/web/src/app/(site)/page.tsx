import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { Logo } from "@/components/logo";
import { ModelCards } from "@/components/model-cards";
import { routes } from "@/lib/routes";

export default function HomePage() {
  return (
    <>
      <section className="mx-auto flex max-w-[920px] flex-col items-center px-7 pb-14 pt-14 text-center sm:pt-20">
        <Logo href={routes.playground} wordmark={false} size="hero" glow />
        <p className="mt-8 text-xs font-semibold tracking-[0.18em] text-muted-foreground uppercase">
          A co-writer for piano
        </p>
        <h1 className="mt-4 text-[clamp(2.6rem,7vw,5rem)] font-bold leading-[0.95] tracking-[-0.055em]">
          Hear a phrase.
          <br />
          Or write one.
        </h1>
        <p className="mt-6 max-w-lg text-lg leading-relaxed text-muted-foreground">
          One click for a new idea. Or open the roll and keep what you play —
          notate writes the next bars.
        </p>
        <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
          <Link href={routes.playground} className={buttonVariants({ size: "lg" })}>
            Generate
          </Link>
          <Link
            href={routes.app}
            className={buttonVariants({ variant: "ghost", size: "lg" })}
          >
            Open clavier
          </Link>
        </div>
      </section>

      <section className="px-7 pb-20" aria-hidden="true">
        <Card className="overflow-hidden bg-card/70">
          <div className="flex justify-between border-b border-white/8 px-4 py-3 text-xs font-semibold tracking-wide uppercase">
            <span>prelude</span>
            <span className="text-muted-foreground">continuing</span>
          </div>
          <div className="relative overflow-hidden">
            <PianoRollMock />
            <div className="hero-playhead" />
          </div>
        </Card>
      </section>

      <section
        id="models"
        className="mx-auto max-w-[1100px] scroll-mt-24 px-7 pb-24"
      >
        <p className="mb-4 text-xs font-semibold tracking-[0.14em] text-muted-foreground uppercase">
          Models
        </p>
        <h2 className="mb-10 max-w-[16ch] text-4xl font-bold tracking-tight sm:text-5xl">
          Pick a voice. Keep writing.
        </h2>
        <ModelCards />
      </section>

      <section className="mx-auto max-w-[1100px] px-7 pb-24">
        <p className="mb-4 text-xs font-semibold tracking-[0.14em] text-muted-foreground uppercase">
          How it works
        </p>
        <h2 className="mb-10 max-w-[16ch] text-4xl font-bold tracking-tight sm:text-5xl">
          A click, or a sketch.
        </h2>
        <div className="grid gap-8 sm:grid-cols-3">
          {[
            ["01", "Generate", "Hit one button. Prelude writes a phrase. Play it."],
            ["02", "Sketch", "Or drop chords on the roll. Those bars stay yours."],
            ["03", "Continue", "The model writes the next ones. Hear it, keep it."],
          ].map(([n, title, body]) => (
            <article key={n}>
              <span className="text-xs font-semibold tracking-[0.12em] text-muted-foreground">
                {n}
              </span>
              <h3 className="mt-2.5 mb-2 text-xl font-semibold tracking-tight">
                {title}
              </h3>
              <p className="m-0 text-muted-foreground">{body}</p>
            </article>
          ))}
        </div>
      </section>

      <section
        id="clavier"
        className="mx-auto max-w-[1100px] scroll-mt-24 px-7 pb-24"
      >
        <p className="mb-4 text-xs font-semibold tracking-[0.14em] text-muted-foreground uppercase">
          The instrument
        </p>
        <h2 className="mb-4 max-w-[16ch] text-4xl font-bold tracking-tight sm:text-5xl">
          clavier is where you write.
        </h2>
        <p className="mb-10 max-w-xl text-lg text-muted-foreground">
          A dark piano roll. Your notes, then the model’s. Play, edit, go again.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <Card>
            <CardContent>
              <h3 className="text-lg font-semibold">Playground</h3>
              <p className="mt-2 text-muted-foreground">
                No grid. One click. A phrase you can play or take with you.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <h3 className="text-lg font-semibold">Clavier</h3>
              <p className="mt-2 text-muted-foreground">
                Sketch, continue, export. The co-writer.
              </p>
            </CardContent>
          </Card>
        </div>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link href={routes.playground} className={buttonVariants({ size: "lg" })}>
            Open playground
          </Link>
          <Link
            href={routes.app}
            className={buttonVariants({ variant: "outline", size: "lg" })}
          >
            Launch clavier
          </Link>
        </div>
      </section>
    </>
  );
}

function PianoRollMock() {
  const notes = [
    { x: 6, y: 58, w: 36, seed: true },
    { x: 6, y: 46, w: 36, seed: true },
    { x: 6, y: 34, w: 36, seed: true },
    { x: 48, y: 64, w: 36, seed: true },
    { x: 48, y: 52, w: 36, seed: true },
    { x: 48, y: 40, w: 36, seed: true },
    { x: 92, y: 70, w: 28 },
    { x: 108, y: 46, w: 22 },
    { x: 124, y: 58, w: 40 },
    { x: 148, y: 34, w: 18 },
    { x: 168, y: 52, w: 32 },
    { x: 188, y: 40, w: 24 },
    { x: 212, y: 64, w: 36 },
    { x: 236, y: 28, w: 20 },
  ];
  return (
    <svg className="block w-full" viewBox="0 0 280 96" role="img">
      <title>a sketched phrase, continued</title>
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <rect
          key={i}
          x="0"
          y={i * 12}
          width="280"
          height="12"
          fill={i % 2 ? "#141414" : "#0e0e0e"}
        />
      ))}
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <line
          key={`b${i}`}
          x1={i * 40}
          y1="0"
          x2={i * 40}
          y2="96"
          stroke={i === 0 ? "#3a3a3a" : "#222"}
          strokeWidth={i % 2 === 0 ? 1 : 0.5}
        />
      ))}
      {notes.map((n, i) => (
        <rect
          key={i}
          x={n.x}
          y={n.y}
          width={n.w}
          height="8"
          rx="1.5"
          fill={n.seed ? "#e8e4dc" : "#c45c26"}
        />
      ))}
    </svg>
  );
}
