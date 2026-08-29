import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { ModelCards } from "@/components/model-cards";
import { routes } from "@/lib/routes";

export default function HomePage() {
  return (
    <>
      <section className="mx-auto max-w-[920px] px-7 pb-10 pt-16 sm:pt-20">
        <p className="mb-4 text-xs font-semibold tracking-[0.14em] text-muted-foreground uppercase">
          A co-writer for piano
        </p>
        <h1 className="text-[clamp(2.8rem,8vw,5.4rem)] font-bold leading-[0.95] tracking-[-0.055em]">
          From a sketch
          <br />
          to the next bar.
        </h1>
        <p className="mt-7 max-w-xl text-lg leading-relaxed text-muted-foreground">
          You play a few bars. notate plays what comes next. Not a song from a
          prompt — a second pair of hands on the roll. The app is{" "}
          <strong className="font-semibold text-foreground">clavier</strong>.
        </p>
        <div className="mt-9 flex flex-wrap items-center gap-3">
          <Link href={routes.app} className={buttonVariants({ size: "lg" })}>
            Open clavier
          </Link>
          <Link
            href={routes.modelsHash}
            className={buttonVariants({ variant: "ghost", size: "lg" })}
          >
            See models
          </Link>
        </div>
      </section>

      <section className="px-7 pb-20" aria-hidden="true">
        <Card className="overflow-hidden bg-muted/40">
          <div className="flex justify-between border-b px-4 py-3 text-xs font-semibold tracking-wide uppercase">
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
          A co-writer, not a jukebox.
        </h2>
        <div className="grid gap-8 sm:grid-cols-3">
          {[
            ["01", "Sketch", "Drop chords or a melody on the roll. Those bars stay yours."],
            ["02", "Continue", "The model writes the next ones."],
            ["03", "Hear", "Play it back, or take the audio with you."],
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
              <h3 className="text-lg font-semibold">In the browser</h3>
              <p className="mt-2 text-muted-foreground">
                Open clavier, sketch, continue, export.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent>
              <h3 className="text-lg font-semibold">On the desktop</h3>
              <p className="mt-2 text-muted-foreground">
                Same editor, in its own window.
              </p>
            </CardContent>
          </Card>
        </div>
        <div className="mt-8">
          <Link href={routes.app} className={buttonVariants({ size: "lg" })}>
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
          fill={i % 2 ? "#f4f4f2" : "#fafaf8"}
        />
      ))}
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <line
          key={`b${i}`}
          x1={i * 40}
          y1="0"
          x2={i * 40}
          y2="96"
          stroke={i === 0 ? "#111" : "#e4e4e0"}
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
          fill={n.seed ? "#111" : "#c45c26"}
        />
      ))}
    </svg>
  );
}
