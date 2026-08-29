import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { routes } from "@/lib/routes";

export default function ClavierInfoPage() {
  return (
    <section className="mx-auto max-w-[1100px] px-7 py-16">
      <p className="mb-4 text-xs font-semibold tracking-[0.14em] text-muted-foreground uppercase">
        Clavier
      </p>
      <h1 className="max-w-[14ch] text-4xl font-bold tracking-tight sm:text-5xl">
        The piano roll that writes back.
      </h1>
      <p className="mt-6 max-w-xl text-lg text-muted-foreground">
        Sketch a progression or a melody. Ask the model to continue. Play it.
        Keep what you like.
      </p>
      <div className="mt-8 flex flex-wrap gap-3">
        <Link href={routes.app} className={buttonVariants({ size: "lg" })}>
          Open clavier
        </Link>
        <Link
          href={routes.playground}
          className={buttonVariants({ variant: "outline", size: "lg" })}
        >
          Just generate
        </Link>
      </div>
      <div className="mt-14 grid gap-4 sm:grid-cols-3">
        <Card>
          <CardContent>
            <h2 className="mb-2 text-lg font-semibold">Draw</h2>
            <p className="text-muted-foreground">
              Click-drag notes, or drop a chord progression onto the grid.
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <h2 className="mb-2 text-lg font-semibold">Continue</h2>
            <p className="text-muted-foreground">
              Select a few bars. Fill with AI. Your notes stay; the next ones
              arrive in amber.
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent>
            <h2 className="mb-2 text-lg font-semibold">Hear</h2>
            <p className="text-muted-foreground">
              Play it in the browser, or export audio to take with you.
            </p>
          </CardContent>
        </Card>
      </div>
    </section>
  );
}
