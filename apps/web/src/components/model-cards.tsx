import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { routes } from "@/lib/routes";

const MODELS = [
  {
    name: "Prelude",
    blurb: "The not-finished Chaconne. Play a few bars — it plays what comes next.",
    status: "available" as const,
    featured: true,
  },
  {
    name: "Chaconne",
    blurb: "Prelude, bigger. Still in the studio.",
    status: "soon" as const,
  },
  {
    name: "Canon",
    blurb: "Bigger again. Fills a gap in the middle of a phrase, not only the ending.",
    status: "soon" as const,
  },
  {
    name: "Sinfonia",
    blurb: "Bigger, and for more than piano.",
    status: "soon" as const,
  },
];

function StatusBadge({ status }: { status: "available" | "soon" }) {
  if (status === "available") {
    return <Badge variant="secondary">Available</Badge>;
  }
  return <Badge variant="outline">Coming soon</Badge>;
}

export function ModelCards() {
  const featured = MODELS.find((m) => m.featured) ?? MODELS[0];
  const rest = MODELS.filter((m) => m !== featured);

  return (
    <div className="grid gap-4 lg:grid-cols-[1.15fr_0.85fr]">
      <Card className="bg-primary text-primary-foreground ring-primary">
        <CardHeader>
          <CardTitle className="font-heading text-3xl font-bold">
            {featured.name}
          </CardTitle>
          <CardAction>
            <Badge variant="secondary">Available</Badge>
          </CardAction>
        </CardHeader>
        <CardContent className="space-y-5 text-[0.98rem] leading-relaxed opacity-90">
          <p>{featured.blurb}</p>
          <div className="flex flex-wrap gap-2">
            <Link
              href={routes.playground}
              className={buttonVariants({ variant: "secondary" })}
            >
              Hear it
            </Link>
            <Link
              href={routes.app}
              className={buttonVariants({ variant: "ghost" })}
            >
              Write in clavier
            </Link>
          </div>
        </CardContent>
      </Card>
      <div className="flex flex-col gap-4">
        {rest.map((m) => (
          <Card key={m.name}>
            <CardHeader>
              <CardTitle className="text-xl font-semibold">{m.name}</CardTitle>
              <CardAction>
                <StatusBadge status={m.status} />
              </CardAction>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">{m.blurb}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
