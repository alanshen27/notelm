"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { buttonVariants } from "@/components/ui/button";
import { Logo } from "@/components/logo";
import { routes } from "@/lib/routes";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: routes.models, label: "Models" },
  { href: routes.research, label: "Lab" },
  { href: routes.clavier, label: "Clavier" },
];

export function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between border-b bg-background/90 px-6 py-3 backdrop-blur-md">
      <Logo />
      <nav className="flex items-center gap-5">
        {LINKS.map((link) => {
          const current =
            (link.href === routes.research && pathname.startsWith("/research")) ||
            (link.href === routes.clavier && pathname.startsWith("/clavier")) ||
            (link.href === routes.models && pathname.startsWith("/models"));
          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "text-sm font-medium",
                current
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {link.label}
            </Link>
          );
        })}
        <Link href={routes.app} className={buttonVariants({ size: "sm" })}>
          Open clavier
        </Link>
      </nav>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-auto flex flex-col items-start justify-between gap-2 border-t px-6 py-7 text-sm text-muted-foreground sm:flex-row sm:items-center">
      <Logo />
      <span>notate · clavier</span>
    </footer>
  );
}
