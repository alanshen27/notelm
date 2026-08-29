"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { buttonVariants } from "@/components/ui/button";
import { Logo } from "@/components/logo";
import { routes } from "@/lib/routes";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: routes.playground, label: "Playground" },
  { href: routes.models, label: "Models" },
  { href: routes.clavier, label: "Clavier" },
  { href: routes.research, label: "Lab" },
];

export function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between border-b bg-background/90 px-5 py-3 backdrop-blur-md sm:px-7">
      <Logo glow />
      <nav className="flex items-center gap-4 sm:gap-6">
        {LINKS.map((link) => {
          const current =
            pathname === link.href || pathname.startsWith(link.href);
          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "hidden text-sm font-medium sm:inline",
                current
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {link.label}
            </Link>
          );
        })}
        <Link href={routes.playground} className={buttonVariants({ size: "sm" })}>
          Generate
        </Link>
      </nav>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-auto flex flex-col items-start justify-between gap-3 border-t px-5 py-8 text-sm text-muted-foreground sm:flex-row sm:items-center sm:px-7">
      <Logo glow />
      <span>one click, or a sketch</span>
    </footer>
  );
}
