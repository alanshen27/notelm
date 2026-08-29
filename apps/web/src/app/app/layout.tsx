import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "clavier",
  description: "Sketch a few bars. Prelude continues.",
};

export default function ClavierLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div className="dark h-dvh overflow-hidden bg-background text-foreground">
      {children}
    </div>
  );
}
