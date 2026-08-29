import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "clavier · dev",
  description: "Try any checkpoint.",
};

export default function DevLayout({ children }: { children: ReactNode }) {
  return (
    <div className="dark">
      <div className="h-dvh overflow-hidden bg-background text-foreground">
        {children}
      </div>
    </div>
  );
}
