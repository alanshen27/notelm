import type { ReactNode } from "react";
import { SiteFooter, SiteNav } from "@/components/site-nav";

export default function SiteLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <SiteNav />
      <main className="flex-1">{children}</main>
      <SiteFooter />
    </>
  );
}
