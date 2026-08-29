"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { ClavierApp } from "@/components/clavier-app";

function DevClavier() {
  const params = useSearchParams();
  const ckpt = params.get("ckpt") || params.get("model") || "";
  return <ClavierApp dev ckptQuery={ckpt} />;
}

export default function DevPage() {
  return (
    <Suspense>
      <DevClavier />
    </Suspense>
  );
}
