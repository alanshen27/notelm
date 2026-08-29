"use client";

import { useEffect } from "react";
import { routes } from "@/lib/routes";

/** Fallback if the host does not apply FastAPI / Next redirects. */
export default function AfterbarRedirect() {
  useEffect(() => {
    window.location.replace(routes.clavier);
  }, []);
  return null;
}
