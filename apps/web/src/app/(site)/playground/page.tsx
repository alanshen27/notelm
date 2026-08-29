import type { Metadata } from "next";
import { Playground } from "@/components/playground";

export const metadata: Metadata = {
  title: "Playground",
  description: "One click. A new piano phrase.",
};

export default function PlaygroundPage() {
  return <Playground />;
}
