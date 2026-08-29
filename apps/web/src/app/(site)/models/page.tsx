import { ModelCards } from "@/components/model-cards";

export default function ModelsPage() {
  return (
    <section className="mx-auto max-w-[1100px] px-7 py-16">
      <p className="mb-4 text-xs font-semibold tracking-[0.14em] text-muted-foreground uppercase">
        Models
      </p>
      <h1 className="max-w-[16ch] text-4xl font-bold tracking-tight sm:text-5xl">
        Pick a voice. Keep writing.
      </h1>
      <p className="mt-6 max-w-xl text-lg text-muted-foreground">
        Each model can write a phrase from nothing, or continue what you play
        in clavier.
      </p>
      <div className="mt-10">
        <ModelCards />
      </div>
    </section>
  );
}
