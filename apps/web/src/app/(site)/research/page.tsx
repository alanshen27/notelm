export default function ResearchPage() {
  return (
    <section className="mx-auto max-w-[720px] px-7 py-16">
      <p className="mb-4 text-xs font-semibold tracking-[0.14em] text-muted-foreground uppercase">
        Lab
      </p>
      <h1 className="max-w-[16ch] text-4xl font-bold tracking-tight sm:text-5xl">
        A co-writer, not a jukebox.
      </h1>
      <div className="mt-8 space-y-6 text-lg leading-relaxed text-muted-foreground">
        <p>
          Most music AI asks for a sentence and returns a finished track. notate
          does the opposite. You write some notes. It writes the next ones.
        </p>
        <p>
          You stay on the piano roll. You keep the sketch. The model is a second
          pair of hands, not a black box that scores the whole song.
        </p>
        <p>
          Prelude is out now — the not-finished Chaconne. Canon, Chaconne, and
          Sinfonia are bigger models on the way.
        </p>
      </div>
    </section>
  );
}
