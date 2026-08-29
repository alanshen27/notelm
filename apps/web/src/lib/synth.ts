import type { SynthNote } from "./api";

const workletUrl = "/synth/worklet.js";

type EngineNode = AudioWorkletNode & { port: MessagePort };

function makeAudioContext(): AudioContext {
  const AC =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  return new AC();
}

export function normalizeNotes(notes: SynthNote[], snapStart = false): SynthNote[] {
  const clean = notes
    .map((n) => ({
      pitch: Number(n.pitch),
      start: Number(n.start),
      duration: Number(n.duration),
      velocity: Number(n.velocity) || 96,
    }))
    .filter(
      (n) =>
        Number.isFinite(n.pitch) &&
        Number.isFinite(n.start) &&
        Number.isFinite(n.duration) &&
        n.duration > 0
    )
    .map((n) => ({
      ...n,
      pitch: Math.min(108, Math.max(21, Math.round(n.pitch))),
      duration: Math.max(0.05, n.duration),
      velocity: Math.min(127, Math.max(1, n.velocity)),
    }));
  if (!clean.length) return [];
  if (!snapStart) return clean;
  const t0 = Math.min(...clean.map((n) => n.start));
  return clean.map((n) => ({ ...n, start: Math.max(0, n.start - t0) }));
}

export class SynthEngine {
  ctx: AudioContext | null = null;
  node: EngineNode | null = null;
  analyser: AnalyserNode | null = null;
  master: GainNode | null = null;
  params: Record<string, number> = {};
  onPos: ((s: number) => void) | null = null;
  onEnded: (() => void) | null = null;

  private bins = new Uint8Array(128);
  private fallbackOsc: OscillatorNode[] = [];
  private fallbackTimer: ReturnType<typeof setTimeout> | null = null;
  private hold: OscillatorNode | null = null;
  private initing: Promise<void> | null = null;

  /** Call synchronously from a click so iOS unlocks audio before any await. */
  unlock() {
    if (typeof window === "undefined") return;
    if (!this.ctx) this.ctx = makeAudioContext();
    if (this.ctx.state === "suspended") void this.ctx.resume();
    this.ensureMaster();
    this.keepAlive();
    try {
      const buf = this.ctx.createBuffer(1, 1, this.ctx.sampleRate);
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.master!);
      src.start(0);
    } catch {
      /* ignore */
    }
    void this.init();
  }

  /** Short blip so the tap is audible before generate returns. */
  tick() {
    this.unlock();
    if (!this.ctx || !this.master) return;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    const t = this.ctx.currentTime;
    osc.type = "sine";
    osc.frequency.value = 523.25;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.16, t + 0.012);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.1);
    osc.connect(g);
    g.connect(this.master);
    osc.start(t);
    osc.stop(t + 0.12);
  }

  private keepAlive() {
    if (this.hold || !this.ctx || !this.master) return;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.frequency.value = 40;
    g.gain.value = 0.00002;
    osc.connect(g);
    g.connect(this.master);
    osc.start();
    this.hold = osc;
  }

  private ensureMaster() {
    if (!this.ctx || this.master) return;
    this.master = this.ctx.createGain();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = 0.72;
    this.master.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
  }

  getLevel() {
    if (!this.analyser) return 0;
    if (this.bins.length !== this.analyser.frequencyBinCount) {
      this.bins = new Uint8Array(this.analyser.frequencyBinCount);
    }
    this.analyser.getByteFrequencyData(this.bins);
    let sum = 0;
    for (let i = 0; i < this.bins.length; i++) sum += this.bins[i];
    return sum / this.bins.length / 255;
  }

  async init() {
    if (this.node || !this.ctx) return;
    if (this.initing) {
      await this.initing;
      return;
    }
    this.initing = (async () => {
      try {
        if (!this.ctx) return;
        await this.ctx.audioWorklet.addModule(workletUrl);
        this.node = new AudioWorkletNode(this.ctx, "notelm-synth", {
          numberOfInputs: 0,
          outputChannelCount: [2],
        }) as EngineNode;
        this.node.connect(this.master!);
        this.node.port.onmessage = (e: MessageEvent) => {
          if (e.data.type === "pos" && this.onPos) this.onPos(e.data.seconds);
          if ((e.data.type === "ended" || e.data.type === "stopped") && this.onEnded)
            this.onEnded();
        };
        if (Object.keys(this.params).length) this.setParams(this.params);
      } catch {
        this.node = null;
      }
    })();
    await this.initing;
    this.initing = null;
  }

  async resume() {
    this.unlock();
    await this.init();
    if (this.ctx?.state === "suspended") await this.ctx.resume();
  }

  setParams(values: Record<string, number>) {
    this.params = { ...this.params, ...values };
    this.node?.port.postMessage({ type: "params", values: this.params });
  }

  async play(notes: SynthNote[], opts?: { simple?: boolean; snapStart?: boolean }) {
    const seq = normalizeNotes(notes, opts?.snapStart);
    if (!seq.length) throw new Error("Nothing to play");
    this.unlock();
    if (this.ctx?.state === "suspended") await this.ctx.resume();
    if (!this.ctx) throw new Error("Synth failed to start");
    this.clearFallback();
    // Playground (and any simple:true caller) uses oscillators so a long
    // generate fetch cannot leave us with a mute worklet node.
    if (!opts?.simple) {
      await this.init();
      if (this.node) {
        this.node.port.postMessage({ type: "sequence", notes: seq });
        return;
      }
    }
    this.playFallback(seq);
  }

  stop() {
    this.node?.port.postMessage({ type: "stop" });
    const hadFallback = this.fallbackOsc.length > 0 || this.fallbackTimer;
    this.clearFallback();
    if (hadFallback) this.onEnded?.();
  }

  async noteOn(pitch: number, velocity = 100) {
    await this.resume();
    this.node?.port.postMessage({ type: "noteOn", pitch, velocity });
  }

  noteOff(pitch: number) {
    this.node?.port.postMessage({ type: "noteOff", pitch });
  }

  private playFallback(notes: SynthNote[]) {
    const ctx = this.ctx!;
    const now = ctx.currentTime + 0.02;
    const bus = ctx.createGain();
    bus.gain.value = 0.38;
    bus.connect(this.master!);
    for (const n of notes) {
      const osc = ctx.createOscillator();
      osc.type = "triangle";
      osc.frequency.value = 440 * 2 ** ((n.pitch - 69) / 12);
      const g = ctx.createGain();
      const t0 = now + n.start;
      const t1 = t0 + Math.max(0.08, n.duration);
      const peak = Math.max(0.04, (n.velocity / 127) * 0.7);
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(peak, t0 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t1);
      osc.connect(g);
      g.connect(bus);
      osc.start(t0);
      osc.stop(t1 + 0.04);
      this.fallbackOsc.push(osc);
    }
    const end = Math.max(...notes.map((n) => n.start + n.duration), 0);
    this.fallbackTimer = setTimeout(() => this.onEnded?.(), end * 1000 + 280);
  }

  private clearFallback() {
    if (this.fallbackTimer) {
      clearTimeout(this.fallbackTimer);
      this.fallbackTimer = null;
    }
    for (const osc of this.fallbackOsc) {
      try {
        osc.stop();
      } catch {
        /* already stopped */
      }
    }
    this.fallbackOsc = [];
  }

  async renderWav(
    notes: SynthNote[],
    { tailSeconds = 3, sampleRate = 44100 } = {}
  ) {
    const end = Math.max(...notes.map((n) => n.start + n.duration), 0);
    const length = Math.ceil((end + tailSeconds) * sampleRate);
    const ctx = new OfflineAudioContext(2, length, sampleRate);
    await ctx.audioWorklet.addModule(workletUrl);
    const node = new AudioWorkletNode(ctx, "notelm-synth", {
      numberOfInputs: 0,
      outputChannelCount: [2],
    });
    node.connect(ctx.destination);
    node.port.postMessage({ type: "params", values: this.params });
    node.port.postMessage({ type: "sequence", notes });
    const buffer = await ctx.startRendering();
    return encodeWav(buffer);
  }
}

export function encodeWav(buffer: AudioBuffer) {
  const numCh = buffer.numberOfChannels;
  const len = buffer.length;
  const sr = buffer.sampleRate;
  const bytesPerSample = 2;
  const dataSize = len * numCh * bytesPerSample;
  const out = new DataView(new ArrayBuffer(44 + dataSize));

  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) out.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  out.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  out.setUint32(16, 16, true);
  out.setUint16(20, 1, true);
  out.setUint16(22, numCh, true);
  out.setUint32(24, sr, true);
  out.setUint32(28, sr * numCh * bytesPerSample, true);
  out.setUint16(32, numCh * bytesPerSample, true);
  out.setUint16(34, 16, true);
  writeStr(36, "data");
  out.setUint32(40, dataSize, true);

  const channels: Float32Array[] = [];
  for (let c = 0; c < numCh; c++) channels.push(buffer.getChannelData(c));
  let offset = 44;
  for (let i = 0; i < len; i++) {
    for (let c = 0; c < numCh; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      out.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([out.buffer], { type: "audio/wav" });
}

export const PRESETS: Record<string, Record<string, number>> = {
  "Neon Keys": {
    osc1Wave: 0, osc2Wave: 1, osc2Detune: 9, osc2Coarse: 0, oscMix: 0.35,
    ampA: 0.004, ampD: 0.35, ampS: 0.5, ampR: 0.35,
    filtA: 0.002, filtD: 0.4, filtS: 0.15, filtR: 0.3,
    cutoff: 900, resonance: 0.3, envAmount: 2.8, keyTrack: 0.6,
    lfoRate: 5.5, lfoDepth: 0.08, lfoTarget: 1,
    delayTime: 0.375, delayFeedback: 0.35, delayMix: 0.2,
    reverbSize: 0.7, reverbDamp: 0.4, reverbMix: 0.2, masterGain: 0.8,
  },
};
