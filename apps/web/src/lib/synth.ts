import type { SynthNote } from "./api";

const workletUrl = "/synth/worklet.js";

type EngineNode = AudioWorkletNode & { port: MessagePort };

export class SynthEngine {
  ctx: AudioContext | null = null;
  node: EngineNode | null = null;
  params: Record<string, number> = {};
  onPos: ((s: number) => void) | null = null;
  onEnded: (() => void) | null = null;

  async init() {
    if (this.ctx) return;
    this.ctx = new AudioContext();
    try {
      await this.ctx.audioWorklet.addModule(workletUrl);
    } catch {
      this.ctx.close();
      this.ctx = null;
      throw new Error("Couldn't load the synth. Refresh and try Play again.");
    }
    this.node = new AudioWorkletNode(this.ctx, "notelm-synth", {
      numberOfInputs: 0,
      outputChannelCount: [2],
    }) as EngineNode;
    this.node.connect(this.ctx.destination);
    this.node.port.onmessage = (e: MessageEvent) => {
      if (e.data.type === "pos" && this.onPos) this.onPos(e.data.seconds);
      if ((e.data.type === "ended" || e.data.type === "stopped") && this.onEnded)
        this.onEnded();
    };
    if (Object.keys(this.params).length) this.setParams(this.params);
  }

  async resume() {
    await this.init();
    if (this.ctx?.state === "suspended") await this.ctx.resume();
  }

  setParams(values: Record<string, number>) {
    this.params = { ...this.params, ...values };
    this.node?.port.postMessage({ type: "params", values: this.params });
  }

  async play(notes: SynthNote[]) {
    if (!notes.length) throw new Error("Nothing to play");
    await this.resume();
    if (!this.node) throw new Error("Synth failed to start");
    this.node.port.postMessage({ type: "sequence", notes });
  }

  stop() {
    this.node?.port.postMessage({ type: "stop" });
  }

  async noteOn(pitch: number, velocity = 100) {
    await this.resume();
    this.node?.port.postMessage({ type: "noteOn", pitch, velocity });
  }

  noteOff(pitch: number) {
    this.node?.port.postMessage({ type: "noteOff", pitch });
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
