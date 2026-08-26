/**
 * Main-thread wrapper for the notelm-synth AudioWorklet.
 * Handles context lifecycle, live playback, and offline WAV rendering.
 */

const workletUrl = new URL("./worklet.js", import.meta.url);

export class SynthEngine {
  constructor() {
    this.ctx = null;
    this.node = null;
    this.params = {};
    this.onPos = null;
    this.onEnded = null;
  }

  async init() {
    if (this.ctx) return;
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule(workletUrl);
    this.node = new AudioWorkletNode(this.ctx, "notelm-synth", {
      numberOfInputs: 0,
      outputChannelCount: [2],
    });
    this.node.connect(this.ctx.destination);
    this.node.port.onmessage = (e) => {
      if (e.data.type === "pos" && this.onPos) this.onPos(e.data.seconds);
      if ((e.data.type === "ended" || e.data.type === "stopped") && this.onEnded)
        this.onEnded();
    };
    if (Object.keys(this.params).length) this.setParams(this.params);
  }

  async resume() {
    await this.init();
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  setParams(values) {
    this.params = { ...this.params, ...values };
    this.node?.port.postMessage({ type: "params", values: this.params });
  }

  async play(notes) {
    await this.resume();
    this.node.port.postMessage({ type: "sequence", notes });
  }

  stop() {
    this.node?.port.postMessage({ type: "stop" });
  }

  async noteOn(pitch, velocity = 100) {
    await this.resume();
    this.node.port.postMessage({ type: "noteOn", pitch, velocity });
  }

  noteOff(pitch) {
    this.node?.port.postMessage({ type: "noteOff", pitch });
  }

  /** Render notes offline through the same DSP and return a WAV blob. */
  async renderWav(notes, { tailSeconds = 3, sampleRate = 44100 } = {}) {
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

/** Interleave an AudioBuffer into a 16-bit PCM WAV blob. */
export function encodeWav(buffer) {
  const numCh = buffer.numberOfChannels;
  const len = buffer.length;
  const sr = buffer.sampleRate;
  const bytesPerSample = 2;
  const dataSize = len * numCh * bytesPerSample;
  const out = new DataView(new ArrayBuffer(44 + dataSize));

  const writeStr = (offset, s) => {
    for (let i = 0; i < s.length; i++) out.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  out.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  out.setUint32(16, 16, true);
  out.setUint16(20, 1, true); // PCM
  out.setUint16(22, numCh, true);
  out.setUint32(24, sr, true);
  out.setUint32(28, sr * numCh * bytesPerSample, true);
  out.setUint16(32, numCh * bytesPerSample, true);
  out.setUint16(34, 16, true);
  writeStr(36, "data");
  out.setUint32(40, dataSize, true);

  const channels = [];
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

export const PRESETS = {
  "Neon Keys": {
    osc1Wave: 0, osc2Wave: 1, osc2Detune: 9, osc2Coarse: 0, oscMix: 0.35,
    ampA: 0.004, ampD: 0.35, ampS: 0.5, ampR: 0.35,
    filtA: 0.002, filtD: 0.4, filtS: 0.15, filtR: 0.3,
    cutoff: 900, resonance: 0.3, envAmount: 2.8, keyTrack: 0.6,
    lfoRate: 5.5, lfoDepth: 0.08, lfoTarget: 1,
    delayTime: 0.375, delayFeedback: 0.35, delayMix: 0.2,
    reverbSize: 0.7, reverbDamp: 0.4, reverbMix: 0.2, masterGain: 0.8,
  },
  "Soft Pad": {
    osc1Wave: 0, osc2Wave: 0, osc2Detune: 12, osc2Coarse: 0, oscMix: 0.5,
    ampA: 0.4, ampD: 0.5, ampS: 0.8, ampR: 1.2,
    filtA: 0.5, filtD: 0.8, filtS: 0.5, filtR: 1.0,
    cutoff: 500, resonance: 0.15, envAmount: 1.5, keyTrack: 0.4,
    lfoRate: 0.6, lfoDepth: 0.25, lfoTarget: 2,
    delayTime: 0.5, delayFeedback: 0.3, delayMix: 0.12,
    reverbSize: 0.95, reverbDamp: 0.5, reverbMix: 0.4, masterGain: 0.75,
  },
  Pluck: {
    osc1Wave: 1, osc2Wave: 0, osc2Detune: 4, osc2Coarse: 12, oscMix: 0.25,
    ampA: 0.001, ampD: 0.18, ampS: 0.0, ampR: 0.18,
    filtA: 0.001, filtD: 0.12, filtS: 0.05, filtR: 0.12,
    cutoff: 600, resonance: 0.45, envAmount: 3.5, keyTrack: 0.7,
    lfoRate: 5, lfoDepth: 0, lfoTarget: 0,
    delayTime: 0.25, delayFeedback: 0.45, delayMix: 0.28,
    reverbSize: 0.6, reverbDamp: 0.35, reverbMix: 0.18, masterGain: 0.85,
  },
  "Warm Bass": {
    osc1Wave: 0, osc2Wave: 1, osc2Detune: 3, osc2Coarse: -12, oscMix: 0.45,
    ampA: 0.003, ampD: 0.25, ampS: 0.7, ampR: 0.15,
    filtA: 0.002, filtD: 0.2, filtS: 0.25, filtR: 0.15,
    cutoff: 300, resonance: 0.35, envAmount: 2.2, keyTrack: 0.3,
    lfoRate: 5, lfoDepth: 0, lfoTarget: 0,
    delayTime: 0.375, delayFeedback: 0.1, delayMix: 0.05,
    reverbSize: 0.4, reverbDamp: 0.6, reverbMix: 0.08, masterGain: 0.85,
  },
};
