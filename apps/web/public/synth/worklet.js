/**
 * notelm-synth — polyphonic subtractive synthesizer, written from scratch.
 *
 * Runs on the audio rendering thread as an AudioWorkletProcessor.
 * Signal path per voice:
 *   2x polyBLEP oscillator -> TPT state-variable lowpass (env + LFO modulated)
 *   -> ADSR VCA
 * Master path:
 *   voice sum -> ping-pong delay -> Schroeder reverb -> soft clip -> out (stereo)
 *
 * No Web Audio built-in nodes are used for synthesis; every sample is
 * computed by hand in process().
 */

const TWO_PI = 2 * Math.PI;
const MAX_VOICES = 12;
const MAX_DELAY_SEC = 1.5;

/* ------------------------------------------------------------ helpers */

function midiToFreq(p) {
  return 440 * Math.pow(2, (p - 69) / 12);
}

/** polyBLEP residual: smooths a unit step discontinuity at phase t (0..1). */
function polyBlep(t, dt) {
  if (t < dt) {
    const x = t / dt;
    return x + x - x * x - 1;
  }
  if (t > 1 - dt) {
    const x = (t - 1) / dt;
    return x * x + x + x + 1;
  }
  return 0;
}

/** One oscillator sample. wave: 0 saw, 1 square, 2 triangle, 3 sine. */
function oscSample(wave, phase, dt) {
  switch (wave) {
    case 0: {
      let v = 2 * phase - 1;
      v -= polyBlep(phase, dt);
      return v;
    }
    case 1: {
      let v = phase < 0.5 ? 1 : -1;
      v += polyBlep(phase, dt);
      v -= polyBlep((phase + 0.5) % 1, dt);
      return v;
    }
    case 2:
      return 4 * Math.abs(phase - 0.5) - 1;
    default:
      return Math.sin(TWO_PI * phase);
  }
}

/* ------------------------------------------------------------ envelope */

const ATTACK = 0, DECAY = 1, SUSTAIN = 2, RELEASE = 3, IDLE = 4;

class ADSR {
  constructor() {
    this.stage = IDLE;
    this.level = 0;
  }

  trigger() {
    this.stage = ATTACK;
  }

  release() {
    if (this.stage !== IDLE) this.stage = RELEASE;
  }

  /** a/d/r in seconds, s 0..1. Exponential-ish segments via one-pole coeffs. */
  next(a, d, s, r, sr) {
    switch (this.stage) {
      case ATTACK: {
        const rate = 1 / Math.max(1, a * sr);
        this.level += rate;
        if (this.level >= 1) {
          this.level = 1;
          this.stage = DECAY;
        }
        break;
      }
      case DECAY: {
        const coeff = Math.exp(-1 / Math.max(1, d * sr * 0.25));
        this.level = s + (this.level - s) * coeff;
        if (this.level - s < 1e-4) this.stage = SUSTAIN;
        break;
      }
      case SUSTAIN:
        this.level = s;
        break;
      case RELEASE: {
        const coeff = Math.exp(-1 / Math.max(1, r * sr * 0.25));
        this.level *= coeff;
        if (this.level < 1e-4) {
          this.level = 0;
          this.stage = IDLE;
        }
        break;
      }
      default:
        this.level = 0;
    }
    return this.level;
  }

  get active() {
    return this.stage !== IDLE;
  }
}

/* ------------------------------------------------------------ voice */

class Voice {
  constructor() {
    this.pitch = -1;
    this.gain = 0;
    this.age = 0;
    this.phase1 = 0;
    this.phase2 = 0;
    this.ampEnv = new ADSR();
    this.filtEnv = new ADSR();
    // TPT SVF integrator state
    this.ic1 = 0;
    this.ic2 = 0;
  }

  noteOn(pitch, velocity, age) {
    this.pitch = pitch;
    this.gain = Math.pow(velocity / 127, 1.5);
    this.age = age;
    this.phase1 = 0;
    this.phase2 = Math.random() * 0.3; // slight decorrelation
    this.ic1 = 0;
    this.ic2 = 0;
    this.ampEnv.trigger();
    this.filtEnv.trigger();
  }

  noteOff() {
    this.ampEnv.release();
    this.filtEnv.release();
  }

  get active() {
    return this.ampEnv.active;
  }
}

/* ------------------------------------------------------------ reverb */

class Comb {
  constructor(size) {
    this.buf = new Float32Array(size);
    this.idx = 0;
    this.filt = 0;
  }

  process(x, feedback, damp) {
    const y = this.buf[this.idx];
    this.filt = y * (1 - damp) + this.filt * damp;
    this.buf[this.idx] = x + this.filt * feedback;
    this.idx = (this.idx + 1) % this.buf.length;
    return y;
  }
}

class Allpass {
  constructor(size) {
    this.buf = new Float32Array(size);
    this.idx = 0;
  }

  process(x) {
    const y = this.buf[this.idx];
    const out = y - x;
    this.buf[this.idx] = x + y * 0.5;
    this.idx = (this.idx + 1) % this.buf.length;
    return out;
  }
}

class ReverbChannel {
  constructor(sr, offset) {
    const scale = sr / 44100;
    const combSizes = [1116, 1188, 1277, 1356].map((n) =>
      Math.round((n + offset) * scale)
    );
    const apSizes = [556, 441].map((n) => Math.round((n + offset) * scale));
    this.combs = combSizes.map((n) => new Comb(n));
    this.aps = apSizes.map((n) => new Allpass(n));
  }

  process(x, feedback, damp) {
    let y = 0;
    for (const c of this.combs) y += c.process(x, feedback, damp);
    y *= 0.25;
    for (const a of this.aps) y = a.process(y);
    return y;
  }
}

/* ------------------------------------------------------------ processor */

class NotelmSynth extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sr = sampleRate;
    this.voices = Array.from({ length: MAX_VOICES }, () => new Voice());
    this.voiceAge = 0;
    this.lfoPhase = 0;

    const dlen = Math.ceil(MAX_DELAY_SEC * this.sr);
    this.delayL = new Float32Array(dlen);
    this.delayR = new Float32Array(dlen);
    this.delayIdx = 0;

    this.revL = new ReverbChannel(this.sr, 0);
    this.revR = new ReverbChannel(this.sr, 23);

    this.events = []; // {frame, type, pitch, velocity} sorted by frame
    this.playing = false;
    this.lastPosPost = 0;
    this.seqStartFrame = 0;

    this.p = {
      osc1Wave: 0,
      osc2Wave: 0,
      osc2Detune: 7, // cents
      osc2Coarse: 0, // semitones
      oscMix: 0.5,
      ampA: 0.005,
      ampD: 0.25,
      ampS: 0.6,
      ampR: 0.3,
      filtA: 0.005,
      filtD: 0.3,
      filtS: 0.2,
      filtR: 0.3,
      cutoff: 1200,
      resonance: 0.25,
      envAmount: 2.5, // octaves
      keyTrack: 0.5,
      lfoRate: 5,
      lfoDepth: 0,
      lfoTarget: 0, // 0 off, 1 pitch, 2 cutoff
      delayTime: 0.375,
      delayFeedback: 0.35,
      delayMix: 0.18,
      reverbSize: 0.75,
      reverbDamp: 0.4,
      reverbMix: 0.22,
      masterGain: 0.8,
    };

    this.port.onmessage = (e) => this.handleMessage(e.data);
  }

  handleMessage(msg) {
    switch (msg.type) {
      case "params":
        Object.assign(this.p, msg.values);
        break;
      case "noteOn":
        this.noteOn(msg.pitch, msg.velocity ?? 100);
        break;
      case "noteOff":
        this.noteOff(msg.pitch);
        break;
      case "sequence": {
        // Replace any previous sequence so Play isn't fighting leftover events.
        this.events = [];
        for (const v of this.voices) v.noteOff();
        const start = currentFrame + Math.round(0.06 * this.sr);
        this.seqStartFrame = start;
        this.lastPosPost = 0;
        for (const n of msg.notes) {
          const on = start + Math.round(n.start * this.sr);
          const off = on + Math.max(1, Math.round(n.duration * this.sr));
          this.events.push({ frame: on, type: 1, pitch: n.pitch, velocity: n.velocity ?? 100 });
          this.events.push({ frame: off, type: 0, pitch: n.pitch });
        }
        this.events.sort((a, b) => a.frame - b.frame);
        this.playing = true;
        break;
      }
      case "stop":
        this.events = [];
        for (const v of this.voices) v.noteOff();
        this.playing = false;
        this.port.postMessage({ type: "stopped" });
        break;
      default:
        break;
    }
  }

  noteOn(pitch, velocity) {
    let voice =
      this.voices.find((v) => !v.active) ||
      this.voices.find((v) => v.ampEnv.stage === RELEASE) ||
      this.voices.reduce((a, b) => (a.age < b.age ? a : b));
    voice.noteOn(pitch, velocity, this.voiceAge++);
  }

  noteOff(pitch) {
    for (const v of this.voices) {
      if (v.pitch === pitch && v.active && v.ampEnv.stage !== RELEASE) {
        v.noteOff();
        return;
      }
    }
  }

  process(_inputs, outputs) {
    const outL = outputs[0][0];
    const outR = outputs[0][1] ?? outputs[0][0];
    const n = outL.length;
    const p = this.p;
    const sr = this.sr;

    // Fire scheduled events that fall inside this block.
    const blockEnd = currentFrame + n;
    while (this.events.length && this.events[0].frame < blockEnd) {
      const ev = this.events.shift();
      if (ev.type === 1) this.noteOn(ev.pitch, ev.velocity);
      else this.noteOff(ev.pitch);
    }

    const lfoInc = p.lfoRate / sr;
    const delaySamples = Math.min(
      this.delayL.length - 1,
      Math.max(1, Math.round(p.delayTime * sr))
    );
    const revFeedback = 0.7 + 0.28 * Math.min(1, Math.max(0, p.reverbSize));

    for (let i = 0; i < n; i++) {
      this.lfoPhase = (this.lfoPhase + lfoInc) % 1;
      const lfo = Math.sin(TWO_PI * this.lfoPhase) * p.lfoDepth;

      // ---- voices
      let dry = 0;
      for (const v of this.voices) {
        if (!v.active) continue;

        const amp = v.ampEnv.next(p.ampA, p.ampD, p.ampS, p.ampR, sr);
        const fenv = v.filtEnv.next(p.filtA, p.filtD, p.filtS, p.filtR, sr);

        const pitchMod = p.lfoTarget === 1 ? lfo * 0.5 : 0; // ±semitone/2 vibrato
        const f1 = midiToFreq(v.pitch + pitchMod);
        const f2 = midiToFreq(v.pitch + p.osc2Coarse + pitchMod + p.osc2Detune / 100);
        const dt1 = f1 / sr;
        const dt2 = f2 / sr;

        v.phase1 = (v.phase1 + dt1) % 1;
        v.phase2 = (v.phase2 + dt2) % 1;
        const osc =
          oscSample(p.osc1Wave, v.phase1, dt1) * (1 - p.oscMix) +
          oscSample(p.osc2Wave, v.phase2, dt2) * p.oscMix;

        // ---- TPT state-variable lowpass (Zavalishin), per-sample coeffs
        let fc =
          p.cutoff *
          Math.pow(2, fenv * p.envAmount + (p.lfoTarget === 2 ? lfo * 3 : 0)) *
          Math.pow(f1 / 261.63, p.keyTrack * 0.5);
        fc = Math.min(0.45 * sr, Math.max(20, fc));
        const g = Math.tan((Math.PI * fc) / sr);
        const k = 2 - 1.9 * Math.min(1, Math.max(0, p.resonance));
        const a1 = 1 / (1 + g * (g + k));
        const v1 = (v.ic1 + g * (osc - v.ic2)) * a1;
        const v2 = v.ic2 + g * v1;
        v.ic1 = 2 * v1 - v.ic1;
        v.ic2 = 2 * v2 - v.ic2;

        dry += v2 * amp * v.gain;
      }
      dry *= 0.35; // headroom for polyphony

      // ---- ping-pong delay
      const rIdx =
        (this.delayIdx - delaySamples + this.delayL.length) % this.delayL.length;
      const dl = this.delayL[rIdx];
      const drr = this.delayR[rIdx];
      this.delayL[this.delayIdx] = dry + drr * p.delayFeedback;
      this.delayR[this.delayIdx] = dl * p.delayFeedback;
      this.delayIdx = (this.delayIdx + 1) % this.delayL.length;

      const preL = dry + dl * p.delayMix;
      const preR = dry + drr * p.delayMix;

      // ---- reverb
      const revIn = (preL + preR) * 0.5;
      const wl = this.revL.process(revIn, revFeedback, p.reverbDamp);
      const wr = this.revR.process(revIn, revFeedback, p.reverbDamp);

      // ---- master: mix, soft clip
      const outSampleL = (preL + wl * p.reverbMix) * p.masterGain;
      const outSampleR = (preR + wr * p.reverbMix) * p.masterGain;
      outL[i] = Math.tanh(outSampleL);
      outR[i] = Math.tanh(outSampleR);
    }

    // ---- transport messages
    if (this.playing) {
      if (currentFrame - this.lastPosPost > sr / 30) {
        this.lastPosPost = currentFrame;
        this.port.postMessage({
          type: "pos",
          seconds: Math.max(0, (currentFrame - this.seqStartFrame) / sr),
        });
      }
      if (!this.events.length && !this.voices.some((v) => v.active)) {
        this.playing = false;
        this.port.postMessage({ type: "ended" });
      }
    }

    return true;
  }
}

registerProcessor("notelm-synth", NotelmSynth);
