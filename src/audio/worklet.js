// worklet.js — AudioWorklet processors for the live campfire audio. Load once per context with
//   await ctx.audioWorklet.addModule(new URL('./worklet.js', import.meta.url))
// (static imports are allowed in worklet modules; dynamic import() is not; needs http(s)/localhost).
//
// 'fire-audio'  : 0 inputs, 2 outputs ([0] dry stereo, [1] reverb send stereo -> ConvolverNode).
//   main -> worklet messages:
//     { type:'ctl', intensity, hrr, dhrr, camDist, camAz, wind }      k-rate controls (smoothed in the DSP)
//     { type:'event', event:{ type, strength, pan, ..., when } }      when = AudioContext time (s); omitted = now
//     { type:'param', name, value }                                    engine parameter (e.g. popRate, mix)
//     { type:'run', on }                                               pause/resume rendering (outputs silence)
//     { type:'dispose' }
//   worklet -> main: { type:'events', count, data: Float64Array }  packed records (EventLog.pack; decodeEvents() on the
//     main thread gives { type, t, n, contextTime, strength, size, pan, x, y, z, stereoPan, source, id }).
//     contextTime = AudioContext time at which the event's first sample is rendered.
// 'fire-master' : 1 stereo input -> EQ, make-up, leveler, 4x oversampled soft clip, true-peak lookahead limiter
//   (never exceeds the ceiling; ~3.2 ms latency = LiveMaster.latencySamples()).
//   main -> worklet: { type:'makeup', db } | { type:'gain', value }
//   worklet -> main (every ~100 ms): { type:'meter', peakDb, rmsDb, grDb (limiter), levelerDb }
// process() allocates nothing on the JS heap: the DSP is allocation-free and both messages are preallocated objects
// (postMessage serialises them; the serialised copy lives outside the worklet's JS heap).
import { FireAudioEngine, LiveMaster, EVF } from './fire-dsp.js';

const EV_MAX = 32; // events per message (more wait for the next render quantum)

class FireProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const po = (options && options.processorOptions) || {};
    this.engine = new FireAudioEngine({ sampleRate, seed: po.seed ?? 1, params: po.params || {} });
    if (po.controls) this.engine.setControls(po.controls);
    this.running = po.running !== false;
    this.alive = true;
    this.logRead = 0;
    this.offset = currentFrame; // context frame of engine sample 0 (updated every quantum)
    this.scratch = new Float32Array(128);
    this.evMsg = { type: 'events', count: 0, data: new Float64Array(EV_MAX * EVF) };
    this.port.onmessage = (e) => this.onMessage(e.data);
  }
  onMessage(m) {
    const e = this.engine;
    switch (m && m.type) {
      case 'ctl': e.setControls(m); break;
      case 'event': {
        const ev = m.event || {};
        const n = typeof ev.when === 'number' && isFinite(ev.when) ? Math.round(ev.when * sampleRate) - this.offset : e.n;
        e.scheduleAt(ev, n);
        break;
      }
      case 'param': e.setParam(m.name, m.value); break;
      case 'run': this.running = !!m.on; break;
      case 'dispose': this.alive = false; break;
    }
  }
  process(inputs, outputs) {
    if (!this.alive) return false;
    const dry = outputs[0], send = outputs[1], n = dry[0].length; // never assume 128
    if (!this.running) { // outputs are zero-initialised by the spec; keep the engine clock aligned: rendering
      // resumes at engine sample e.n in the *next* quantum (frame currentFrame + n), so events posted meanwhile land right
      this.offset = currentFrame + n - this.engine.n;
      return true;
    }
    if (this.scratch.length < n) this.scratch = new Float32Array(n);
    const e = this.engine;
    this.offset = currentFrame - e.n;
    e.render(dry[0], dry[1] || this.scratch, send[0], send[1] || this.scratch, n);
    if (e.log.w > this.logRead) {
      const m = this.evMsg, d = m.data;
      m.count = e.log.pack(this.logRead, d, EV_MAX);
      this.logRead = e.log.packedTo;
      for (let k = 0; k < m.count; k++) d[k * EVF + 13] = (d[k * EVF + 1] + this.offset) / sampleRate; // contextTime
      this.port.postMessage(m);
    }
    return true;
  }
}

class MasterProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const po = (options && options.processorOptions) || {};
    this.m = new LiveMaster(sampleRate, po);
    this.alive = true;
    this.pk = 0; this.ss = 0; this.cnt = 0; this.every = Math.round(0.1 * sampleRate);
    this.meterMsg = { type: 'meter', peakDb: 0, rmsDb: 0, grDb: 0, levelerDb: 0 };
    this.port.onmessage = (e) => {
      const d = e.data || {};
      if (d.type === 'makeup') this.m.setMakeupDb(d.db);
      else if (d.type === 'gain') this.m.gainTarget = Math.max(0, +d.value || 0);
      else if (d.type === 'dispose') this.alive = false;
    };
  }
  process(inputs, outputs) {
    if (!this.alive) return false;
    const inp = inputs[0], out = outputs[0], n = out[0].length;
    const l = inp && inp.length ? inp[0] : null, r = inp && inp.length > 1 ? inp[1] : l;
    this.m.process(l, r, out[0], out[1] || null, n);
    const oL = out[0], oR = out[1] || out[0];
    for (let i = 0; i < n; i++) {
      const a = Math.abs(oL[i]), b = Math.abs(oR[i]);
      if (a > this.pk) this.pk = a; if (b > this.pk) this.pk = b;
      this.ss += 0.5 * (oL[i] * oL[i] + oR[i] * oR[i]);
    }
    this.cnt += n;
    if (this.cnt >= this.every) {
      const mm = this.meterMsg;
      mm.peakDb = 20 * Math.log10(this.pk + 1e-12); mm.rmsDb = 10 * Math.log10(this.ss / this.cnt + 1e-24); mm.grDb = -20 * Math.log10(this.m.grMin); mm.levelerDb = this.m.levDb;
      this.port.postMessage(mm);
      this.pk = 0; this.ss = 0; this.cnt = 0; this.m.grMin = 1;
    }
    return true;
  }
}

registerProcessor('fire-audio', FireProcessor);
registerProcessor('fire-master', MasterProcessor);
