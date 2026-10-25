// fire-worklet.js — AudioWorkletProcessor wrapper around the shared DSP core.
// Load with: await ctx.audioWorklet.addModule('fire-worklet.js')   (static imports are allowed in worklets,
// dynamic import() is not). Outputs: [0] = dry stereo, [1] = reverb send stereo (-> ConvolverNode).
import { FireAudioEngine } from './fire-audio-dsp.mjs';

class FireProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'intensity', defaultValue: 0.8, minValue: 0, maxValue: 1.6, automationRate: 'k-rate' },
      { name: 'wind', defaultValue: 0, minValue: 0, maxValue: 1.5, automationRate: 'k-rate' },
      { name: 'distance', defaultValue: 2, minValue: 0.4, maxValue: 30, automationRate: 'k-rate' },
    ];
  }
  constructor(options) {
    super();
    const po = options.processorOptions || {};
    // sampleRate is a global of AudioWorkletGlobalScope
    this.engine = new FireAudioEngine({ sampleRate, seed: po.seed || 1, params: po.params || {} });
    this.sent = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      // events carry an absolute context time; engine time == context time when the node starts at t = 0
      if (m.type === 'event') this.engine.schedule({ ...m.event, t: m.event.t - this.t0 });
      else if (m.type === 'param') this.engine.setParam(m.name, m.value);
    };
    this.t0 = currentTime; // context time at which engine sample 0 is rendered
  }
  process(inputs, outputs, params) {
    const dry = outputs[0], send = outputs[1];
    const e = this.engine;
    // k-rate params -> engine targets (engine smooths with tau = 150 ms)
    e.p.intensity = params.intensity[0]; e.p.wind = params.wind[0]; e.p.distance = params.distance[0];
    const n = dry[0].length; // always check the actual quantum size (it is not guaranteed to be 128)
    e.render(dry[0], dry[1] || dry[0], send[0], send[1] || send[0], n);
    // forward hero events (pops, log shifts, gusts) to the main thread for spark/flare sync
    if (e.eventLog.length > this.sent) {
      this.port.postMessage({ type: 'events', events: e.eventLog.slice(this.sent).map((x) => ({ ...x, contextTime: x.t + this.t0 })) });
      this.sent = e.eventLog.length;
      if (this.sent > 256) { e.eventLog.length = 0; this.sent = 0; } // bound memory in long sessions
    }
    return true;
  }
}
registerProcessor('fire-audio', FireProcessor);
