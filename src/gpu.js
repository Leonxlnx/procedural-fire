// WebGPU helpers: device init, textures, pipelines, uniform structs.

export async function initGPU() {
  if (!navigator.gpu) throw new Error('WebGPU not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter');
  const want = ['timestamp-query', 'float32-filterable', 'shader-f16', 'rg11b10ufloat-renderable', 'texture-formats-tier1', 'texture-formats-tier2'];
  const requiredFeatures = want.filter(f => adapter.features.has(f));
  // 32-bit float textures are sampled with filtering and the accumulator is read_write rgba32float
  const mustHave = ['float32-filterable', 'texture-formats-tier2'];
  const missing = mustHave.filter(f => !adapter.features.has(f));
  if (missing.length) throw new Error(`This WebGPU adapter lacks required feature(s): ${missing.join(', ')}. Use a recent Chrome/Edge on a GPU that supports them.`);
  const L = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageTexturesPerShaderStage: Math.min(8, L.maxStorageTexturesPerShaderStage),
      maxStorageBuffersPerShaderStage: Math.min(10, L.maxStorageBuffersPerShaderStage),
      maxComputeInvocationsPerWorkgroup: Math.min(512, L.maxComputeInvocationsPerWorkgroup),
      maxComputeWorkgroupSizeX: Math.min(512, L.maxComputeWorkgroupSizeX),
      maxBufferSize: Math.min(1 << 30, L.maxBufferSize),
      maxStorageBufferBindingSize: Math.min(1 << 30, L.maxStorageBufferBindingSize),
    },
  });
  device.lost.then(info => console.error('WebGPU device lost:', info.message));
  device.addEventListener?.('uncapturederror', e => console.error('WebGPU error:', e.error.message));
  return { adapter, device, features: new Set(requiredFeatures) };
}

export const TU = () => GPUTextureUsage;
export const BU = () => GPUBufferUsage;

export function tex3D(device, [x, y, z], format = 'rgba16float', label = '') {
  return device.createTexture({
    label, size: [x, y, z], dimension: '3d', format,
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
}

export function tex2D(device, [w, h], format = 'rgba16float', label = '', extraUsage = 0) {
  return device.createTexture({
    label, size: [w, h], format,
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | extraUsage,
  });
}

// Builds a WGSL struct of f32 fields + a matching Float32Array writer.
export class UniformStruct {
  constructor(device, name, fields, label = name) {
    this.name = name;
    this.fields = fields;
    this.index = Object.fromEntries(fields.map((f, i) => [f, i]));
    const n = Math.ceil(fields.length / 4) * 4;
    this.data = new Float32Array(n);
    this.buffer = device.createBuffer({ label, size: n * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device = device;
  }
  wgsl() {
    return `struct ${this.name} {\n${this.fields.map(f => `  ${f}: f32,`).join('\n')}\n${
      Array.from({ length: this.data.length - this.fields.length }, (_, i) => `  _pad${i}: f32,`).join('\n')}\n};\n`;
  }
  set(obj) {
    for (const k in obj) {
      const i = this.index[k];
      if (i === undefined) throw new Error(`unknown uniform ${this.name}.${k}`);
      this.data[i] = obj[k];
    }
    return this;
  }
  upload() { this.device.queue.writeBuffer(this.buffer, 0, this.data); return this; }
}

// Derives an explicit bind group layout from the WGSL declarations of group 0, so bindings that the
// compiler would strip from an 'auto' layout (unused in a variant) stay valid.
export function layoutFromWGSL(device, code, visibility = GPUShaderStage.COMPUTE) {
  const re = /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<([^>]+)>)?\s+\w+\s*:\s*([^;]+);/g;
  const entries = [];
  let m;
  while ((m = re.exec(code))) {
    const binding = Number(m[1]);
    const space = (m[2] || '').replace(/\s/g, '');
    const type = m[3].trim();
    const e = { binding, visibility };
    if (space.startsWith('uniform')) e.buffer = { type: 'uniform' };
    else if (space.startsWith('storage')) e.buffer = { type: space.includes('read_write') ? 'storage' : 'read-only-storage' };
    else if (type.startsWith('sampler')) e.sampler = { type: 'filtering' };
    else if (type.startsWith('texture_storage_')) {
      const [, dim, fmt, acc] = type.match(/texture_storage_(\w+)<\s*(\w+)\s*,\s*(\w+)\s*>/);
      e.storageTexture = { format: fmt, access: acc === 'read_write' ? 'read-write' : acc === 'read' ? 'read-only' : 'write-only', viewDimension: dim };
    } else if (type.startsWith('texture_')) {
      const dim = type.match(/texture_(\w+?)</)[1];
      e.texture = { sampleType: 'float', viewDimension: dim };
    } else continue;
    entries.push(e);
  }
  return device.createBindGroupLayout({ entries });
}

// Pipeline/bind-group convenience. Resources are given in binding order.
export function computePipeline(device, code, entryPoint = 'main', label = entryPoint) {
  const module = device.createShaderModule({ label, code });
  module.getCompilationInfo?.().then(info => {
    for (const m of info.messages) if (m.type === 'error') console.error(`[${label}] ${m.lineNum}:${m.linePos} ${m.message}`);
  });
  const bgl = layoutFromWGSL(device, code);
  const pipe = device.createComputePipeline({ label, layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint } });
  pipe.bgl = bgl;
  return pipe;
}

export function bindGroup(device, pipeline, resources, group = 0, label = '') {
  const entries = resources.map((r, i) => {
    let resource = r;
    if (r instanceof GPUBuffer) resource = { buffer: r };
    else if (r instanceof GPUTexture) resource = r.createView();
    return { binding: i, resource };
  });
  return device.createBindGroup({ label, layout: pipeline.bgl || pipeline.getBindGroupLayout(group), entries });
}

export function dispatch3(pass, pipeline, bg, [x, y, z], [wx, wy, wz] = [8, 8, 4]) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(x / wx), Math.ceil(y / wy), Math.ceil(z / wz));
}

// GPU timestamp profiler (optional).
export class Profiler {
  constructor(device, enabled, max = 64) {
    this.enabled = enabled && device.features.has('timestamp-query');
    this.device = device;
    if (!this.enabled) return;
    this.max = max;
    this.qs = device.createQuerySet({ type: 'timestamp', count: max * 2 });
    this.resolve = device.createBuffer({ size: max * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.read = device.createBuffer({ size: max * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.labels = [];
    this.busy = false;
    this.results = {};
  }
  begin() { if (this.enabled) this.labels = []; }
  pass(label) {
    if (!this.enabled || this.busy || this.labels.length >= this.max) return undefined;
    const i = this.labels.length;
    this.labels.push(label);
    return { querySet: this.qs, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 };
  }
  end(enc) {
    if (!this.enabled || this.busy || !this.labels.length) return;
    const n = this.labels.length * 2;
    enc.resolveQuerySet(this.qs, 0, n, this.resolve, 0);
    enc.copyBufferToBuffer(this.resolve, 0, this.read, 0, n * 8);
    this.pending = this.labels.slice();
  }
  async collect() {
    if (!this.enabled || this.busy || !this.pending) return;
    this.busy = true;
    const labels = this.pending; this.pending = null;
    await this.read.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(this.read.getMappedRange());
    const agg = {};
    for (let i = 0; i < labels.length; i++) {
      const ms = Number(t[i * 2 + 1] - t[i * 2]) / 1e6;
      const k = labels[i];
      agg[k] = (agg[k] || 0) + ms;
    }
    this.read.unmap();
    for (const k in agg) this.results[k] = this.results[k] ? this.results[k] * 0.9 + agg[k] * 0.1 : agg[k];
    this.busy = false;
  }
}
