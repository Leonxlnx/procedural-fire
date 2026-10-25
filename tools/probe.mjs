import puppeteer from 'puppeteer-core';
import { createServer } from './server.mjs';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const headless = process.argv[2] !== 'headful';
const server = await createServer(8799);
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: headless ? 'new' : false,
  protocolTimeout: 60000,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--no-sandbox'],
});
const page = await browser.newPage();
await page.goto('http://localhost:8799/package.json');
const info = await page.evaluate(async () => {
  if (!navigator.gpu) return { err: 'no navigator.gpu', secure: isSecureContext };
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!a) return { err: 'no adapter' };
  const feats = [...a.features];
  const lim = {};
  for (const k of ['maxTextureDimension3D','maxStorageTexturesPerShaderStage','maxStorageBuffersPerShaderStage','maxComputeWorkgroupSizeX','maxComputeInvocationsPerWorkgroup','maxBufferSize','maxStorageBufferBindingSize','maxComputeWorkgroupStorageSize','maxBindGroups','maxSampledTexturesPerShaderStage']) lim[k] = a.limits[k];
  const wgslFeats = navigator.gpu.wgslLanguageFeatures ? [...navigator.gpu.wgslLanguageFeatures] : [];
  // quick compute benchmark: 128^3 rgba16float texture write
  const dev = await a.requestDevice({ requiredFeatures: feats.filter(f => ['timestamp-query', 'float32-filterable', 'shader-f16'].includes(f)) });
  const N = 128;
  const tex = dev.createTexture({ size: [N, N, N], dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
  const tex2 = dev.createTexture({ size: [N, N, N], dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
  const mod = dev.createShaderModule({ code: `
    @group(0) @binding(0) var src: texture_3d<f32>;
    @group(0) @binding(1) var smp: sampler;
    @group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
    @compute @workgroup_size(8,8,4) fn main(@builtin(global_invocation_id) id: vec3u) {
      let p = (vec3f(id) + 0.5) / ${N}.0;
      var acc = vec4f(0);
      for (var i = 0; i < 4; i++) { acc += textureSampleLevel(src, smp, p + vec3f(f32(i) * 0.01), 0.0); }
      textureStore(dst, id, acc * 0.25 + vec4f(p, 1.0) * 0.001);
    }` });
  const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  const smp = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: tex.createView() }, { binding: 1, resource: smp }, { binding: 2, resource: tex2.createView() }] });
  const run = async (n) => {
    const enc = dev.createCommandEncoder();
    for (let i = 0; i < n; i++) { const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(N/8, N/8, N/4); p.end(); }
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
  };
  await run(5);
  const t0 = performance.now(); await run(100); const t1 = performance.now();
  return { info: a.info ? { vendor: a.info.vendor, arch: a.info.architecture, desc: a.info.description, isFallback: a.info.isFallbackAdapter } : null, feats, lim, wgslFeats, pref: navigator.gpu.getPreferredCanvasFormat(), msPerPass128: (t1 - t0) / 100 };
});
console.log(JSON.stringify(info, null, 1));
await browser.close();
server.close();
