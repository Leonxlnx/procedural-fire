// GPU micro-benchmark: sustained 3D texture sampling throughput.
import puppeteer from 'puppeteer-core';
import { createServer } from './server.mjs';
const server = await createServer(8798);
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', protocolTimeout: 120000,
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--no-sandbox'],
});
const page = await browser.newPage();
await page.goto('http://localhost:8798/package.json');
const r = await page.evaluate(async () => {
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const dev = await a.requestDevice();
  const out = {};
  const N = 128, G = [80, 160, 80];
  const noise = dev.createTexture({ size: [N, N, N], dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const data = new Uint16Array(N * N * N * 4);
  for (let i = 0; i < data.length; i++) data[i] = 0x3800 + ((Math.random() * 1000) | 0); // ~0.5..1 halfs
  dev.queue.writeTexture({ texture: noise }, data, { bytesPerRow: N * 8, rowsPerImage: N }, [N, N, N]);
  const dst = dev.createTexture({ size: G, dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING });
  const smp = dev.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' });
  async function test(name, body, wg = '8,8,4', disp = [G[0] / 8, G[1] / 8, G[2] / 4]) {
    const code = `
      @group(0) @binding(0) var src: texture_3d<f32>;
      @group(0) @binding(1) var smp: sampler;
      @group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
      @compute @workgroup_size(${wg}) fn main(@builtin(global_invocation_id) id: vec3u) {
        let p = vec3f(id) * 0.01;
        var acc = vec4f(0);
        ${body}
        acc += textureSampleLevel(src, smp, vec3f(0.5), 0.0) * 1e-9;
        textureStore(dst, id, acc);
      }`;
    const mod = dev.createShaderModule({ code });
    const pipe = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: noise.createView() }, { binding: 1, resource: smp }, { binding: 2, resource: dst.createView() }] });
    const run = async (n) => {
      const enc = dev.createCommandEncoder();
      const p = enc.beginComputePass();
      for (let i = 0; i < n; i++) { p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(...disp); }
      p.end();
      dev.queue.submit([enc.finish()]);
      await dev.queue.onSubmittedWorkDone();
    };
    await run(20);
    const t0 = performance.now(); await run(100); out[name] = +((performance.now() - t0) / 100).toFixed(3);
  }
  await test('1 sample', 'acc += textureSampleLevel(src, smp, p, 0.0);');
  await test('12 samples', 'for (var i = 0; i < 12; i++) { acc += textureSampleLevel(src, smp, p + f32(i) * 0.013, 0.0); }');
  await test('12 manual trilinear', `for (var i = 0; i < 12; i++) {
      let g = (p + f32(i) * 0.013) * 128.0 - 0.5; let g0 = vec3i(floor(g)); let f = g - floor(g);
      var a = vec4f(0);
      for (var k = 0; k < 8; k++) { let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1); let w = mix(1.0 - f, f, vec3f(o));
        a += textureLoad(src, (g0 + o) & vec3i(127), 0) * (w.x * w.y * w.z); }
      acc += a; }`);
  await test('12 samples nearest-ish coords', 'for (var i = 0; i < 12; i++) { acc += textureSampleLevel(src, smp, (floor(p * 128.0) + 0.5) / 128.0 + f32(i) * 0.0078125, 0.0); }');
  await test('12 loads strided', 'for (var i = 0; i < 12; i++) { acc += textureLoad(src, (vec3i(vec3f(id) * 1.28) + vec3i(i * 3, i * 5, i * 7)) & vec3i(127), 0); }');
  return out;
});
console.log(JSON.stringify(r, null, 1));
await browser.close(); server.close();
