// Simple look-at camera.
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

export function lookAt(pos, target, fovYdeg = 40, roll = 0) {
  const fwd = norm(sub(target, pos));
  let right = norm(cross(fwd, [0, 1, 0]));
  let up = cross(right, fwd);
  if (roll) {
    const c = Math.cos(roll), s = Math.sin(roll);
    const r2 = right.map((v, i) => v * c + up[i] * s);
    const u2 = up.map((v, i) => v * c - right[i] * s);
    right = r2; up = u2;
  }
  return { pos, target, fwd, right, up, fovY: (fovYdeg * Math.PI) / 180 };
}

export function orbit({ yaw = 0, pitch = 0.15, dist = 2.4, target = [0, 0.45, 0], fov = 40 }) {
  const pos = [
    target[0] + dist * Math.cos(pitch) * Math.sin(yaw),
    target[1] + dist * Math.sin(pitch),
    target[2] + dist * Math.cos(pitch) * Math.cos(yaw),
  ];
  return lookAt(pos, target, fov);
}
