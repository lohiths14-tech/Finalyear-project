import type { GridCell } from '../types';

/**
 * Generates a smooth Gaussian thermal heatmap PNG data URL from real model
 * grid patch severities. Used as a client-side fallback if backend
 * heatmap_base64 is unavailable or fails to decode.
 */
export function buildThermalFallbackDataUrl(
  gridData: GridCell[] | undefined,
  width = 320,
  height = 240
): string | null {
  if (!gridData || gridData.length === 0 || typeof document === 'undefined') {
    return null;
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  // Precompute patch centers and radii in pixel coordinates
  const patches = gridData.map((cell) => {
    const cx = ((cell.x_percent + cell.w_percent * 0.5) / 100) * width;
    const cy = ((cell.y_percent + cell.h_percent * 0.5) / 100) * height;
    const rx = Math.max(12, ((cell.w_percent / 100) * width) * 1.15);
    const ry = Math.max(12, ((cell.h_percent / 100) * height) * 1.15);
    return {
      cx,
      cy,
      invTwoSigmaSqX: 1 / (2 * rx * rx),
      invTwoSigmaSqY: 1 / (2 * ry * ry),
      severity: Math.min(100, Math.max(0, cell.severity || 0)),
    };
  });

  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      let weightSum = 0;
      let sevSum = 0;
      let maxWeight = 0;

      for (let i = 0; i < patches.length; i++) {
        const p = patches[i];
        const dx = px - p.cx;
        const dy = py - p.cy;
        const d2 = dx * dx * p.invTwoSigmaSqX + dy * dy * p.invTwoSigmaSqY;
        if (d2 > 4.5) continue;
        const w = Math.exp(-d2);
        weightSum += w;
        sevSum += w * p.severity;
        if (w > maxWeight) maxWeight = w;
      }

      const idx = (py * width + px) * 4;
      if (weightSum < 0.08) {
        data[idx + 3] = 0;
        continue;
      }

      const sev = sevSum / weightSum;
      // Color ramp: 0-20% emerald/green -> 20-50% amber/yellow -> 50-100% orange/red
      let r: number;
      let g: number;
      let b: number;
      if (sev < 25) {
        const t = sev / 25;
        r = Math.round(16 + t * 110);
        g = Math.round(185 + t * 25);
        b = Math.round(129 - t * 80);
      } else if (sev < 55) {
        const t = (sev - 25) / 30;
        r = Math.round(126 + t * 124);
        g = Math.round(210 - t * 45);
        b = Math.round(49 - t * 35);
      } else {
        const t = Math.min(1, (sev - 55) / 45);
        r = Math.round(250 - t * 12);
        g = Math.round(165 - t * 125);
        b = Math.round(14 + t * 20);
      }

      const density = Math.min(1, maxWeight * 1.35);
      const alpha = Math.round((70 + (sev / 100) * 145) * density);

      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = alpha;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}
