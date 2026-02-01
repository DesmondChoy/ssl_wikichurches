/**
 * Client-side heatmap rendering using Canvas.
 * Renders similarity values as a smooth Turbo colormap overlay.
 * Uses small canvas + browser bilinear interpolation for DINO-style smooth heatmaps.
 */

// Turbo colormap (256 colors, R, G, B values 0-255)
// Created by Google Research as a perceptually improved rainbow colormap
// Provides blue → cyan → green → yellow → orange color progression
const TURBO: [number, number, number][] = [
  [48, 18, 59], [50, 21, 67], [51, 24, 74], [52, 27, 81], [53, 30, 88],
  [54, 33, 95], [55, 36, 102], [56, 39, 109], [57, 42, 115], [58, 45, 121],
  [59, 47, 128], [60, 50, 134], [61, 53, 139], [62, 56, 145], [63, 59, 151],
  [63, 62, 156], [64, 64, 162], [65, 67, 167], [65, 70, 172], [66, 73, 177],
  [66, 75, 181], [67, 78, 186], [68, 81, 191], [68, 84, 195], [68, 86, 199],
  [69, 89, 203], [69, 92, 207], [69, 94, 211], [70, 97, 214], [70, 100, 218],
  [70, 102, 221], [70, 105, 224], [70, 107, 227], [71, 110, 230], [71, 113, 233],
  [71, 115, 235], [71, 118, 238], [71, 120, 240], [71, 123, 242], [70, 125, 244],
  [70, 128, 246], [70, 130, 248], [70, 133, 250], [70, 135, 251], [69, 138, 252],
  [69, 140, 253], [68, 143, 254], [67, 145, 254], [66, 148, 255], [65, 150, 255],
  [64, 153, 255], [62, 155, 254], [61, 158, 254], [59, 160, 253], [58, 163, 252],
  [56, 165, 251], [55, 168, 250], [53, 171, 248], [51, 173, 247], [49, 175, 245],
  [47, 178, 244], [46, 180, 242], [44, 183, 240], [42, 185, 238], [40, 188, 235],
  [39, 190, 233], [37, 192, 231], [35, 195, 228], [34, 197, 226], [32, 199, 223],
  [31, 201, 221], [30, 203, 218], [28, 205, 216], [27, 208, 213], [26, 210, 210],
  [26, 212, 208], [25, 213, 205], [24, 215, 202], [24, 217, 200], [24, 219, 197],
  [24, 221, 194], [24, 222, 192], [24, 224, 189], [25, 226, 187], [25, 227, 185],
  [26, 228, 182], [28, 230, 180], [29, 231, 178], [31, 233, 175], [32, 234, 172],
  [34, 235, 170], [37, 236, 167], [39, 238, 164], [42, 239, 161], [44, 240, 158],
  [47, 241, 155], [50, 242, 152], [53, 243, 148], [56, 244, 145], [60, 245, 142],
  [63, 246, 138], [67, 247, 135], [70, 248, 132], [74, 248, 128], [78, 249, 125],
  [82, 250, 122], [85, 250, 118], [89, 251, 115], [93, 252, 111], [97, 252, 108],
  [101, 253, 105], [105, 253, 102], [109, 254, 98], [113, 254, 95], [117, 254, 92],
  [121, 254, 89], [125, 255, 86], [128, 255, 83], [132, 255, 81], [136, 255, 78],
  [139, 255, 75], [143, 255, 73], [146, 255, 71], [150, 254, 68], [153, 254, 66],
  [156, 254, 64], [159, 253, 63], [161, 253, 61], [164, 252, 60], [167, 252, 58],
  [169, 251, 57], [172, 251, 56], [175, 250, 55], [177, 249, 54], [180, 248, 54],
  [183, 247, 53], [185, 246, 53], [188, 245, 52], [190, 244, 52], [193, 243, 52],
  [195, 241, 52], [198, 240, 52], [200, 239, 52], [203, 237, 52], [205, 236, 52],
  [208, 234, 52], [210, 233, 53], [212, 231, 53], [215, 229, 53], [217, 228, 54],
  [219, 226, 54], [221, 224, 55], [223, 223, 55], [225, 221, 55], [227, 219, 56],
  [229, 217, 56], [231, 215, 57], [233, 213, 57], [235, 211, 57], [236, 209, 58],
  [238, 207, 58], [239, 205, 58], [241, 203, 58], [242, 201, 58], [244, 199, 58],
  [245, 197, 58], [246, 195, 58], [247, 193, 58], [248, 190, 57], [249, 188, 57],
  [250, 186, 57], [251, 184, 56], [251, 182, 55], [252, 179, 54], [252, 177, 54],
  [253, 174, 53], [253, 172, 52], [254, 169, 51], [254, 167, 50], [254, 164, 49],
  [254, 161, 48], [254, 158, 47], [254, 155, 45], [254, 153, 44], [254, 150, 43],
  [254, 147, 42], [254, 144, 41], [253, 141, 39], [253, 138, 38], [252, 135, 37],
  [252, 132, 35], [251, 129, 34], [251, 126, 33], [250, 123, 31], [249, 120, 30],
  [249, 117, 29], [248, 114, 28], [247, 111, 26], [246, 108, 25], [245, 105, 24],
  [244, 102, 23], [243, 99, 21], [242, 96, 20], [241, 93, 19], [240, 91, 18],
  [239, 88, 17], [237, 85, 16], [236, 83, 15], [235, 80, 14], [234, 78, 13],
  [232, 75, 12], [231, 73, 12], [229, 71, 11], [228, 69, 10], [226, 67, 10],
  [225, 65, 9], [223, 63, 8], [221, 61, 8], [220, 59, 7], [218, 57, 7],
  [216, 55, 6], [214, 53, 6], [212, 51, 5], [210, 49, 5], [208, 47, 5],
  [206, 45, 4], [204, 43, 4], [202, 42, 4], [200, 40, 3], [197, 38, 3],
  [195, 37, 3], [193, 35, 2], [190, 33, 2], [188, 32, 2], [185, 30, 2],
  [183, 29, 2], [180, 27, 1], [178, 26, 1], [175, 24, 1], [172, 23, 1],
  [169, 22, 1], [167, 20, 1], [164, 19, 1], [161, 18, 1], [158, 16, 1],
  [155, 15, 1], [152, 14, 1], [149, 13, 1], [146, 11, 1], [142, 10, 1],
  [139, 9, 2], [136, 8, 2], [133, 7, 2], [129, 6, 2], [126, 5, 2],
  [122, 4, 3],
];

/**
 * Interpolate a value (0-1) to a Turbo color.
 */
function turboColor(value: number): [number, number, number] {
  const clampedValue = Math.max(0, Math.min(1, value));
  const index = Math.floor(clampedValue * (TURBO.length - 1));
  return TURBO[index];
}

export interface RenderHeatmapOptions {
  similarity: number[];
  patchGrid: [number, number];
  width?: number;
  height?: number;
  opacity?: number;
  minValue?: number;
  maxValue?: number;
}

/**
 * Render a smooth similarity heatmap to a data URL.
 * Uses small canvas + browser bilinear interpolation for DINO-style smooth gradients.
 *
 * @param options - Rendering options
 * @returns Data URL of the rendered heatmap image
 */
export function renderHeatmap(options: RenderHeatmapOptions): string {
  const {
    similarity,
    patchGrid,
    width = 224,
    height = 224,
    opacity = 0.7,
    minValue,
    maxValue,
  } = options;

  const [gridRows, gridCols] = patchGrid;

  // Normalize similarity values
  const min = minValue ?? Math.min(...similarity);
  const max = maxValue ?? Math.max(...similarity);
  const range = max - min || 1;

  // Step 1: Create small canvas at patch grid resolution
  const smallCanvas = document.createElement('canvas');
  smallCanvas.width = gridCols;
  smallCanvas.height = gridRows;
  const smallCtx = smallCanvas.getContext('2d');

  if (!smallCtx) {
    throw new Error('Could not get small canvas context');
  }

  // Step 2: Fill each pixel with color + opacity
  const imageData = smallCtx.createImageData(gridCols, gridRows);
  for (let i = 0; i < similarity.length; i++) {
    const normalizedValue = (similarity[i] - min) / range;
    const [r, g, b] = turboColor(normalizedValue);
    const idx = i * 4;
    imageData.data[idx] = r;
    imageData.data[idx + 1] = g;
    imageData.data[idx + 2] = b;
    imageData.data[idx + 3] = Math.round(opacity * 255);
  }
  smallCtx.putImageData(imageData, 0, 0);

  // Step 3: Scale up with bilinear interpolation
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Could not get canvas context');
  }

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(smallCanvas, 0, 0, width, height);

  return canvas.toDataURL('image/png');
}

/**
 * Render a color legend for the heatmap using Turbo colormap.
 *
 * @param width - Legend width
 * @param height - Legend height
 * @returns Data URL of the legend image
 */
export function renderHeatmapLegend(
  width = 200,
  height = 20
): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Could not get canvas context');
  }

  // Draw gradient using Turbo colormap
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    const [r, g, b] = turboColor(t);
    gradient.addColorStop(t, `rgb(${r}, ${g}, ${b})`);
  }

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  return canvas.toDataURL('image/png');
}

export interface SimilarityStats {
  min: number;
  max: number;
  mean: number;
  median: number;
}

/**
 * Compute statistics for similarity values.
 */
export function computeSimilarityStats(similarity: number[]): SimilarityStats {
  if (similarity.length === 0) {
    return { min: 0, max: 0, mean: 0, median: 0 };
  }

  const sorted = [...similarity].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const mean = similarity.reduce((a, b) => a + b, 0) / similarity.length;
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];

  return { min, max, mean, median };
}
