/**
 * Client-side heatmap rendering using Canvas.
 * Renders similarity values as a viridis colormap overlay.
 */

// Viridis colormap (256 colors, R, G, B values 0-255)
// This is a perceptually uniform colormap that works well for scientific visualization
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [68, 2, 86], [69, 4, 87], [69, 5, 89], [70, 7, 90],
  [70, 8, 92], [70, 10, 93], [70, 11, 94], [71, 13, 96], [71, 14, 97],
  [71, 16, 99], [71, 17, 100], [71, 19, 101], [72, 20, 103], [72, 22, 104],
  [72, 23, 105], [72, 24, 106], [72, 26, 108], [72, 27, 109], [72, 28, 110],
  [72, 29, 111], [72, 31, 112], [72, 32, 113], [72, 33, 115], [72, 35, 116],
  [72, 36, 117], [72, 37, 118], [72, 38, 119], [72, 40, 120], [72, 41, 121],
  [71, 42, 122], [71, 44, 122], [71, 45, 123], [71, 46, 124], [71, 47, 125],
  [70, 48, 126], [70, 50, 126], [70, 51, 127], [69, 52, 128], [69, 53, 129],
  [69, 55, 129], [68, 56, 130], [68, 57, 131], [68, 58, 131], [67, 60, 132],
  [67, 61, 132], [66, 62, 133], [66, 63, 133], [66, 64, 134], [65, 66, 134],
  [65, 67, 135], [64, 68, 135], [64, 69, 136], [63, 71, 136], [63, 72, 137],
  [62, 73, 137], [62, 74, 137], [62, 76, 138], [61, 77, 138], [61, 78, 138],
  [60, 79, 139], [60, 80, 139], [59, 82, 139], [59, 83, 140], [58, 84, 140],
  [58, 85, 140], [57, 86, 141], [57, 88, 141], [56, 89, 141], [56, 90, 141],
  [55, 91, 142], [55, 92, 142], [54, 94, 142], [54, 95, 142], [53, 96, 142],
  [53, 97, 143], [52, 98, 143], [52, 100, 143], [51, 101, 143], [51, 102, 143],
  [50, 103, 143], [50, 104, 144], [49, 106, 144], [49, 107, 144], [49, 108, 144],
  [48, 109, 144], [48, 110, 144], [47, 111, 144], [47, 113, 144], [46, 114, 144],
  [46, 115, 144], [45, 116, 144], [45, 117, 144], [45, 118, 144], [44, 120, 144],
  [44, 121, 144], [43, 122, 144], [43, 123, 144], [43, 124, 144], [42, 125, 144],
  [42, 126, 144], [42, 128, 144], [41, 129, 144], [41, 130, 143], [41, 131, 143],
  [40, 132, 143], [40, 133, 143], [40, 134, 143], [39, 136, 142], [39, 137, 142],
  [39, 138, 142], [38, 139, 141], [38, 140, 141], [38, 141, 141], [37, 142, 140],
  [37, 144, 140], [37, 145, 139], [37, 146, 139], [36, 147, 138], [36, 148, 138],
  [36, 149, 137], [36, 150, 137], [36, 151, 136], [35, 152, 136], [35, 154, 135],
  [35, 155, 134], [35, 156, 134], [35, 157, 133], [35, 158, 132], [35, 159, 132],
  [35, 160, 131], [35, 161, 130], [35, 162, 130], [35, 163, 129], [35, 165, 128],
  [35, 166, 127], [35, 167, 127], [35, 168, 126], [36, 169, 125], [36, 170, 124],
  [36, 171, 123], [37, 172, 123], [37, 173, 122], [37, 174, 121], [38, 175, 120],
  [38, 176, 119], [39, 177, 118], [39, 178, 117], [40, 179, 116], [41, 180, 115],
  [41, 181, 114], [42, 182, 113], [43, 183, 112], [44, 184, 111], [44, 185, 110],
  [45, 186, 109], [46, 187, 108], [47, 188, 107], [48, 189, 105], [49, 190, 104],
  [50, 190, 103], [51, 191, 102], [52, 192, 101], [54, 193, 99], [55, 194, 98],
  [56, 195, 97], [58, 195, 96], [59, 196, 94], [60, 197, 93], [62, 198, 92],
  [63, 198, 90], [65, 199, 89], [66, 200, 88], [68, 200, 86], [70, 201, 85],
  [71, 202, 83], [73, 202, 82], [75, 203, 80], [77, 204, 79], [78, 204, 77],
  [80, 205, 76], [82, 205, 74], [84, 206, 73], [86, 206, 71], [88, 207, 69],
  [90, 207, 68], [92, 208, 66], [94, 208, 64], [96, 209, 63], [98, 209, 61],
  [100, 210, 59], [102, 210, 58], [104, 210, 56], [106, 211, 54], [108, 211, 52],
  [111, 211, 51], [113, 212, 49], [115, 212, 47], [117, 212, 46], [119, 213, 44],
  [122, 213, 42], [124, 213, 40], [126, 213, 39], [128, 214, 37], [131, 214, 35],
  [133, 214, 33], [135, 214, 32], [138, 214, 30], [140, 215, 28], [142, 215, 27],
  [145, 215, 25], [147, 215, 24], [149, 215, 22], [152, 215, 21], [154, 215, 19],
  [156, 215, 18], [159, 215, 17], [161, 215, 16], [163, 215, 15], [166, 215, 14],
  [168, 215, 13], [170, 215, 13], [172, 215, 12], [175, 215, 12], [177, 214, 12],
  [179, 214, 13], [181, 214, 13], [184, 214, 14], [186, 213, 15], [188, 213, 16],
  [190, 212, 18], [192, 212, 19], [194, 211, 21], [196, 211, 23], [198, 210, 25],
  [200, 209, 27], [202, 209, 29], [204, 208, 31], [206, 207, 34], [208, 206, 36],
  [209, 205, 39], [211, 205, 41], [213, 204, 44], [214, 203, 47], [216, 202, 50],
  [217, 201, 53], [219, 200, 56], [220, 198, 59], [221, 197, 62], [223, 196, 66],
  [224, 195, 69], [225, 194, 72], [226, 192, 76], [227, 191, 79], [228, 190, 83],
  [229, 188, 86], [230, 187, 90], [231, 186, 93], [232, 184, 97], [232, 183, 101],
  [233, 181, 104], [234, 180, 108], [234, 178, 111], [235, 177, 115], [235, 175, 119],
  [236, 173, 122], [236, 172, 126], [236, 170, 130], [237, 168, 133], [237, 167, 137],
  [237, 165, 141], [237, 163, 144], [238, 161, 148], [238, 160, 152], [238, 158, 155],
  [238, 156, 159], [238, 154, 162], [238, 152, 166], [238, 150, 170], [238, 148, 173],
  [238, 147, 177], [238, 145, 180], [237, 143, 184], [237, 141, 187], [237, 139, 191],
  [237, 137, 194], [236, 135, 198], [236, 133, 201], [235, 131, 205], [235, 129, 208],
  [234, 127, 211], [234, 125, 215], [233, 123, 218], [232, 121, 221], [231, 119, 224],
  [231, 117, 227], [230, 116, 230], [229, 114, 233], [228, 112, 235], [227, 110, 238],
  [225, 108, 240], [224, 106, 242], [223, 105, 244], [221, 103, 246], [220, 101, 248],
  [218, 100, 249], [216, 98, 250], [215, 97, 252], [213, 95, 253], [211, 94, 254],
  [209, 92, 254], [207, 91, 255], [205, 90, 255], [203, 89, 255], [201, 88, 255],
];

/**
 * Interpolate a value (0-1) to a viridis color.
 */
function viridisColor(value: number): [number, number, number] {
  const clampedValue = Math.max(0, Math.min(1, value));
  const index = Math.floor(clampedValue * (VIRIDIS.length - 1));
  return VIRIDIS[index];
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
 * Render a similarity heatmap to a data URL.
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
  const cellWidth = width / gridCols;
  const cellHeight = height / gridRows;

  // Normalize similarity values
  const min = minValue ?? Math.min(...similarity);
  const max = maxValue ?? Math.max(...similarity);
  const range = max - min || 1;

  // Create canvas
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Could not get canvas context');
  }

  // Clear canvas
  ctx.clearRect(0, 0, width, height);

  // Draw each patch as a colored rectangle
  for (let i = 0; i < similarity.length; i++) {
    const row = Math.floor(i / gridCols);
    const col = i % gridCols;

    const normalizedValue = (similarity[i] - min) / range;
    const [r, g, b] = viridisColor(normalizedValue);

    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${opacity})`;
    ctx.fillRect(
      col * cellWidth,
      row * cellHeight,
      cellWidth + 0.5, // Small overlap to avoid gaps
      cellHeight + 0.5
    );
  }

  return canvas.toDataURL('image/png');
}

/**
 * Render a color legend for the heatmap.
 *
 * @param width - Legend width
 * @param height - Legend height
 * @param _minLabel - Label for minimum value (reserved for future use)
 * @param _maxLabel - Label for maximum value (reserved for future use)
 * @returns Data URL of the legend image
 */
export function renderHeatmapLegend(
  width = 200,
  height = 20,
  _minLabel = '0',
  _maxLabel = '1'
): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Could not get canvas context');
  }

  // Draw gradient
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    const [r, g, b] = viridisColor(t);
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
