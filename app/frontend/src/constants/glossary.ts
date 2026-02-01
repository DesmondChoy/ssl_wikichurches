/**
 * Glossary of technical terms for educational tooltips.
 * Keys should match labels used in the UI for easy lookup.
 */

export const GLOSSARY: Record<string, string> = {
  'Attention Method':
    'How attention is computed. CLS uses class token attention. Rollout accumulates attention across layers.',
  'Attention Threshold':
    'Filters to show only top-attended regions. "Top 10%" shows patches in the highest 10% of attention values.',
  'Similarity Heatmap':
    'Shows how similar each region is to the selected bounding box based on learned features.',
  'Heatmap Opacity': 'Controls transparency of the similarity overlay.',
  'Heatmap Style':
    'Visual style: Smooth uses interpolation, Squares/Circles show discrete patch values.',
  Layer:
    'Network depth. Early layers capture edges/textures. Later layers capture semantic concepts. Layer count varies by model.',
  Model: 'Vision model for feature extraction. Each has different architecture and training.',
  'Show Bounding Boxes': 'Toggle visibility of annotated bounding boxes on the image.',
};
