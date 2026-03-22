/**
 * Generate 13-slide mid-project progress presentation for SSL WikiChurches.
 *
 * Uses PptxGenJS to create the deck from scratch.
 * Images are pre-generated in outputs/slides/.
 *
 * Run: node experiments/scripts/create_presentation.js
 */

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
const ROOT = path.resolve(__dirname, "../..");
const SLIDES_DIR = path.join(ROOT, "outputs", "slides");
const FIGURES_DIR = path.join(ROOT, "outputs", "figures");
const OUT_FILE = path.join(SLIDES_DIR, "presentation.pptx");

function imgPath(name) {
  const p = path.join(SLIDES_DIR, name);
  if (!fs.existsSync(p)) {
    console.warn(`WARNING: Image not found: ${p}`);
  }
  return p;
}

function figPath(name) {
  const p = path.join(FIGURES_DIR, name);
  if (!fs.existsSync(p)) {
    console.warn(`WARNING: Figure not found: ${p}`);
  }
  return p;
}

// ---------------------------------------------------------------------------
// Design system
// ---------------------------------------------------------------------------
const C = {
  CHARCOAL: "2D3436",
  STEEL: "4E79A7",
  TEAL: "93B7BE",
  TERRA: "D4764E",
  LIGHT_BG: "F8F9FA",
  WHITE: "FFFFFF",
  BODY: "333333",
  MUTED: "64748B",
  SUCCESS: "3A7D44",
  FAIL: "C04E4E",
  WARM_GRAY: "8A817C",
  LIGHT_GRAY: "E8E8E8",
  ACCENT_BAR: "4E79A7",
};

const FONT = {
  TITLE: "Georgia",
  BODY: "Calibri",
  MONO: "Consolas",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function addAccentBar(slide) {
  slide.addShape("rect", {
    x: 0, y: 5.525, w: 10, h: 0.1,
    fill: { color: C.ACCENT_BAR },
  });
}

function addSlideNumber(slide, num) {
  slide.addText(String(num), {
    x: 9.2, y: 5.2, w: 0.6, h: 0.3,
    fontSize: 10, color: C.MUTED, fontFace: FONT.BODY,
    align: "right",
  });
}

function addSectionTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontSize: 28, fontFace: FONT.TITLE, color: C.STEEL, bold: true,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.5, y: 0.85, w: 9, h: 0.35,
      fontSize: 14, fontFace: FONT.BODY, color: C.MUTED,
    });
  }
}

// Factory for text options (avoid mutation pitfall)
function bodyOpts(overrides) {
  return Object.assign({
    fontSize: 14, fontFace: FONT.BODY, color: C.BODY,
    valign: "top", paraSpaceAfter: 6,
  }, overrides || {});
}

// ---------------------------------------------------------------------------
// SLIDE 1: Title
// ---------------------------------------------------------------------------
function slide1_title(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.CHARCOAL };

  // Hero image on right half
  slide.addImage({
    path: imgPath("slide01_title_hero.png"),
    x: 5.0, y: 0, w: 5.0, h: 5.625,
    sizing: { type: "cover", w: 5.0, h: 5.625 },
  });

  // Semi-transparent overlay on right for text readability
  slide.addShape("rect", {
    x: 5.0, y: 0, w: 5.0, h: 5.625,
    fill: { color: C.CHARCOAL, transparency: 40 },
  });

  // Title text
  slide.addText("Do Self-Supervised Vision\nModels Learn What\nExperts See?", {
    x: 0.6, y: 0.8, w: 4.6, h: 2.0,
    fontSize: 30, fontFace: FONT.TITLE, color: C.WHITE, bold: true,
    lineSpacingMultiple: 1.15,
  });

  // Subtitle
  slide.addText("Attention Alignment with Human-Annotated\nArchitectural Features", {
    x: 0.6, y: 2.9, w: 4.6, h: 0.8,
    fontSize: 16, fontFace: FONT.BODY, color: C.TEAL,
  });

  // Course info
  slide.addText("ISY5004 \u2014 Intelligent Sensing Systems Practice Module\nMid-Project Progress Update \u2014 March 2026", {
    x: 0.6, y: 4.2, w: 4.6, h: 0.7,
    fontSize: 12, fontFace: FONT.BODY, color: C.WARM_GRAY,
  });

  // Accent bar
  slide.addShape("rect", {
    x: 0, y: 5.4, w: 10, h: 0.225,
    fill: { color: C.STEEL },
  });
}

// ---------------------------------------------------------------------------
// SLIDE 2: Motivation & Research Gap
// ---------------------------------------------------------------------------
function slide2_motivation(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Why This Matters", "SSL models achieve strong benchmarks \u2014 but what are they looking at?");
  addAccentBar(slide);
  addSlideNumber(slide, 2);

  // Left: bullet points
  slide.addText([
    { text: "The Problem", options: { fontSize: 16, bold: true, color: C.STEEL, breakLine: true } },
    { text: "SSL models can classify \u201cGothic cathedral\u201d by attending to ", options: { fontSize: 13, breakLine: false } },
    { text: "overcast skies", options: { fontSize: 13, bold: true, color: C.FAIL, breakLine: false } },
    { text: " instead of ", options: { fontSize: 13, breakLine: false } },
    { text: "pointed arches", options: { fontSize: 13, bold: true, color: C.SUCCESS, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "The Gap", options: { fontSize: 16, bold: true, color: C.STEEL, breakLine: true } },
    { text: "\u2022 No cross-model benchmark for expert alignment", options: { fontSize: 13, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 No study on how fine-tuning shifts attention", options: { fontSize: 13, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 No strategy comparison (LP vs LoRA vs Full)", options: { fontSize: 13, breakLine: true } },
  ], {
    x: 0.5, y: 1.3, w: 4.2, h: 3.8,
    valign: "top", margin: 0,
  });

  // Right: good vs bad attention image
  slide.addImage({
    path: imgPath("slide02_good_vs_bad.png"),
    x: 4.9, y: 1.3, w: 4.8, h: 2.8,
    sizing: { type: "contain", w: 4.8, h: 2.8 },
  });

  // Caption
  slide.addText("Same image: DINOv3 focuses on architectural features (left),\nMAE attention is diffuse (right)", {
    x: 4.9, y: 4.2, w: 4.8, h: 0.6,
    fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, align: "center",
  });
}

// ---------------------------------------------------------------------------
// SLIDE 3: Research Questions
// ---------------------------------------------------------------------------
function slide3_rqs(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Three Research Questions");
  addAccentBar(slide);
  addSlideNumber(slide, 3);

  const cards = [
    { q: "Q1", title: "Frozen Model Alignment", desc: "Do SSL models attend to expert-identified diagnostic features?", method: "IoU between attention maps and 631 expert bounding boxes across 7 models \u00d7 12 layers", status: "Complete", statusColor: C.SUCCESS },
    { q: "Q2", title: "Fine-Tuning Effects", desc: "Does fine-tuning shift attention toward expert features? Does strategy matter?", method: "\u0394 IoU (fine-tuned \u2212 frozen) with paired Wilcoxon tests, Holm correction, Cohen\u2019s d", status: "Complete", statusColor: C.SUCCESS },
    { q: "Q3", title: "Per-Head Specialization", desc: "Do individual attention heads specialize for different architectural features?", method: "Per-head IoU \u00d7 feature-type matrix, rank-based head analysis", status: "Planned", statusColor: C.WARM_GRAY },
  ];

  cards.forEach((c, i) => {
    const y = 1.2 + i * 1.35;

    // Card background
    slide.addShape("roundedRectangle", {
      x: 0.5, y: y, w: 9.0, h: 1.2,
      fill: { color: C.WHITE },
      rectRadius: 0.08,
      shadow: { type: "outer", blur: 4, offset: 2, angle: 135, color: "CCCCCC", opacity: 0.3 },
    });

    // Status indicator bar (left edge)
    slide.addShape("rect", {
      x: 0.5, y: y, w: 0.08, h: 1.2,
      fill: { color: c.statusColor },
    });

    // Q label
    slide.addText(c.q, {
      x: 0.75, y: y + 0.15, w: 0.6, h: 0.5,
      fontSize: 22, fontFace: FONT.TITLE, color: C.STEEL, bold: true, margin: 0,
    });

    // Title
    slide.addText(c.title, {
      x: 1.4, y: y + 0.1, w: 5.0, h: 0.35,
      fontSize: 16, fontFace: FONT.BODY, color: C.BODY, bold: true, margin: 0,
    });

    // Description
    slide.addText(c.desc, {
      x: 1.4, y: y + 0.42, w: 5.5, h: 0.35,
      fontSize: 12, fontFace: FONT.BODY, color: C.BODY, margin: 0,
    });

    // Method
    slide.addText(c.method, {
      x: 1.4, y: y + 0.78, w: 5.5, h: 0.3,
      fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, margin: 0,
    });

    // Status badge
    slide.addShape("roundedRectangle", {
      x: 8.2, y: y + 0.35, w: 1.1, h: 0.4,
      fill: { color: c.statusColor },
      rectRadius: 0.05,
    });
    slide.addText(c.status, {
      x: 8.2, y: y + 0.35, w: 1.1, h: 0.4,
      fontSize: 11, fontFace: FONT.BODY, color: C.WHITE, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });
}

// ---------------------------------------------------------------------------
// SLIDE 4: Dataset
// ---------------------------------------------------------------------------
function slide4_dataset(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "WikiChurches Dataset", "Barz & Denzler, NeurIPS 2021");
  addAccentBar(slide);
  addSlideNumber(slide, 4);

  // Left: stats
  const statsRows = [
    [{ text: "Total images", options: { bold: true, color: C.BODY, fill: { color: C.LIGHT_GRAY } } },
     { text: "9,485", options: { align: "right", fill: { color: C.LIGHT_GRAY } } }],
    [{ text: "Annotated eval set", options: { bold: true, color: C.BODY } },
     { text: "139 images, 631 bboxes", options: { align: "right" } }],
    [{ text: "Styles", options: { bold: true, color: C.BODY, fill: { color: C.LIGHT_GRAY } } },
     { text: "Romanesque, Gothic, Renaissance, Baroque", options: { align: "right", fill: { color: C.LIGHT_GRAY } } }],
    [{ text: "Feature types", options: { bold: true, color: C.BODY } },
     { text: "106 categories", options: { align: "right" } }],
    [{ text: "Training (Q2)", options: { bold: true, color: C.BODY, fill: { color: C.LIGHT_GRAY } } },
     { text: "~4,588 labelled images", options: { align: "right", fill: { color: C.LIGHT_GRAY } } }],
    [{ text: "Eval holdout", options: { bold: true, color: C.BODY } },
     { text: "139 images (zero leakage)", options: { align: "right", color: C.SUCCESS, bold: true } }],
  ];

  slide.addTable(statsRows, {
    x: 0.5, y: 1.3, w: 4.2, h: 2.5,
    fontSize: 12, fontFace: FONT.BODY, color: C.BODY,
    border: { type: "none" },
    colW: [2.0, 2.2],
    margin: [4, 6, 4, 6],
  });

  // Style distribution
  slide.addText("Style Distribution", {
    x: 0.5, y: 3.9, w: 4.2, h: 0.3,
    fontSize: 12, fontFace: FONT.BODY, color: C.MUTED, bold: true,
  });

  const styles = [
    { name: "Romanesque", count: 54, color: C.STEEL },
    { name: "Gothic", count: 49, color: C.TEAL },
    { name: "Renaissance", count: 22, color: C.TERRA },
    { name: "Baroque", count: 17, color: C.WARM_GRAY },
  ];
  const maxCount = 54;
  styles.forEach((s, i) => {
    const barY = 4.25 + i * 0.28;
    const barW = (s.count / maxCount) * 2.5;
    slide.addText(s.name, {
      x: 0.5, y: barY, w: 1.2, h: 0.22,
      fontSize: 10, fontFace: FONT.BODY, color: C.BODY, align: "right", margin: 0,
    });
    slide.addShape("rect", {
      x: 1.8, y: barY + 0.02, w: barW, h: 0.18,
      fill: { color: s.color },
    });
    slide.addText(String(s.count), {
      x: 1.85 + barW, y: barY, w: 0.5, h: 0.22,
      fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, margin: 0,
    });
  });

  // Right: 2x2 grid image
  slide.addImage({
    path: imgPath("slide04_style_grid.png"),
    x: 5.0, y: 1.2, w: 4.6, h: 4.0,
    sizing: { type: "contain", w: 4.6, h: 4.0 },
  });
}

// ---------------------------------------------------------------------------
// SLIDE 5: Models
// ---------------------------------------------------------------------------
function slide5_models(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "7 Models, 4 SSL Paradigms + 1 Supervised Baseline", "All ViT-B architecture (12 layers, 768 dim, 12 heads, ~86M params) except ResNet-50");
  addAccentBar(slide);
  addSlideNumber(slide, 5);

  const headerOpts = { bold: true, color: C.WHITE, fill: { color: C.STEEL }, fontSize: 11, align: "center" };
  const rows = [
    [
      { text: "Model", options: Object.assign({}, headerOpts) },
      { text: "Paradigm", options: Object.assign({}, headerOpts) },
      { text: "Patch", options: Object.assign({}, headerOpts) },
      { text: "Method", options: Object.assign({}, headerOpts) },
      { text: "Key Feature", options: Object.assign({}, headerOpts) },
    ],
    makeModelRow("DINOv2", "Self-distillation", "14\u00d714", "CLS, Rollout", "4 register tokens", C.STEEL, false),
    makeModelRow("DINOv3", "Self-distillation + Gram", "16\u00d716", "CLS, Rollout", "RoPE encoding", C.STEEL, true),
    makeModelRow("MAE", "Masked Autoencoding", "16\u00d716", "CLS, Rollout", "Pixel reconstruction", C.SUCCESS, false),
    makeModelRow("CLIP", "Contrastive (softmax)", "16\u00d716", "CLS, Rollout", "Language-image align", C.TERRA, true),
    makeModelRow("SigLIP", "Contrastive (sigmoid)", "16\u00d716", "Mean", "No CLS token", C.TERRA, false),
    makeModelRow("SigLIP 2", "Contrastive (sigmoid)", "16\u00d716", "Mean", "Dense features", C.TERRA, true),
    makeModelRow("ResNet-50", "Supervised (ImageNet)", "\u2014", "Grad-CAM", "CNN baseline", C.WARM_GRAY, false),
  ];

  slide.addTable(rows, {
    x: 0.5, y: 1.3, w: 9.0, h: 3.8,
    fontSize: 12, fontFace: FONT.BODY, color: C.BODY,
    border: { type: "solid", color: C.LIGHT_GRAY, pt: 0.5 },
    colW: [1.2, 2.2, 0.8, 1.3, 1.8],
    margin: [4, 6, 4, 6],
    autoPage: false,
  });
}

function makeModelRow(name, paradigm, patch, method, feature, accentColor, altBg) {
  const bg = altBg ? { fill: { color: "F0F4F8" } } : {};
  return [
    { text: name, options: Object.assign({ bold: true, color: accentColor }, bg) },
    { text: paradigm, options: Object.assign({}, bg) },
    { text: patch, options: Object.assign({ align: "center" }, bg) },
    { text: method, options: Object.assign({}, bg) },
    { text: feature, options: Object.assign({ fontSize: 11, color: C.MUTED }, bg) },
  ];
}

// ---------------------------------------------------------------------------
// SLIDE 6: Methodology
// ---------------------------------------------------------------------------
function slide6_methodology(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Measuring Attention-Expert Alignment", "From attention map to alignment score");
  addAccentBar(slide);
  addSlideNumber(slide, 6);

  // Pipeline image
  slide.addImage({
    path: imgPath("slide06_pipeline.png"),
    x: 0.3, y: 1.2, w: 9.4, h: 2.2,
    sizing: { type: "contain", w: 9.4, h: 2.2 },
  });

  // Metrics table below
  slide.addText("5 Complementary Metrics", {
    x: 0.5, y: 3.5, w: 3, h: 0.3,
    fontSize: 14, fontFace: FONT.BODY, color: C.STEEL, bold: true,
  });

  const mHeaderOpts = { bold: true, color: C.WHITE, fill: { color: C.STEEL }, fontSize: 10, align: "center" };
  const mRows = [
    [
      { text: "Metric", options: Object.assign({}, mHeaderOpts) },
      { text: "Measures", options: Object.assign({}, mHeaderOpts) },
      { text: "Threshold?", options: Object.assign({}, mHeaderOpts) },
    ],
    [{ text: "IoU", options: { bold: true } }, { text: "Spatial overlap of top-k attention with expert regions" }, { text: "Yes", options: { align: "center" } }],
    [{ text: "Coverage", options: { bold: true } }, { text: "Fraction of attention energy inside expert regions" }, { text: "No", options: { align: "center" } }],
    [{ text: "Gaussian MSE", options: { bold: true } }, { text: "Distance from attention to Gaussian-blurred GT" }, { text: "No", options: { align: "center" } }],
    [{ text: "KL Divergence", options: { bold: true } }, { text: "Distribution divergence vs ground truth" }, { text: "No", options: { align: "center" } }],
    [{ text: "EMD", options: { bold: true } }, { text: "Optimal transport cost between distributions" }, { text: "No", options: { align: "center" } }],
  ];

  slide.addTable(mRows, {
    x: 0.5, y: 3.85, w: 5.5, h: 1.6,
    fontSize: 10, fontFace: FONT.BODY, color: C.BODY,
    border: { type: "solid", color: C.LIGHT_GRAY, pt: 0.5 },
    colW: [1.2, 3.2, 1.1],
    margin: [3, 4, 3, 4],
    autoPage: false,
  });

  // Statistical rigor box
  slide.addShape("roundedRectangle", {
    x: 6.3, y: 3.85, w: 3.3, h: 1.6,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "CCCCCC", opacity: 0.3 },
  });
  slide.addText([
    { text: "Statistical Rigor", options: { fontSize: 13, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "\u2022 Paired Wilcoxon signed-rank tests", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "   (139 image pairs per comparison)", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 5 } },
    { text: "\u2022 Holm-Bonferroni correction", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 5 } },
    { text: "\u2022 Cohen\u2019s d effect sizes", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 3 } },
    { text: "   with 95% bootstrap CIs", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
  ], {
    x: 6.5, y: 3.95, w: 3.0, h: 1.4,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 7: Q1 Results
// ---------------------------------------------------------------------------
function slide7_q1(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Q1: Frozen Model Leaderboard", "Which SSL paradigm best aligns with expert perception?");
  addAccentBar(slide);
  addSlideNumber(slide, 7);

  // Leaderboard bar chart (native PptxGenJS)
  slide.addChart(pres.charts.BAR, [
    {
      name: "IoU @ 90th %ile",
      labels: ["DINOv3", "ResNet50", "DINOv2", "CLIP", "SigLIP", "SigLIP2", "MAE"],
      values: [0.133, 0.090, 0.082, 0.049, 0.047, 0.047, 0.037],
    },
  ], {
    x: 0.4, y: 1.3, w: 4.5, h: 3.5,
    showTitle: true, title: "Frozen IoU @ 90th Percentile",
    titleFontSize: 12, titleColor: C.BODY,
    chartColors: [C.STEEL],
    catAxisLabelColor: C.BODY, catAxisLabelFontSize: 10,
    valAxisLabelColor: C.MUTED, valAxisLabelFontSize: 9,
    valGridLine: { color: C.LIGHT_GRAY, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9,
    dataLabelColor: C.BODY,
    valAxisMaxVal: 0.16,
  });

  // Key findings
  slide.addShape("roundedRectangle", {
    x: 5.2, y: 1.3, w: 4.4, h: 3.5,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "CCCCCC", opacity: 0.3 },
  });
  slide.addText([
    { text: "Key Findings", options: { fontSize: 15, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Self-distillation dominates", options: { fontSize: 13, bold: true, color: C.BODY, breakLine: true } },
    { text: "DINOv3 achieves 1.6\u00d7 the IoU of its nearest SSL competitor (DINOv2)", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Supervised ResNet-50 ranks #2", options: { fontSize: 13, bold: true, color: C.BODY, breakLine: true } },
    { text: "ImageNet labels provide strong localization even without transformer attention", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 8 } },
    { text: "CLIP peaks at layer 0", options: { fontSize: 13, bold: true, color: C.BODY, breakLine: true } },
    { text: "Language-image alignment pushes semantic attention to early layers", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 8 } },
    { text: "MAE lowest alignment", options: { fontSize: 13, bold: true, color: C.BODY, breakLine: true } },
    { text: "Pixel reconstruction \u2260 object localization; attention is diffuse", options: { fontSize: 11, color: C.MUTED, breakLine: true } },
  ], {
    x: 5.4, y: 1.4, w: 4.0, h: 3.3,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 8: Q2 Setup
// ---------------------------------------------------------------------------
function slide8_q2setup(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Q2: Fine-Tuning Experiment Design", "Does task-specific training shift attention toward expert features?");
  addAccentBar(slide);
  addSlideNumber(slide, 8);

  // Three strategy cards
  const strategies = [
    { name: "Linear Probe", params: "~3K", desc: "Classifier head only\nBackbone frozen", color: C.TEAL, purpose: "Baseline \u2014 attention unchanged" },
    { name: "LoRA", params: "~300K", desc: "Low-rank adapters on\nattention layers + head", color: C.STEEL, purpose: "Parameter-efficient; preserves pre-training" },
    { name: "Full Fine-tune", params: "~86M", desc: "Entire backbone +\nclassifier head", color: C.TERRA, purpose: "Maximum adaptation capacity" },
  ];

  strategies.forEach((s, i) => {
    const x = 0.5 + i * 3.1;

    slide.addShape("roundedRectangle", {
      x: x, y: 1.3, w: 2.9, h: 2.3,
      fill: { color: C.WHITE },
      rectRadius: 0.06,
      shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "CCCCCC", opacity: 0.3 },
    });

    // Top accent
    slide.addShape("rect", {
      x: x, y: 1.3, w: 2.9, h: 0.06,
      fill: { color: s.color },
    });

    slide.addText(s.name, {
      x: x + 0.15, y: 1.45, w: 2.6, h: 0.35,
      fontSize: 16, fontFace: FONT.BODY, color: s.color, bold: true, margin: 0,
    });

    slide.addText(s.params + " params", {
      x: x + 0.15, y: 1.8, w: 2.6, h: 0.25,
      fontSize: 20, fontFace: FONT.MONO, color: C.BODY, bold: true, margin: 0,
    });

    slide.addText(s.desc, {
      x: x + 0.15, y: 2.15, w: 2.6, h: 0.55,
      fontSize: 11, fontFace: FONT.BODY, color: C.MUTED, margin: 0,
    });

    slide.addText(s.purpose, {
      x: x + 0.15, y: 2.8, w: 2.6, h: 0.35,
      fontSize: 10, fontFace: FONT.BODY, color: C.BODY, italic: true, margin: 0,
    });
  });

  // Training config box
  slide.addShape("roundedRectangle", {
    x: 0.5, y: 3.85, w: 9.0, h: 1.3,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
  });

  slide.addText([
    { text: "Training Configuration:  ", options: { fontSize: 12, bold: true, color: C.STEEL } },
    { text: "4-class style classification (Romanesque, Gothic, Renaissance, Baroque)  \u2022  ", options: { fontSize: 11 } },
    { text: "3 epochs, batch 16  \u2022  ", options: { fontSize: 11 } },
    { text: "Cosine LR + warmup  \u2022  ", options: { fontSize: 11 } },
    { text: "Class-weighted loss  \u2022  ", options: { fontSize: 11 } },
    { text: "139 annotated images strictly excluded from training", options: { fontSize: 11, bold: true, color: C.SUCCESS } },
  ], {
    x: 0.7, y: 3.95, w: 8.6, h: 0.5,
    valign: "top", margin: 0,
  });

  slide.addText("Measurement:  \u0394 IoU = IoU(fine-tuned) \u2212 IoU(frozen), per image, paired Wilcoxon signed-rank tests with Holm correction", {
    x: 0.7, y: 4.5, w: 8.6, h: 0.4,
    fontSize: 12, fontFace: FONT.BODY, color: C.BODY, margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 9: Q2 Results
// ---------------------------------------------------------------------------
function slide9_q2results(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Q2: Fine-Tuning Shifts Attention", "But only for models with room to improve");
  addAccentBar(slide);
  addSlideNumber(slide, 9);

  // Embed the diverging bars figure
  slide.addImage({
    path: figPath("03_all_metrics_diverging_bars.png"),
    x: 0.2, y: 1.2, w: 5.5, h: 4.0,
    sizing: { type: "contain", w: 5.5, h: 4.0 },
  });

  // Key takeaways on right
  slide.addShape("roundedRectangle", {
    x: 5.9, y: 1.2, w: 3.8, h: 4.0,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "CCCCCC", opacity: 0.3 },
  });

  slide.addText([
    { text: "Key Takeaways", options: { fontSize: 15, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 5, breakLine: true } },
    { text: "1. Contrastive models improve most", options: { fontSize: 12, bold: true, color: C.BODY, breakLine: true } },
    { text: "CLIP LoRA: \u0394 IoU +0.063 (d=1.33)", options: { fontSize: 11, color: C.SUCCESS, breakLine: true } },
    { text: "SigLIP Full: \u0394 IoU +0.036 (d=0.78)", options: { fontSize: 11, color: C.SUCCESS, breakLine: true, paraSpaceAfter: 6 } },
    { text: "2. DINO: ceiling effect", options: { fontSize: 12, bold: true, color: C.BODY, breakLine: true } },
    { text: "Already well-aligned \u2192 \u0394 \u2248 0", options: { fontSize: 11, color: C.MUTED, breakLine: true } },
    { text: "Full FT risks regression (DINOv2: -0.003)", options: { fontSize: 11, color: C.FAIL, breakLine: true, paraSpaceAfter: 6 } },
    { text: "3. LoRA \u2265 Full for attention shift", options: { fontSize: 12, bold: true, color: C.BODY, breakLine: true } },
    { text: "CLIP LoRA (+0.063) > Full (+0.041)", options: { fontSize: 11, color: C.BODY, breakLine: true } },
    { text: "285\u00d7 fewer parameters, no forgetting", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Best validation accuracies:", options: { fontSize: 11, bold: true, color: C.BODY, breakLine: true } },
    { text: "DINOv3 LoRA: 91.2% \u2022 CLIP Full: 89.6%\nSigLIP Full: 88.0% \u2022 MAE Full: 75.2%", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
  ], {
    x: 6.1, y: 1.3, w: 3.4, h: 3.8,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 10: App Demo
// ---------------------------------------------------------------------------
function slide10_demo(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Interactive Visualization Platform", "Precompute \u2192 HDF5/SQLite/PNG cache \u2192 FastAPI (25 endpoints) \u2192 React + Vite");
  addAccentBar(slide);
  addSlideNumber(slide, 10);

  // 2x2 screenshot grid
  const screenshots = [
    { file: "screenshot_gallery.png", label: "Gallery" },
    { file: "screenshot_dashboard.png", label: "Dashboard" },
    { file: "screenshot_image_detail.png", label: "Image Detail" },
    { file: "screenshot_compare.png", label: "Compare" },
  ];

  screenshots.forEach((s, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.4 + col * 4.7;
    const y = 1.2 + row * 2.15;

    slide.addImage({
      path: imgPath(s.file),
      x: x, y: y, w: 4.4, h: 1.85,
      sizing: { type: "contain", w: 4.4, h: 1.85 },
    });

    // Label
    slide.addText(s.label, {
      x: x, y: y + 1.85, w: 4.4, h: 0.22,
      fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, align: "center",
    });
  });
}

// ---------------------------------------------------------------------------
// SLIDE 11: Engineering
// ---------------------------------------------------------------------------
function slide11_engineering(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Engineering Highlights", "Built for reproducibility and scale");
  addAccentBar(slide);
  addSlideNumber(slide, 11);

  // Architecture flow (shapes)
  const flowBoxes = [
    { label: "WikiChurches\nDataset", x: 0.3, color: C.WARM_GRAY },
    { label: "Precompute\nPipeline", x: 2.3, color: C.STEEL },
    { label: "FastAPI\nBackend", x: 4.8, color: C.TEAL },
    { label: "React\nFrontend", x: 7.3, color: C.TERRA },
  ];

  flowBoxes.forEach((b) => {
    slide.addShape("roundedRectangle", {
      x: b.x, y: 1.35, w: 1.7, h: 0.9,
      fill: { color: b.color },
      rectRadius: 0.06,
    });
    slide.addText(b.label, {
      x: b.x, y: 1.35, w: 1.7, h: 0.9,
      fontSize: 12, fontFace: FONT.BODY, color: C.WHITE, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
  });

  // Arrows
  [2.05, 4.5, 7.0].forEach((ax) => {
    slide.addText("\u25B6", {
      x: ax, y: 1.55, w: 0.3, h: 0.5,
      fontSize: 16, color: C.MUTED, align: "center", valign: "middle",
    });
  });

  // Output formats under precompute
  slide.addText("HDF5  \u2022  SQLite  \u2022  PNG", {
    x: 2.3, y: 2.3, w: 2.2, h: 0.25,
    fontSize: 9, fontFace: FONT.MONO, color: C.MUTED, align: "center",
  });

  // Stats
  const stats = [
    { num: "7", label: "Vision models with unified VisionBackbone protocol" },
    { num: "25", label: "REST API endpoints across 4 routers" },
    { num: "5", label: "Precompute scripts (attention, features, heatmaps, metrics)" },
    { num: "50", label: "Tracked issues via bd (beads), 48 closed" },
    { num: "0", label: "GPU needed at runtime \u2014 all cache-served, <100ms responses" },
  ];

  stats.forEach((s, i) => {
    const y = 2.75 + i * 0.52;
    slide.addText(s.num, {
      x: 0.5, y: y, w: 0.7, h: 0.4,
      fontSize: 22, fontFace: FONT.TITLE, color: C.STEEL, bold: true,
      align: "right", valign: "middle", margin: 0,
    });
    slide.addText(s.label, {
      x: 1.35, y: y, w: 5.0, h: 0.4,
      fontSize: 13, fontFace: FONT.BODY, color: C.BODY,
      valign: "middle", margin: 0,
    });
  });

  // Quality box
  slide.addShape("roundedRectangle", {
    x: 6.5, y: 2.75, w: 3.1, h: 2.35,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "CCCCCC", opacity: 0.3 },
  });
  slide.addText([
    { text: "Quality Gates", options: { fontSize: 13, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 5, breakLine: true } },
    { text: "\u2022 pytest test suite", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 mypy type checking", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 ruff linting", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 IoU quantile bias fix", options: { fontSize: 11, breakLine: true } },
    { text: "   (torch.quantile \u2192 torch.topk)", options: { fontSize: 10, color: C.MUTED, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 Deep audit epic (#9ct)", options: { fontSize: 11, breakLine: true } },
    { text: "   metrics, frontend, security", options: { fontSize: 10, color: C.MUTED, breakLine: true } },
  ], {
    x: 6.7, y: 2.85, w: 2.7, h: 2.15,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 12: Roadmap
// ---------------------------------------------------------------------------
function slide12_roadmap(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.LIGHT_BG };
  addSectionTitle(slide, "Remaining Work & Roadmap");
  addAccentBar(slide);
  addSlideNumber(slide, 12);

  // Timeline
  const phases = [
    { label: "Q1\nFrozen Alignment", x: 0.5, w: 2.8, color: C.SUCCESS, status: "Complete" },
    { label: "Q2\nFine-Tuning Effects", x: 3.5, w: 2.8, color: C.SUCCESS, status: "Complete" },
    { label: "Q3\nPer-Head Analysis", x: 6.5, w: 3.0, color: C.TEAL, status: "Next" },
  ];

  // Timeline line
  slide.addShape("rect", {
    x: 0.5, y: 2.0, w: 9.0, h: 0.04,
    fill: { color: C.LIGHT_GRAY },
  });

  phases.forEach((p) => {
    slide.addShape("roundedRectangle", {
      x: p.x, y: 1.4, w: p.w, h: 0.5,
      fill: { color: p.color },
      rectRadius: 0.05,
    });
    slide.addText(p.label, {
      x: p.x, y: 1.4, w: p.w, h: 0.5,
      fontSize: 12, fontFace: FONT.BODY, color: C.WHITE, bold: true,
      align: "center", valign: "middle", margin: 0,
    });
    slide.addText(p.status, {
      x: p.x, y: 2.05, w: p.w, h: 0.25,
      fontSize: 10, fontFace: FONT.BODY, color: C.MUTED, align: "center",
    });
  });

  // Q3 details
  slide.addShape("roundedRectangle", {
    x: 0.5, y: 2.6, w: 5.5, h: 2.5,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
    shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "CCCCCC", opacity: 0.3 },
  });
  slide.addText([
    { text: "Q3: Per-Head Attention Specialization", options: { fontSize: 14, bold: true, color: C.STEEL, breakLine: true } },
    { text: "(Primary remaining work)", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 8 } },
    { text: "\u2022 Compute IoU for each of 12 attention heads individually", options: { fontSize: 12, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 Build head \u00d7 feature-type matrix", options: { fontSize: 12, breakLine: true, paraSpaceAfter: 2 } },
    { text: "   Do specific heads specialize for windows, arches, buttresses?", options: { fontSize: 11, color: C.MUTED, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 Rank heads by alignment consistency across images", options: { fontSize: 12, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 Academic grounding: Voita et al. (2019), Caron et al. (2021)", options: { fontSize: 11, color: C.MUTED, breakLine: true } },
  ], {
    x: 0.7, y: 2.7, w: 5.1, h: 2.3,
    valign: "top", margin: 0,
  });

  // Open items
  slide.addShape("roundedRectangle", {
    x: 6.3, y: 2.6, w: 3.3, h: 2.5,
    fill: { color: C.WHITE },
    rectRadius: 0.06,
  });
  slide.addText([
    { text: "Open Items", options: { fontSize: 14, bold: true, color: C.STEEL, breakLine: true } },
    { text: "", options: { fontSize: 5, breakLine: true } },
    { text: "\u2022 Attention shift diff-map viz", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 Fine-tuning test coverage", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 Dashboard chart axis scaling", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Stretch Goals", options: { fontSize: 13, bold: true, color: C.TERRA, breakLine: true } },
    { text: "", options: { fontSize: 3, breakLine: true } },
    { text: "\u2022 Attention-supervised fine-tuning", options: { fontSize: 11, breakLine: true, paraSpaceAfter: 4 } },
    { text: "\u2022 COIN / core-tuning methods", options: { fontSize: 11, breakLine: true } },
  ], {
    x: 6.5, y: 2.7, w: 2.9, h: 2.3,
    valign: "top", margin: 0,
  });
}

// ---------------------------------------------------------------------------
// SLIDE 13: Summary
// ---------------------------------------------------------------------------
function slide13_summary(pres) {
  const slide = pres.addSlide();
  slide.background = { color: C.CHARCOAL };

  // Title
  slide.addText("Key Takeaways", {
    x: 0.6, y: 0.4, w: 4.5, h: 0.6,
    fontSize: 28, fontFace: FONT.TITLE, color: C.WHITE, bold: true,
  });

  // Takeaway bullets
  slide.addText([
    { text: "Q1 \u2014 Frozen Alignment", options: { fontSize: 14, bold: true, color: C.TEAL, breakLine: true } },
    { text: "Self-distillation (DINO) produces the most expert-aligned attention. DINOv3 leads at IoU = 0.133, 1.6\u00d7 its nearest SSL rival.", options: { fontSize: 12, color: C.WHITE, breakLine: true, paraSpaceAfter: 12 } },
    { text: "Q2 \u2014 Fine-Tuning Effects", options: { fontSize: 14, bold: true, color: C.TEAL, breakLine: true } },
    { text: "LoRA is the best strategy: equal or greater attention shift than full fine-tuning with 285\u00d7 fewer parameters and no catastrophic forgetting.", options: { fontSize: 12, color: C.WHITE, breakLine: true, paraSpaceAfter: 12 } },
    { text: "The Bigger Picture", options: { fontSize: 14, bold: true, color: C.TERRA, breakLine: true } },
    { text: "High classification accuracy \u2260 expert-aligned attention. This framework generalises to any domain with expert annotations.", options: { fontSize: 12, color: C.WHITE, breakLine: true } },
  ], {
    x: 0.6, y: 1.2, w: 4.5, h: 3.3,
    valign: "top", margin: 0,
  });

  // Next step
  slide.addText("Next: Q3 per-head specialization analysis \u2192 final presentation", {
    x: 0.6, y: 4.7, w: 4.5, h: 0.4,
    fontSize: 12, fontFace: FONT.BODY, color: C.WARM_GRAY, italic: true,
  });

  // Scatter plot on right
  slide.addImage({
    path: imgPath("slide13_scatter.png"),
    x: 5.2, y: 0.3, w: 4.6, h: 4.5,
    sizing: { type: "contain", w: 4.6, h: 4.5 },
  });

  // Caption
  slide.addText("Frozen IoU vs \u0394 IoU: pre-training determines alignment & plasticity", {
    x: 5.2, y: 4.85, w: 4.6, h: 0.3,
    fontSize: 9, fontFace: FONT.BODY, color: C.MUTED, align: "center",
  });

  // Bottom accent
  slide.addShape("rect", {
    x: 0, y: 5.4, w: 10, h: 0.225,
    fill: { color: C.STEEL },
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  console.log("Creating presentation...");

  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "SSL WikiChurches Team";
  pres.title = "SSL Attention Alignment - Mid-Project Progress Update";

  slide1_title(pres);
  console.log("  Slide 1: Title");

  slide2_motivation(pres);
  console.log("  Slide 2: Motivation");

  slide3_rqs(pres);
  console.log("  Slide 3: Research Questions");

  slide4_dataset(pres);
  console.log("  Slide 4: Dataset");

  slide5_models(pres);
  console.log("  Slide 5: Models");

  slide6_methodology(pres);
  console.log("  Slide 6: Methodology");

  slide7_q1(pres);
  console.log("  Slide 7: Q1 Results");

  slide8_q2setup(pres);
  console.log("  Slide 8: Q2 Setup");

  slide9_q2results(pres);
  console.log("  Slide 9: Q2 Results");

  slide10_demo(pres);
  console.log("  Slide 10: App Demo");

  slide11_engineering(pres);
  console.log("  Slide 11: Engineering");

  slide12_roadmap(pres);
  console.log("  Slide 12: Roadmap");

  slide13_summary(pres);
  console.log("  Slide 13: Summary");

  await pres.writeFile({ fileName: OUT_FILE });
  console.log(`\nDone! Presentation saved to: ${OUT_FILE}`);
}

main().catch(console.error);
