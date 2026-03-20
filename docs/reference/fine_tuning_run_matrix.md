# Fine-Tuning Run Matrix

Git-tracked reference for the current fine-tuning run settings and checkpoint-backed
training results. This page exists because `outputs/` is ignored by git, so the
raw artifacts in `outputs/checkpoints/` and `outputs/results/` are not durable team
documentation on their own.

This document summarizes three kinds of information:

- Strategy-level training settings shared across the current fine-tuned checkpoints
- Per-model training results that are already available directly from saved checkpoints
- A compact Q2 metric snapshot showing how each fine-tuned strategy shifts the
  frozen baseline on the metrics surfaced in the React app

It does **not** try to duplicate every generated analysis artifact. The full Q2
metrics JSON should still be generated into `outputs/results/` and surfaced in the
app/backend from there. This page only keeps the current human-readable summary.

## Strategy Comparison

| Strategy | What trains | Attention can change? | Epochs | Batch size | Backbone LR | Head LR | Extra params |
|---|---|---|---:|---:|---:|---:|---|
| full | Backbone + classifier head | Yes | 3 | 16 | 1e-05 | 0.001 | None |
| linear_probe | Classifier head only | No | 3 | 16 | 1e-05 | 0.001 | None |
| lora | LoRA adapters + classifier head | Yes | 3 | 16 | 0.0001 | 0.001 | rank=8, alpha=32, dropout=0.1 |

Notes:

- The current checkpoint set uses `3` epochs for all strategies.
- `linear_probe` records `learning_rate_backbone=1e-5` in config, but the backbone
  is frozen, so that value is effectively unused during training.
- `lora` automatically raises the backbone learning rate from `1e-5` to `1e-4`
  when the default config is left unchanged.

## Shared Training Settings

| Parameter | Value |
|---|---|
| Seed | `42` |
| Weight decay | `0.01` |
| Warmup ratio | `0.1` |
| Max grad norm | `1.0` |
| Data augmentation | `True` |
| Validation source | `annotated_eval` |
| Train samples | `4588` |
| Validation samples | `125` |
| Excluded eval samples | `125` |

## Per-Model Run Matrix

The table below is sourced from the saved checkpoint configs plus the best
checkpoint metadata in each `.pt` file.

| Model | Strategy | Best epoch | Best val acc | Backbone LR | Head LR | Backbone frozen? | LoRA? | Checkpoint |
|---|---:|---:|---:|---:|---:|---|---|---|
| clip | full | 1 | 89.6% | 1e-05 | 0.001 | No | No | `clip_full_finetuned.pt` |
| clip | linear_probe | 3 | 83.2% | 1e-05 | 0.001 | Yes | No | `clip_linear_probe_finetuned.pt` |
| clip | lora | 3 | 84.8% | 0.0001 | 0.001 | No | Yes | `clip_lora_finetuned.pt` |
| dinov2 | full | 2 | 87.2% | 1e-05 | 0.001 | No | No | `dinov2_full_finetuned.pt` |
| dinov2 | linear_probe | 1 | 89.6% | 1e-05 | 0.001 | Yes | No | `dinov2_linear_probe_finetuned.pt` |
| dinov2 | lora | 1 | 88.8% | 0.0001 | 0.001 | No | Yes | `dinov2_lora_finetuned.pt` |
| dinov3 | full | 2 | 89.6% | 1e-05 | 0.001 | No | No | `dinov3_full_finetuned.pt` |
| dinov3 | linear_probe | 1 | 90.4% | 1e-05 | 0.001 | Yes | No | `dinov3_linear_probe_finetuned.pt` |
| dinov3 | lora | 2 | 91.2% | 0.0001 | 0.001 | No | Yes | `dinov3_lora_finetuned.pt` |
| mae | full | 2 | 75.2% | 1e-05 | 0.001 | No | No | `mae_full_finetuned.pt` |
| mae | linear_probe | 2 | 60.0% | 1e-05 | 0.001 | Yes | No | `mae_linear_probe_finetuned.pt` |
| mae | lora | 3 | 72.8% | 0.0001 | 0.001 | No | Yes | `mae_lora_finetuned.pt` |
| siglip | full | 2 | 88.0% | 1e-05 | 0.001 | No | No | `siglip_full_finetuned.pt` |
| siglip | linear_probe | 1 | 85.6% | 1e-05 | 0.001 | Yes | No | `siglip_linear_probe_finetuned.pt` |
| siglip | lora | 2 | 87.2% | 0.0001 | 0.001 | No | Yes | `siglip_lora_finetuned.pt` |
| siglip2 | full | 1 | 88.8% | 1e-05 | 0.001 | No | No | `siglip2_full_finetuned.pt` |
| siglip2 | linear_probe | 3 | 81.6% | 1e-05 | 0.001 | Yes | No | `siglip2_linear_probe_finetuned.pt` |
| siglip2 | lora | 3 | 88.8% | 0.0001 | 0.001 | No | Yes | `siglip2_lora_finetuned.pt` |

## Q2 Metric Deltas vs Frozen Baseline

The tables below are sourced from `outputs/results/q2_metrics_analysis.json` and
summarize the aggregate Q2 metric shifts at the analyzed last layer (`layer 11`).
Each strategy cell is shown as `fine_mean (delta_vs_frozen)`.

- These tables were regenerated on `2026-03-20`.
- For `iou` and `coverage`, positive deltas are improvements.
- For `mse`, `kl`, and `emd`, negative deltas are improvements.
- `linear_probe` should usually stay close to the frozen baseline because the
  backbone attention is not updated.

### IoU at 90th Percentile

Higher is better. `Delta = fine - frozen` at the analyzed layer.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `0.0181` | `0.0181 (+0.0000)` | `0.0815 (+0.0634)` | `0.0587 (+0.0406)` |
| dinov2 | `0.0816` | `0.0816 (+0.0000)` | `0.0838 (+0.0021)` | `0.0787 (-0.0029)` |
| dinov3 | `0.1327` | `0.1327 (+0.0000)` | `0.1418 (+0.0090)` | `0.1353 (+0.0025)` |
| mae | `0.0359` | `0.0367 (+0.0008)` | `0.0372 (+0.0014)` | `0.0345 (-0.0014)` |
| siglip | `0.0364` | `0.0364 (+0.0000)` | `0.0614 (+0.0250)` | `0.0724 (+0.0360)` |
| siglip2 | `0.0220` | `0.0220 (+0.0000)` | `0.0486 (+0.0266)` | `0.0584 (+0.0364)` |

### IoU at 80th Percentile

Higher is better. `Delta = fine - frozen` at the analyzed layer.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `0.0514` | `0.0514 (+0.0000)` | `0.1003 (+0.0490)` | `0.0881 (+0.0367)` |
| dinov2 | `0.0922` | `0.0922 (+0.0000)` | `0.0953 (+0.0031)` | `0.0895 (-0.0027)` |
| dinov3 | `0.1375` | `0.1375 (+0.0000)` | `0.1396 (+0.0021)` | `0.1337 (-0.0038)` |
| mae | `0.0491` | `0.0510 (+0.0019)` | `0.0505 (+0.0014)` | `0.0478 (-0.0013)` |
| siglip | `0.0584` | `0.0584 (+0.0000)` | `0.0845 (+0.0262)` | `0.0933 (+0.0349)` |
| siglip2 | `0.0478` | `0.0478 (+0.0000)` | `0.0757 (+0.0279)` | `0.0855 (+0.0377)` |

### IoU at 70th Percentile

Higher is better. `Delta = fine - frozen` at the analyzed layer.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `0.0682` | `0.0682 (+0.0000)` | `0.1056 (+0.0374)` | `0.0999 (+0.0317)` |
| dinov2 | `0.0959` | `0.0959 (+0.0000)` | `0.0982 (+0.0023)` | `0.0930 (-0.0029)` |
| dinov3 | `0.1313` | `0.1313 (+0.0000)` | `0.1300 (-0.0013)` | `0.1256 (-0.0057)` |
| mae | `0.0571` | `0.0586 (+0.0015)` | `0.0580 (+0.0009)` | `0.0564 (-0.0007)` |
| siglip | `0.0678` | `0.0678 (+0.0000)` | `0.0896 (+0.0218)` | `0.0960 (+0.0282)` |
| siglip2 | `0.0588` | `0.0588 (+0.0000)` | `0.0838 (+0.0251)` | `0.0934 (+0.0346)` |

### IoU at 60th Percentile

Higher is better. `Delta = fine - frozen` at the analyzed layer.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `0.0766` | `0.0766 (+0.0000)` | `0.1070 (+0.0304)` | `0.1035 (+0.0269)` |
| dinov2 | `0.0958` | `0.0958 (+0.0000)` | `0.0971 (+0.0012)` | `0.0929 (-0.0029)` |
| dinov3 | `0.1230` | `0.1230 (+0.0000)` | `0.1204 (-0.0026)` | `0.1169 (-0.0061)` |
| mae | `0.0625` | `0.0633 (+0.0007)` | `0.0631 (+0.0005)` | `0.0617 (-0.0008)` |
| siglip | `0.0720` | `0.0720 (+0.0000)` | `0.0899 (+0.0179)` | `0.0937 (+0.0217)` |
| siglip2 | `0.0648` | `0.0648 (+0.0000)` | `0.0861 (+0.0214)` | `0.0941 (+0.0294)` |

### IoU at 50th Percentile

Higher is better. `Delta = fine - frozen` at the analyzed layer.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `0.0808` | `0.0808 (+0.0000)` | `0.1042 (+0.0234)` | `0.1013 (+0.0205)` |
| dinov2 | `0.0948` | `0.0948 (+0.0000)` | `0.0947 (-0.0001)` | `0.0911 (-0.0037)` |
| dinov3 | `0.1137` | `0.1137 (+0.0000)` | `0.1123 (-0.0014)` | `0.1090 (-0.0047)` |
| mae | `0.0662` | `0.0668 (+0.0006)` | `0.0667 (+0.0005)` | `0.0655 (-0.0007)` |
| siglip | `0.0744` | `0.0744 (+0.0000)` | `0.0887 (+0.0143)` | `0.0907 (+0.0163)` |
| siglip2 | `0.0683` | `0.0683 (+0.0000)` | `0.0859 (+0.0176)` | `0.0923 (+0.0240)` |

### Coverage Delta vs Frozen Baseline

Higher is better. `Delta = fine - frozen` at the analyzed layer.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `0.0510` | `0.0510 (+0.0000)` | `0.1085 (+0.0575)` | `0.0917 (+0.0407)` |
| dinov2 | `0.1004` | `0.1004 (+0.0000)` | `0.1026 (+0.0021)` | `0.1014 (+0.0010)` |
| dinov3 | `0.1373` | `0.1373 (+0.0000)` | `0.1359 (-0.0015)` | `0.1313 (-0.0060)` |
| mae | `0.0737` | `0.0739 (+0.0002)` | `0.0738 (+0.0001)` | `0.0730 (-0.0007)` |
| siglip | `0.0505` | `0.0505 (+0.0000)` | `0.0582 (+0.0077)` | `0.0654 (+0.0149)` |
| siglip2 | `0.0458` | `0.0458 (+0.0000)` | `0.0590 (+0.0132)` | `0.0618 (+0.0160)` |

### Gaussian MSE Delta vs Frozen Baseline

Lower is better. Negative deltas indicate improvement.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `0.0217` | `0.0217 (+0.0000)` | `0.0445 (+0.0228)` | `0.0244 (+0.0028)` |
| dinov2 | `0.0545` | `0.0545 (+0.0000)` | `0.0504 (-0.0040)` | `0.0435 (-0.0110)` |
| dinov3 | `0.0547` | `0.0547 (+0.0000)` | `0.0569 (+0.0022)` | `0.0556 (+0.0009)` |
| mae | `0.1053` | `0.1056 (+0.0003)` | `0.0466 (-0.0587)` | `0.0710 (-0.0343)` |
| siglip | `0.0181` | `0.0181 (+0.0000)` | `0.0180 (-0.0001)` | `0.0183 (+0.0003)` |
| siglip2 | `0.0190` | `0.0190 (+0.0000)` | `0.0190 (+0.0000)` | `0.0191 (+0.0001)` |

### KL Divergence Delta vs Frozen Baseline

Lower is better. Negative deltas indicate improvement.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `3.6873` | `3.6873 (+0.0000)` | `2.6332 (-1.0541)` | `2.9293 (-0.7580)` |
| dinov2 | `2.6842` | `2.6842 (+0.0000)` | `2.6676 (-0.0166)` | `2.7016 (+0.0174)` |
| dinov3 | `2.3247` | `2.3247 (+0.0000)` | `2.3176 (-0.0071)` | `2.3458 (+0.0211)` |
| mae | `3.1922` | `3.1870 (-0.0052)` | `3.3555 (+0.1633)` | `3.3532 (+0.1610)` |
| siglip | `3.6554` | `3.6554 (+0.0000)` | `3.4071 (-0.2483)` | `3.3182 (-0.3372)` |
| siglip2 | `3.7221` | `3.7221 (+0.0000)` | `3.3495 (-0.3726)` | `3.3172 (-0.4049)` |

### EMD Delta vs Frozen Baseline

Lower is better. Negative deltas indicate improvement.

| Model | Frozen | Linear probe | LoRA | Full |
|---|---:|---:|---:|---:|
| clip | `0.4096` | `0.4096 (+0.0000)` | `0.2959 (-0.1138)` | `0.3438 (-0.0658)` |
| dinov2 | `0.2978` | `0.2978 (+0.0000)` | `0.2967 (-0.0012)` | `0.2984 (+0.0006)` |
| dinov3 | `0.2600` | `0.2600 (+0.0000)` | `0.2620 (+0.0020)` | `0.2664 (+0.0063)` |
| mae | `0.3444` | `0.3454 (+0.0010)` | `0.3498 (+0.0054)` | `0.3464 (+0.0020)` |
| siglip | `0.4359` | `0.4359 (+0.0000)` | `0.4212 (-0.0146)` | `0.4070 (-0.0288)` |
| siglip2 | `0.4210` | `0.4210 (+0.0000)` | `0.3995 (-0.0215)` | `0.4021 (-0.0189)` |

## Best-Epoch Distribution

This helps show whether the current `3`-epoch sweep was obviously too short.

| Best epoch | Number of checkpoints |
|---:|---:|
| 1 | 6 |
| 2 | 7 |
| 3 | 5 |

Interpretation:

- Most runs peaked at epoch `1` or `2`.
- A minority peaked at epoch `3`.
- That means the current sweep does not look obviously undertrained, but it also
  does not prove that `3` is globally optimal for attention-shift analysis.

## Result Storage Map

This table is the recommended split between durable reference docs and generated
artifacts.

| Result family | Scope | Primary storage | Git-tracked? | Notes |
|---|---|---|---|---|
| Strategy defaults and run matrix | Strategy-level and model-level training setup | `docs/reference/fine_tuning_run_matrix.md` | Yes | Human-readable team reference |
| Checkpoints | `model × strategy` weights | `outputs/checkpoints/` | No | Source of truth for trained weights |
| Run manifests | `model × strategy` training metadata | `outputs/results/fine_tuning_manifests/` | No | Includes split counts, seed, and epoch count |
| Latest single-run summary | Most recently trained run only | `outputs/results/fine_tuning_results.json` | No | Overwritten by the latest run; not a full ledger |
| Q2 aggregate attention metrics | `model × strategy × metric` analysis | `outputs/results/q2_metrics_analysis.json` | No | Consumed by `/api/metrics/q2_summary` and `/q2` |
| Compare-page image and bbox metrics | Image-level / bbox-level attention analysis | `outputs/cache/metrics.db` plus on-demand backend computation | No | Depends on regenerated attention, feature, heatmap, and metrics caches |
