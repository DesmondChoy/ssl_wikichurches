# Pipeline Diagram (Mermaid source)

Source for the end-to-end ML pipeline figure used in the final report
(Section 3, "Proposed Approach"). Render via the Mermaid Live editor
(<https://mermaid.live>) or the VSCode Mermaid preview, then export PNG to
`docs/final_report/figures/pipeline.png`.

```mermaid
flowchart TB
    %% ============ Lane 1: Data ============
    subgraph DATA[" Data "]
        direction TB
        WC["<b>WikiChurches</b><br/><i>9,485 images</i>"]
        FT["FT pool<br/><i>style-labelled</i>"]
        EV["<b>Eval set</b><br/>139 imgs, 631 boxes<br/><i>held out</i>"]
        WC --> FT
        WC -. disjoint .-> EV
    end

    %% ============ Lane 2: Models ============
    subgraph MODELS[" Models "]
        direction TB
        D["DINOv2 + DINOv3<br/><i>self-distillation</i>"]
        MA["MAE<br/><i>masked AE</i>"]
        CL["CLIP + SigLIP {1,2}<br/><i>language-image</i>"]
        RN["ResNet-50<br/><i>supervised CNN</i>"]
    end

    %% ============ Lane 3: Adaptation ============
    subgraph ADAPT[" Adaptation "]
        direction TB
        FRZ["Frozen (Q1)"]
        LP["Linear Probe<br/><i>ΔA = 0 control</i>"]
        LORA["LoRA"]
        FULL["Full FT<br/><i>ResNet-50 excluded</i>"]
    end

    %% ============ Lane 4: Attention ============
    subgraph ATTN[" Attention "]
        direction TB
        ACLS["CLS attention"]
        ROLL["Rollout"]
        MEAN["Mean attention<br/><i>(proxy)</i>"]
        GC["Grad-CAM"]
    end

    %% ============ Lane 5: Metrics ============
    subgraph METRICS[" Metrics "]
        direction TB
        TGT["Expert boxes<br/>→ M, G"]
        MET["<b>IoU | Cov | MSE<br/>KL | EMD</b>"]
        BASE["Naive baselines<br/><i>random / center<br/>saliency / Sobel</i>"]
        TGT --> MET
        BASE -- calibrate --> MET
    end

    %% ============ Lane 6: Cache + outputs ============
    subgraph OUT[" Cache &amp; outputs "]
        direction TB
        H5[("HDF5<br/><i>attention</i>")]
        DB[("SQLite<br/><i>metrics</i>")]
        Q1["<b>Q1</b> leaderboard"]
        Q2["<b>Q2</b> ΔFT effects"]
        Q3["<b>Q3</b> per-head<br/><i>native CLS-tok only</i>"]
        STAT["Wilcoxon + Holm<br/>bootstrap CIs"]
        H5 --> Q1
        DB --> Q2
        DB --> Q3
        Q2 --- STAT
        Q3 --- STAT
    end

    %% ============ Lane-to-lane bundle flows ============
    FT ==> MODELS
    MODELS ==> ADAPT
    ADAPT ==> ATTN
    ATTN ==> MET
    EV -. expert targets .-> TGT
    MET ==> H5
    MET ==> DB

    %% ============ Styling ============
    classDef data    fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef model   fill:#fff3e0,stroke:#f57c00,color:#000
    classDef adapt   fill:#e8f5e9,stroke:#388e3c,color:#000
    classDef attn    fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef metric  fill:#fffde7,stroke:#f9a825,color:#000
    classDef cache   fill:#ffebee,stroke:#c62828,color:#000
    classDef output  fill:#e0f7fa,stroke:#0097a7,color:#000
    classDef note    fill:#fafafa,stroke:#9e9e9e,color:#555,font-style:italic

    class WC,FT,EV data
    class D,MA,CL,RN model
    class FRZ,LP,LORA,FULL adapt
    class ACLS,ROLL,MEAN,GC attn
    class TGT,MET,BASE metric
    class H5,DB cache
    class Q1,Q2,Q3 output
    class STAT note
```

## What this encodes (matches Section 3 of the report)

1. **The 139-image eval set is disjoint from fine-tuning.** Dashed
   `disjoint` arrow from WikiChurches to the Eval set; no path from the
   FT pool to Eval.
2. **Linear Probe is a zero-Δ control by construction** — annotated
   on the LP node so the §4.4 explanation isn't load-bearing on its own.
3. **CLS-token vs. proxy split drives Q3 scope.** Q3 output node is
   annotated *"native CLS-tok only"*, matching §3.6.
4. **Baselines are part of evaluation, not a sidebar.** They sit in the
   Metrics lane with an explicit `calibrate` edge into the metric node.
5. **Cache is the join point between research and app.** HDF5 + SQLite
   sit between metrics and the Q1/Q2/Q3 result surfaces.

## Export workflow

1. Open the fenced ````mermaid ```` block in <https://mermaid.live>
   (or right-click the rendered diagram in VSCode's preview).
2. Export PNG at high DPI (the Live editor's "PNG" button supports
   2x/3x scale).
3. Save to `docs/final_report/figures/pipeline.png`.
4. Add to the report at the top of Section 3 with a standard
   `\begin{figure*}[t] \includegraphics[width=\textwidth]{pipeline.png}`
   block.
