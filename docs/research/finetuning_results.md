## Fine-tuning Results Analysis
**Note(KM): In the validation images, not all the key features were labeled with bounding boxes. One potential extension here is to observe if the fine-tuned model outputs have strong attention regions on those features as well. This would not be accounted for in our current metric calculation.** 


---

### Why classification-only fine-tuning still increases IoU in CLIP/SigLIP/MAE

Even though the training signal is just a class label, that label implicitly carries spatial information in typical vision datasets: the class corresponds to the *primary object* in the image, which is exactly what the bounding box annotates. When you apply a classification head on top of the [CLS] token and backpropagate the cross-entropy loss, gradients flow back through the attention layers and effectively ask: *which patches are most discriminative for this class?* In most images, the answer is the object region. So the attention reweights toward it, and your IoU measurement picks this up.

This is essentially the same mechanism behind **GradCAM and attention rollout** for classification ViTs — the classification objective acts as a weak spatial inductive signal even though no bounding boxes are seen during training. Naseer et al. (2021 — [arXiv:2105.10497](https://arxiv.org/abs/2105.10497)) in "Intriguing Properties of Vision Transformers" document exactly this: ViT attention maps shift substantially when fine-tuned on classification, becoming more class-discriminative and spatially concentrated. For models whose pretraining left attention diffuse (CLIP, SigLIP, MAE), classification fine-tuning does real work reshaping where attention lands.

For **MAE** specifically, the reconstruction pretraining distributes attention broadly to capture texture context needed for pixel-level reconstruction — there's no pressure for any single patch to "own" the class identity. Classification fine-tuning is the first signal that forces the model to locate the object, explaining the IoU jump.

For **CLIP/SigLIP**, the global contrastive objective never back-propagated gradients specifically for object localization. Classification fine-tuning is again the first objective that rewards attending to the primary object rather than globally aggregating scene features.

---

### Why DINOv2/3 barely moves — and why this is actually the more telling result

The ceiling effect still holds, but your clarified setup adds a second, arguably more important interpretation: **DINOv2's spatial attention is robust to objective shift**.

When you fine-tune DINOv2 on classification, the [CLS] token is already aggregating from semantically relevant, object-centric patches. The gradient of the classification loss finds a model that already attends to the right regions, so there is little pressure to change the attention pattern — you're essentially confirming an existing alignment rather than creating a new one.

This is a property Park & Kim (2022 — [arXiv:2202.06709](https://arxiv.org/abs/2202.06709)) connect to the *spectral structure* of attention in discriminatively pretrained ViTs: the attention is already low-entropy (concentrated) before fine-tuning, leaving less room for the classification objective to reshape it.

A deeper implication: for CLIP/SigLIP/MAE, the IoU improvement you're seeing is partly a sign that the pretrained representations needed the fine-tuning to become spatially coherent. For DINOv2, the spatial coherence was a *byproduct of pretraining*, not a product of supervision — which also means it generalizes more robustly across domains, whereas the CLIP/SigLIP localization you measured may be more tightly coupled to the specific classification vocabulary it was fine-tuned on.
