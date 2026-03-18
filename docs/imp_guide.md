Below is a clean, re-implementation-oriented **English implementation details** section you can use directly. I made the most important point fully explicit: **during adaptation, prototypes must always be computed from the source domain only; the target sample only uses its pseudo-label to select which source prototype to compare against.** 

## Implementation Details

We follow a two-stage unsupervised domain adaptation pipeline. Let the model be (h(x)=g(f(x))), where (f) is the feature extractor and (g) is the classifier. In the first stage, the model is pretrained on the labeled source domain using the standard cross-entropy loss. After source pretraining, class-level source prototypes are constructed in both the feature space and the prediction space. For class (c), the source feature prototype is defined as the mean feature embedding of all source samples belonging to class (c), and the source prediction prototype is defined as the mean softmax prediction of those same source samples. These prototypes summarize the source-domain class structure and serve as alignment anchors during target adaptation. 

Formally, for each class (c), the feature prototype (z_c^s) is the average of (f(x_i^s)) over source samples in class (c), and the prediction prototype (p_c^s) is the average of (g(f(x_i^s))) over the same source samples. During adaptation, a target sample is **not** matched to target-domain prototypes. Instead, it is always aligned to the corresponding **source** prototypes selected by its label or pseudo-label. This source-only prototype design is stated consistently in the prototype definition, the adaptation objective, and Algorithm 1.

In the second stage, the pretrained model is adapted on unlabeled target data. For each target sample (x_t), we compute its feature representation (f_t=f(x_t)) and prediction vector (p_t=g(f_t)). A pseudo-label is assigned as (\hat y_t=\arg\max_c p_t^{(c)}). The model then retrieves the corresponding **source** prototypes ((z_{\hat y_t}^s, p_{\hat y_t}^s)), computes the feature discrepancy between (f_t) and (z_{\hat y_t}^s), and computes the prediction discrepancy between (p_t) and (p_{\hat y_t}^s). The paper uses cosine distance as the default feature-space distance and Jensen-Shannon divergence as the default prediction-space divergence, with both terms normalized as (d/(1+d)) for numerical stability. 

The feature discrepancy is therefore implemented as
[
d_{\text{feat}}(f_t,z_{\hat y_t}^s)=\frac{d(f_t,z_{\hat y_t}^s)}{1+d(f_t,z_{\hat y_t}^s)},
]
where (d(\cdot,\cdot)) is cosine distance by default. The prediction discrepancy is implemented as
[
d_{\text{pred}}(p_t,p_{\hat y_t}^s)=\frac{D(p_t,p_{\hat y_t}^s)}{1+D(p_t,p_{\hat y_t}^s)},
]
where (D(\cdot,\cdot)) is Jensen-Shannon divergence by default. 

Two trust weights are used. The uncertainty-aware trust is defined from prediction entropy, and the semantic-alignment trust is defined from feature proximity. In the paper, trust is first introduced for a generic source-target pair; once the prototype-based adaptation objective is introduced, the consistent implementation is to substitute the retrieved **source prediction prototype** (p_{\hat y_t}^s) and **source feature prototype** (z_{\hat y_t}^s) in place of the generic source sample. Accordingly, a faithful re-implementation should compute
[
\psi_t=\frac{1}{1+H(p_t)+H(p_{\hat y_t}^s)},
\qquad
\phi_t=\frac{1}{1+d_{\text{feat}}(f_t,z_{\hat y_t}^s)}.
]
This is the key point that keeps the adaptation stage consistent with the source-prototype formulation.

The first baseline is **PGFD** (prediction-guided feature discrepancy), which keeps only the feature-space alignment term weighted by prediction confidence:
[
L_{\text{PGFD}}=\psi_t, d_{\text{feat}}(f_t,z_{\hat y_t}^s).
]
Although the name contains “prediction-guided,” this baseline actually optimizes a **feature-space** discrepancy, with the guidance coming from prediction confidence. In the full objective, this corresponds to the (\alpha=1) extreme.

The second baseline is **FGPD** (feature-guided prediction divergence), which keeps only the prediction-space alignment term weighted by feature alignment:
[
L_{\text{FGPD}}=\phi_t, d_{\text{pred}}(p_t,p_{\hat y_t}^s).
]
Again, the naming is easy to misread: FGPD is a **prediction-space** loss guided by feature alignment. In the full objective, this corresponds to the (\alpha=0) extreme. This naming should be kept very carefully in code, because swapping FGPD and PGFD is one of the easiest implementation mistakes in this paper.

The full method, **JFPD**, combines both terms:
[
L_{\text{JFPD}}=\alpha,\psi_t, d_{\text{feat}}(f_t,z_{\hat y_t}^s) + (1-\alpha),\phi_t, d_{\text{pred}}(p_t,p_{\hat y_t}^s).
]
The ablation discussion shows that performance is poor when the objective degenerates into prediction-guided feature discrepancy only, stronger when it degenerates into FGPD only, and best when both terms are balanced. The reported sensitivity analysis indicates that performance remains stable for (\alpha \in [0.1,0.9]) and peaks around (\alpha=0.5) to (0.6). For re-implementation, (\alpha=0.5) is therefore a sensible default unless you are reproducing the ablation exactly.

A critical implementation detail is the **dynamic prototype estimation** strategy. The paper does not keep a single frozen prototype bank computed once from the whole source dataset. Instead, at each training iteration, it randomly samples a subset of source examples from every class and recomputes mini-batch source prototypes. This stochastic prototype construction is intended to avoid overfitting to fixed summary statistics and to provide more diverse alignment anchors during adaptation. Importantly, even under this dynamic strategy, the prototypes are still built from **source samples only**, never from target features or pseudo-labeled target batches. The reported setting uses **32 source samples per class** to estimate prototypes.

In practical code, the safest adaptation loop is therefore: sample a target mini-batch; separately sample source examples for each class; recompute (z_c^s) and (p_c^s) from that source subset; run the current target batch through the network; assign pseudo-labels by argmax; retrieve the matching **source** prototypes; compute (d_{\text{feat}}), (d_{\text{pred}}), (\psi_t), and (\phi_t); then optimize FGPD, PGFD, or JFPD depending on the experiment. The pseudo-label is used only to choose the source class anchor. It must **not** be used to form target-domain class prototypes unless you intentionally want to deviate from the paper.

The paper also reports two external reference baselines, namely **scratch** and **standard fine-tuning**, but the two internal ablation baselines of the proposed framework are FGPD and PGFD. If your goal is to re-implement the method family itself, the core comparison should therefore be FGPD vs. PGFD vs. JFPD, with scratch and standard fine-tuning serving as additional reference points.

For the reported training setups, the paper uses benchmark-specific backbones and schedules. On Digits, it evaluates a lightweight VGG-style CNN and a small ViT-S; source pretraining lasts 60 epochs, and JFPD uses target batch sizes from 128 to 512 with 32 source samples per class for prototype estimation. On Office-Home, it uses ImageNet-pretrained ResNet-34, ResNet-50, ViT-B/32, and ViT-B/16, replaces the final layer with a 65-way classifier, pretrains on source for 100 epochs, and adapts for 50 epochs; JFPD again uses 32 source samples per class. On VisDA-2017, it uses an ImageNet-pretrained ResNet-101, trains on source for 30 epochs, adapts for 100 epochs, and uses 32 source samples per class for prototypes. For DomainNet, the visible paper text specifies ResNet-101 and ViT-B/16 with ImageNet-1K initialization and a 345-way output head, but the exact optimization schedule is not fully specified in the visible excerpt, so that part should be treated as under-specified unless you have the authors’ code.

If you want, I can next turn this into a **paper-style “Implementation Details” subsection** with a polished academic tone, or into a **step-by-step PyTorch checklist** for coding.
