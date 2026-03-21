可以，我幫你整理成一個**融合後的簡化 pipeline**，並且明確標出：

* **[FFTAT]**：原本強基線自己的 component
* **[JFPD]**：你們要加上去的 component
* **[融合設計]**：不是原 paper 直接寫的，而是把兩者接起來時最合理的實作方式

先講核心原則：**FFTAT 繼續負責 patch-level transferable modeling，JFPD 只掛在最後的 global feature / prediction 上**。這樣分工最清楚，也最不容易互相干擾。FFTAT 本來就有 feature fusion、patch discriminator、transferability graph-guided attention、classifier、domain discriminator、self-clustering 這些模組；而 JFPD 本質上是用 **feature prototype + prediction prototype + trust weighting** 去做 target 對齊。 

---

## 融合後 pipeline

### Step 1. Source / Target 影像輸入

[
x_s,; x_t
]

### Step 2. Patch embedding + positional embedding + class token

**[FFTAT]**

source / target 影像都先切成 patches，投影到 latent space，再加上 positional embedding，並 prepend 一個 class token。這就是 FFTAT 的 ViT backbone 起點。

---

### Step 3. Feature Fusion Layer

**[FFTAT]**

在進入 transferability-aware transformer 之前，先做 feature fusion，讓每個 sample 的 patch embedding 融合 batch 內其他 sample 的資訊，提升 graph 的穩定性與 generalization。FFTAT 的 feature fusion 是在 latent space 做，不是在 image space。

輸出可記成：
[
\tilde{B}_s,; \tilde{B}_t
]

---

### Step 4. Patch Discriminator → Transferability Graph

**[FFTAT]**

FFTAT 會對 patch token 做 transferability assessment，得到 patch-level 的 transferability score，進一步形成 transferability graph / adjacency matrix。這是 FFTAT 很核心的地方。

輸出可記成：
[
M_{ts}
]

---

### Step 5. Transferability-Aware / TG-guided Transformer Layers

**[FFTAT]**

FFTAT 用 transferability graph 去改寫 self-attention，讓 attention 更偏向高 transferable patches。這部分包含：

* class token 對 patch 的 transferability-aware self-attention
* patch-level 的 transferability graph-guided self-attention

也就是 FFTAT 最有辨識度的 backbone 主體。

輸出最後一層表示：

* patch tokens
* final class token (h)

---

### Step 6. Classifier head

**[FFTAT]**

把 final class token / global representation 丟進 classifier，得到 logits 與 prediction：

[
p_s = g(h_s),\qquad p_t = g(h_t)
]

FFTAT 的分類 head 本來就存在，source 上用 cross-entropy 監督。

---

## FFTAT 原生訓練分支

### Step 7A. Source classification loss

**[FFTAT]**

[
L_{clc}
]

這是 source label supervision。不能拿掉。

### Step 7B. Domain discriminator on class token

**[FFTAT]**

FFTAT 的 domain discriminator 是吃 **class token / image-level representation**，分 source / target domain。

[
L_{dis}
]

### Step 7C. Patch discriminator loss

**[FFTAT]**

這個 loss 用來學 patch 的 transferability，支撐 transferability graph。

[
L_{pat}
]

### Step 7D. Self-clustering on target

**[FFTAT]，可選保留**

FFTAT 還有 target 的 self-clustering / mutual information maximization 分支。

[
L_{sc}
]

---

## JFPD 外掛分支

### Step 8. 用 FFTAT 最後的 global feature 當成 JFPD feature

**[融合設計]**

這一步不是 paper 直接寫 FFTAT+JFPD，而是**我建議的接法**：

* 用 FFTAT 最後一層的 **class token / classifier 前 global embedding** 當作 JFPD 的 feature (f)
* 不去碰 patch token

也就是：
[
f_s := h_s,\qquad f_t := h_t
]

原因是 FFTAT 的 patch token 已經被 patch discriminator + transferability graph 佔滿角色；JFPD 比較適合接在 image-level / semantic-level 對齊。這個設計是根據 FFTAT 的 image-level class token 分支與 JFPD 的 feature-prediction joint discrepancy 架構推導出的融合方式。

---

### Step 9. 建 source prototypes

**[JFPD]**

JFPD 會為每個 class 建兩種 prototype：

1. **feature prototype**
   [
   z_c^s = \frac{1}{|D_c^s|}\sum f(x_i^s)
   ]

2. **prediction prototype**
   [
   p_c^s = \frac{1}{|D_c^s|}\sum g(f(x_i^s))
   ]

這是 JFPD 的核心設計。

在你們這裡就是：

* feature prototype 用 FFTAT 的 global feature 平均
* prediction prototype 用 FFTAT classifier 的 softmax 平均

---

### Step 10. Target pseudo-label

**[JFPD]**

對 target sample 先用目前 classifier 輸出做 pseudo label：

[
\hat y_t = \arg\max_c p_t^{(c)}
]

然後把 target sample 對齊到對應的 source class prototypes
[
(z_{\hat y_t}^s,; p_{\hat y_t}^s)
]
。

---

### Step 11. Trust weights

**[JFPD]**

JFPD 有兩個 trust：

1. **uncertainty-aware trust**
   [
   \psi = \frac{1}{1 + H(p_s) + H(p_t)}
   ]

2. **semantic-alignment trust**
   [
   \phi = \frac{1}{1 + d_{feat}(f_s,f_t)}
   ]

其中一個看 prediction entropy，另一個看 feature proximity。

在融合版裡，實作上可寫成 target 與其對應 prototype 的 trust：

* 用 (p_t) 和 prototype prediction 估可信度
* 用 (f_t) 和 prototype feature 算 semantic alignment

這是 prototype 版 JFPD 的自然落地方式。

---

### Step 12. JFPD loss

**[JFPD]**

最後對每個 target sample 算：

[
L_{JFPD}
========

\alpha \psi, d_{feat}(f_t, z_{\hat y_t}^s)
+
(1-\alpha)\phi, d_{pred}(p_t, p_{\hat y_t}^s)
]

也就是 joint feature-prediction discrepancy，用 trust 去調節 feature discrepancy 和 prediction discrepancy。

---

## 最後總 loss

### 建議版

**[融合設計]**

[
L_{total}
=========

L_{clc}
+\lambda_{dis}L_{dis}
+\lambda_{pat}L_{pat}
+\lambda_{sc}L_{sc}
+\lambda_j L_{JFPD}
]

其中：

* (L_{clc}, L_{dis}, L_{pat}, L_{sc}) 來自 **FFTAT**
* (L_{JFPD}) 來自 **JFPD**

---

## 哪些 component 屬於誰

### FFTAT 的 components

* Patch embedding / positional embedding / class token
* Feature Fusion Layer
* Patch Discriminator
* Transferability Graph
* Transferability-Aware Self-Attention
* Transferability Graph Guided Transformer Layers
* Classifier Head
* Domain Discriminator
* Self-Clustering module
* 對應的 loss：(L_{clc}, L_{dis}, L_{pat}, L_{sc})

### JFPD 的 components

* Source feature prototypes (z_c^s)
* Source prediction prototypes (p_c^s)
* Target pseudo-label assignment
* Uncertainty-aware trust (\psi)
* Semantic-alignment trust (\phi)
* Joint feature-prediction discrepancy
* 對應的 loss：(L_{JFPD})

### 你們這次 fusion 的關鍵設計

* **FFTAT 負責 patch-level transferable learning**
* **JFPD 負責最後 global feature / prediction 的 trust-aware class-conditional alignment**
* **JFPD 不碰 patch graph，不改 FFTAT backbone 主流程**
* **JFPD 掛在 final class token 之後最穩**
  這幾點是融合策略，不是兩篇 paper 原文直接給的 combined architecture。是根據兩者模組職責做的最合理接法。

---

## 一句話版本

你可以把整個融合想成：

[
\text{FFTAT backbone 做 patch-level transferable representation}
]
[
\text{JFPD head 做 global prototype-based trustworthy alignment}
]

也就是：

**前半段是 FFTAT 在學「哪些 patch 可轉移、怎麼用 graph attention 聚合」；後半段是 JFPD 在學「target 這個 global feature / prediction 應該可信地靠近哪個 source class prototype」。**

如果你要，我下一步可以直接幫你畫成**論文裡可放的 method 流程圖文字版**，或直接寫成 **PyTorch training step 偽碼**。
