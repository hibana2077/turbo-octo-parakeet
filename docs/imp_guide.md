可以，這篇其實可以先砍到只剩一條很乾淨的 **JFPD pipeline**：

1. 先用 **source domain 有標註資料**訓練一個分類模型。
2. 進入 adaptation 時，**target domain 不用標註**，只做：

   * 用目前模型算 target feature `f_t` 與 prediction `p_t`
   * 用 `argmax(p_t)` 當 pseudo-label
   * 從 source 端取出該類別的 **feature prototype** `z_s[c]` 與 **prediction prototype** `p_s[c]`
   * 算

     * feature divergence：`d_feat`
     * prediction divergence：`d_pred`
     * uncertainty trust：`ψ = 1 / (1 + H(p_s) + H(p_t))`
     * semantic trust：`ϕ = 1 / (1 + d_feat)`
   * 最後 loss：
     `L_jfpd = α * ψ * d_feat + (1 - α) * ϕ * d_pred`
3. 對 target batch 平均後反傳。
   這就是 paper 的核心。論文也明確寫了：feature divergence 用 cosine distance、prediction divergence 用 JS divergence、pseudo-label 用 `argmax`，而 prototype 可以用每次 iteration 動態抽 source 每類樣本來估計。  

---

## 你最小實作時，建議直接這樣定義

### 1. 模型：timm backbone + 自己接 linear classifier

這樣最穩。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class JFPDNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,      # 只取 feature
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)              # [B, D]
        logits = self.classifier(feat)       # [B, C]
        prob = torch.softmax(logits, dim=-1)
        return feat, logits, prob
```

---

## 2. JFPD 所需函數

### cosine feature divergence

論文的 feature divergence 是把距離正規化成 `[0,1)`：
`d_feat = d / (1 + d)`，其中 `d` 可用 cosine distance。 

```python
def cosine_distance(x, y, eps=1e-8):
    # x, y: [B, D]
    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)
    cos = (x * y).sum(dim=-1)          # [B]
    dist = 1.0 - cos                   # cosine distance in [0, 2]
    return dist

def normalized_feature_divergence(ft, zs):
    d = cosine_distance(ft, zs)        # [B]
    return d / (1.0 + d)
```

### JS divergence

論文 prediction divergence 用 probability divergence，再做同樣正規化；文中實作預設是 JS divergence。

```python
def entropy_from_prob(p, eps=1e-8):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)

def kl_div_prob(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)

def js_divergence(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    return 0.5 * kl_div_prob(p, m, eps) + 0.5 * kl_div_prob(q, m, eps)

def normalized_prediction_divergence(pt, ps):
    d = js_divergence(pt, ps)          # [B]
    return d / (1.0 + d)
```

### JFPD loss

`ψ` 來自 source/target prediction entropy，`ϕ` 來自 feature divergence。這是論文的 cross-guided trust 設計：

* `ψ` 去加權 feature discrepancy
* `ϕ` 去加權 prediction discrepancy  

```python
def jfpd_loss(
    ft, pt,               # target features/probs: [B,D], [B,C]
    zs, ps,               # matched source prototypes: [B,D], [B,C]
    alpha=0.5
):
    d_feat = normalized_feature_divergence(ft, zs)   # [B]
    d_pred = normalized_prediction_divergence(pt, ps)  # [B]

    Hs = entropy_from_prob(ps)   # source prototype prediction entropy
    Ht = entropy_from_prob(pt)

    psi = 1.0 / (1.0 + Hs + Ht)        # uncertainty-aware trust
    phi = 1.0 / (1.0 + d_feat)         # semantic-alignment trust

    loss = alpha * psi * d_feat + (1.0 - alpha) * phi * d_pred
    return loss.mean(), {
        "d_feat": d_feat.mean().item(),
        "d_pred": d_pred.mean().item(),
        "psi": psi.mean().item(),
        "phi": phi.mean().item(),
    }
```

---

## 3. source prototype 建法

論文是每類各有：

* feature prototype `z_s^c`
* prediction prototype `p_s^c` 

最小版有兩種：

### 版本 A：最簡單

先把整個 source 跑過一次，存每類平均 prototype。
優點是最省事。缺點是沒有論文說的 dynamic prototype。

### 版本 B：比較貼論文

每個 iteration 對每個 class 隨機抽 `K` 個 source 樣本做 prototype。論文 implementation section 明確提到動態 prototype estimation，而且在多個實驗裡用 **每類 32 個 source samples** 估 prototype。 

我建議你一開始先用 **版本 A 跑通**，之後再改成 B。

---

## 4. prototype 預先計算（最小可跑版）

```python
from collections import defaultdict

@torch.no_grad()
def build_source_prototypes(model, loader, num_classes, device):
    model.eval()
    feat_sum = defaultdict(list)
    prob_sum = defaultdict(list)

    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = batch["label"].to(device)

        feat, _, prob = model(x)
        for c in range(num_classes):
            mask = (y == c)
            if mask.any():
                feat_sum[c].append(feat[mask])
                prob_sum[c].append(prob[mask])

    feat_proto = []
    prob_proto = []
    for c in range(num_classes):
        feats_c = torch.cat(feat_sum[c], dim=0)
        probs_c = torch.cat(prob_sum[c], dim=0)
        feat_proto.append(feats_c.mean(dim=0))
        prob_proto.append(probs_c.mean(dim=0))

    feat_proto = torch.stack(feat_proto, dim=0).to(device)   # [C,D]
    prob_proto = torch.stack(prob_proto, dim=0).to(device)   # [C,C]
    return feat_proto, prob_proto
```

---

## 5. target adaptation loop

論文的 target adaptation 就是：
target sample → feature/prediction → pseudo-label → 對應 source prototype → 算 JFPD loss → update model。 

```python
def adapt_one_epoch(model, target_loader, source_feat_proto, source_prob_proto,
                    optimizer, device, alpha=0.5):
    model.train()
    meter = {"loss": 0.0, "d_feat": 0.0, "d_pred": 0.0, "psi": 0.0, "phi": 0.0}
    n = 0

    for batch in target_loader:
        x_t = batch["pixel_values"].to(device)

        ft, _, pt = model(x_t)                      # [B,D], [B,C]
        pseudo = pt.argmax(dim=-1)                 # [B]

        zs = source_feat_proto[pseudo]             # [B,D]
        ps = source_prob_proto[pseudo]             # [B,C]

        loss, stat = jfpd_loss(ft, pt, zs, ps, alpha=alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x_t.size(0)
        meter["loss"] += loss.item() * bs
        for k in stat:
            meter[k] += stat[k] * bs
        n += bs

    for k in meter:
        meter[k] /= max(n, 1)
    return meter
```

---

## 6. 兩階段完整 pipeline

這就是你要的最小版：

```python
# stage 1: source pretrain
for epoch in range(source_epochs):
    model.train()
    for batch in source_loader:
        x = batch["pixel_values"].to(device)
        y = batch["label"].to(device)

        _, logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# build source prototypes once
source_feat_proto, source_prob_proto = build_source_prototypes(
    model, source_loader, num_classes, device
)

# stage 2: target adaptation with JFPD
for epoch in range(target_epochs):
    stats = adapt_one_epoch(
        model,
        target_loader,
        source_feat_proto,
        source_prob_proto,
        optimizer,
        device,
        alpha=0.5
    )
    print(epoch, stats)
```

---

## 7. Hugging Face datasets / DomainNet 你只要接這一層

你用 `datasets` 時，只要保證 dataloader 最後吐出：

```python
{
    "pixel_values": Tensor[B,3,H,W],
    "label": Tensor[B]   # source 才需要
}
```

如果你的 DomainNet 是單一 dataset + `domain` 欄位，你就先 filter 成：

* `source_ds = ds.filter(lambda x: x["domain"] == source_name)`
* `target_ds = ds.filter(lambda x: x["domain"] == target_name)`

如果你的 HF 版本不是這種欄位設計，也沒關係；原則只是把 source/target 切開，JFPD 本身不依賴 HF 的特定 schema。論文對 DomainNet 的設定是 **345 classes、6 domains**。

### 最小 transform

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def hf_transform(example):
    image = example["image"].convert("RGB")
    example["pixel_values"] = train_tf(image)
    return example
```

---

## 8. 建議你一開始先用的超參數

這些不是我亂猜，是根據文中可見設定壓到最小可用版後的保守選法：

* `alpha = 0.5`
  因為文中 ablation 指出 `α ∈ [0.1, 0.9]` 都算穩，但最好大約在 `0.5~0.6`；`α=1.0`（只靠 prediction）會明顯掉很多。 
* backbone：`vit_base_patch16_224` 或 `resnet50`
* image size：`224`
* source pretrain lr：`1e-4 ~ 5e-4`
* target adapt lr：`5e-5`
* batch size：先 `64` 或 `128`
* prototype：先 **全 source 平均一次**；跑通後再換成 **每類隨機取 32 個 source 樣本** 做 dynamic prototype，比較接近論文。

---

## 9. 你實作時最容易踩的坑

### 不要把 source CE loss 混進最小版

論文 adaptation 核心是 target 上的 JFPD。最小重現先不要混別的 loss，不然你會分不清是 JFPD 有效還是別的 regularizer 在救。

### prototype prediction 要用「機率平均」，不是 logits 平均

論文 prototype 是 `g(f(x))` 的平均，也就是 prediction prototype。比較穩的是先 softmax 再平均。

### `psi` 裡的 source prediction

你現在用的是 class prediction prototype `p_s[c]`，所以 `H(p_s[c])` 是合理的最小替代。

### target early stage 會有錯 pseudo-label

這是正常的，所以 paper 才用 `ψ` 和 `ϕ` 去壓低不可信樣本。

---

## 10. 你可以直接照這個最小版本理解成

**JFPD = target feature 靠近 source class prototype + target prediction 靠近 source class prediction prototype，
但不是每個 target 都一樣重要，而是用 entropy 與 feature closeness 做 trust weighting。**
這就是 paper 的本體。 

你要的話，我下一則可以直接幫你補成一個 **單檔 `train_jfpd.py`**，用 `timm + datasets + torch` 可直接改 domain pair 跑。
