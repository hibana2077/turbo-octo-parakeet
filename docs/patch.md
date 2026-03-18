對，現在可以更明確地下結論：

## 結論

這不是單純「adapt 沒有效」，而是 **pseudo-label driven collapse**。
而且從你補的 A–D 來看，**最可疑的不是 classifier 本身，也不是 evaluation mapping，而是 adaptation loop 的自我強化機制**。

你這四點的含義很清楚：

* **A. pseudo hist 往 0 類集中**

  * 代表 target pseudo-label 一開始就偏向 class 0
* **B. selected proto hist 也往 0 類集中**

  * 代表 prototype selection 本身是在「忠實跟隨錯的 pseudo-label」
* **C. pre-adapt acc = 0.29，但 adapt 後掉到 0.078 / 0.041**

  * 代表 source model 本來還有一定可用性
  * adaptation 正在**破壞**模型，而不是修正模型
* **D. classifier bias / norm / prototype norm 都正常**

  * 代表不是「class 0 權重特別大」這種很表層的 bug

所以這件事的核心機制是：

> 初始 target 預測有偏 → pseudo-label 多數變成 0 → 全部去對齊 source class-0 prototype → 模型更偏向 0 → 下一輪更多 sample 被標成 0 → 最後全 collapse 到 class 0。

這是一個非常典型的 **self-training positive feedback failure**。

---

## 這表示哪裡有問題？

目前最像的是下面三種之一，而且我會把第 1 個排最前面。

### 1. 你沒有做有效的 pseudo-label quality control

你前面的 log 一直是：

```python
used_samples=776, skipped_samples=0
```

這很危險。
如果 early pseudo-label 品質不夠好，但你把 **全部 target samples** 都拿去更新，collapse 非常合理。

也就是說，你的 pipeline 現在很像：

```python
pseudo = argmax(p_t)
loss = align(target, source_proto[pseudo])
backward()
```

但少了這類保護：

* confidence threshold
* entropy threshold
* class-balance constraint
* source anchor
* teacher / EMA stabilization

這時最容易出現「全部被一個錯誤類別吸過去」。

---

### 2. 你的 adaptation objective 對「全對齊同一類」沒有任何阻力

即使 implementation 沒有 tensor bug，objective 也可能有漏洞。

如果 loss 只有這種吸引力：

* target sample 靠近 pseudo-label 對應的 source prototype
* discrepancy 越小越好

那麼對 optimizer 來說，一個很容易的解就是：

* 把大多數 target sample 都推向同一個 prototype

這時你會看到：

* `loss ↓`
* `d_feat ↓`
* `d_pred ↓`
* 但 `accuracy 也 ↓`

因為它在優化的是「prototype agreement」，不是「正確分類」。

---

### 3. 你更新了不該 aggressively 更新的部分

如果 adaptation 時你把整個模型都放開更新，尤其是 classifier head 也一起大步更新，target-only loss 很容易造成：

* classifier semantics drift
* catastrophic forgetting
* pseudo-label 更快崩掉

你現在 `pre-adapt=0.29`，代表 source classifier 原本還能用。
adapt 後立刻掉到 0.07，這很像 adaptation 在毀掉原本的 decision boundary。

---

## 現在最合理的判斷

所以答案不是：

* dataset 壞了
* 測試有問題
* class mapping 壞了
* class 0 權重天生特別大

而是：

> **你的 adaptation pipeline 缺乏抑制 pseudo-label collapse 的機制，導致 target-only 更新把原本還能用的 source model 拉爛。**

---

## 我建議你立刻做的 4 個修正

### 修正 1：加 confidence filtering

先做最直接的版本：

```python
probs_t = torch.softmax(logits_t, dim=-1)
conf, pseudo = probs_t.max(dim=-1)

mask = conf >= 0.95
# 或先試 0.9 / 0.95 / 0.99
```

然後只用 high-confidence samples：

```python
if mask.sum() == 0:
    continue

z_t_used = z_t[mask]
p_t_used = probs_t[mask]
pseudo_used = pseudo[mask]
```

你的 `skipped_samples=0` 現在很不正常。
在 domain shift 下，不可能每一筆 pseudo-label 都值得信。

---

### 修正 2：先 freeze classifier head

先做保守版 ablation：

* **只更新 backbone / feature extractor**
* **classifier head 固定不動**

如果這樣 collapse 明顯減輕，就表示你之前是在用 target-only noisy signal 破壞分類器。

概念上像這樣：

```python
for p in model.head.parameters():
    p.requires_grad = False
```

如果你用的是 timm ViT，確認你 freeze 的真的是分類 head 那層。

---

### 修正 3：加 source anchor loss 防止 forgetting

adaptation 時，不要只靠 target pseudo-label loss。
每一步混入一小批 source：

```python
loss = loss_adapt + lambda_src * ce(model(x_s), y_s)
```

例如先試：

* `lambda_src = 0.1`
* 或 `0.2`

如果加了 source anchor 後，不再快速 collapse，幾乎可以確定你原本是 catastrophic forgetting。

---

### 修正 4：限制每類被選中的 target 樣本數

就算用 confidence filtering，也可能還是大多數都落在 class 0。
所以再加一層 class-balanced cap。

例如：

* 每個 batch / epoch
* 每個 pseudo class 最多取前 `k` 個高信心樣本

這會強迫 adaptation 不要被單一類壟斷。

---

## 最值得做的 3 個 isolation experiments

這三個實驗能很快告訴你是「objective 問題」還是「程式 bug」。

### 實驗 A：只加 confidence threshold

其他不動，只改：

```python
mask = conf >= 0.95
```

看：

* `used_samples`
* `pseudo hist`
* `target acc`

如果 collapse 大幅減輕，根因就是 noisy pseudo-label 被全量拿去更新。

---

### 實驗 B：freeze head

其他不動，只 freeze head。
如果 acc 不再急速掉爛，代表 adaptation 在破壞 classifier。

---

### 實驗 C：adapt 1 step 就 evaluate

不是等一整個 epoch，直接做：

* pre-adapt evaluate
* 1 個 optimizer step 後 evaluate
* 5 steps 後 evaluate

如果只要 1–5 steps 就開始全面偏向 class 0，那問題就在單步梯度方向，不是 long-run accumulation。

---

## 你現在還要補印兩個 log

### 1. batch-average class distribution

這很重要：

```python
mean_probs = probs_t.mean(dim=0)
print("batch mean probs:", mean_probs.cpu().tolist())
print("batch mean entropy:", -(mean_probs * (mean_probs + 1e-8).log()).sum().item())
```

如果 batch mean 也快速集中到 class 0，代表整體 prediction diversity 正在消失。

---

### 2. adaptation 前幾步的 gradient / update 強度

尤其是 head 和 backbone：

```python
for name, p in model.named_parameters():
    if p.grad is not None and ("head" in name or "fc" in name):
        print(name, p.grad.norm().item())
```

如果 head 的梯度特別大，很可能就是它在被 target-only noisy loss 快速拉壞。

---

## 我現在對 root cause 的排序

根據你目前的 A–D，我會這樣排：

1. **沒有 confidence filtering，全部 pseudo-label 都被用來更新**
2. **adaptation objective 缺少 anti-collapse 機制**
3. **classifier head 被 target-only noisy signal 破壞**
4. prototype selection 雖然可能還有細節 bug，但目前看起來它是在跟隨 pseudo-label，不像是主因
5. evaluation / label mapping 問題目前反而不像

---

## 一句話總結

**是，模型 collapse 了；而且從你現在的證據看，主因不是 class 0 權重異常，也不是 test bug，而是 target pseudo-label 在 adaptation 中形成了自我強化回路。**

你下一步最有效的順序是：

1. 加 `confidence threshold`
2. freeze head
3. 加 source anchor loss
4. 做 `1-step / 5-step` evaluation

把你的 `adapt_step()` 或 `compute_jfpd_loss()` 那段貼出來，我可以直接幫你改成一個不那麼容易 collapse 的版本。
