對，**這已經是很明顯的 collapse**，而且是 **single-class prediction collapse**。

你這段 log 幾乎可以直接下結論：

* `epoch 1`

  * `dominant_pred_ratio=0.8385`
  * 192 個 target test sample 裡，有 **161 個都被預測成 class 0 / Backpack**
* `epoch 2`

  * `dominant_pred_ratio=1.0000`
  * `pred_hist_top=[{'class_id': 0, 'count': 192, 'label': 'Backpack'}]`

這代表在 `epoch 2` 時，**整個 target test set 全部被預測成同一類**。
所以答案是：**是，模型發生了 collapse。**

不過更精確地說，這是：

* **classifier / output collapse**
* **pseudo-label collapse**

不一定已經能直接斷定是 **feature collapse**。
因為 feature collapse 通常還要看：

* feature variance 是否趨近 0
* 不同 sample 的 embedding 是否幾乎一樣
* prototype distance 是否全面消失

但就分類輸出來看，已經是 collapse 了。

---

## 你這個 collapse 很有代表性

從 log 看起來是這樣的動態：

### epoch 1：已經嚴重偏向 class 0

* `mean_max_prob=0.8005`
* `dominant_pred_ratio=0.8385`

這表示模型不是隨機亂猜，而是**很有信心地把大多數 target sample 都壓到 class 0**。

### epoch 2：完全崩成單一類

* `mean_max_prob=0.9297`
* `dominant_pred_ratio=1.0000`

這更糟，因為代表模型不只是 collapse，還是**高信心 collapse**。
也就是它不是不確定，而是非常自信地全猜錯方向。

---

## 這通常代表什麼

這通常不是單純「adaptation 沒有效」，而是：

### 1. pseudo-label feedback loop 爆掉了

如果 adaptation 用 target pseudo-label，而 early stage 幾乎都被分到 class 0：

* target sample → pseudo-label = 0
* 全部去對齊 source prototype of class 0
* 參數更新後更傾向 class 0
* 下一輪更多 sample 被預測成 class 0
* 最後全體 collapse 到 class 0

你的 log 很符合這個模式。

而且這一行很可疑：

* `used_samples=776, skipped_samples=0`

也就是說你**沒有濾掉低品質 pseudo-label**，全部 target sample 都被拿去更新。
如果一開始 pseudo-label 就偏掉，整個 adaptation 會快速自我強化到崩潰。

---

## 2. 你的 adaptation objective 可能在鼓勵 collapse

如果 loss 只要求 target sample 靠近某個 prototype，但沒有足夠的：

* 類別平衡約束
* entropy regularization 設計正確
* pseudo-label quality control
* source supervision / anchor 保護

那最簡單的最小化方式，就是把所有 target sample 都推向一個 prototype。

這時會出現你看到的現象：

* `d_feat` 下降
* `d_pred` 下降
* adaptation loss 很漂亮
* target accuracy 崩掉

---

## 3. class 0 可能在你的 pipeline 裡被「特別偏好」

這可能來自：

* pseudo-label `argmax` 前 logits / softmax 有偏置
* prototype indexing 錯，很多 sample 都拿到 class 0 prototype
* label mapping 錯
* class 0 的 prototype norm / logit bias 異常大
* 某個 broadcast bug 導致每筆都對到第 0 類

這種 bug 很常見，而且會造成你現在這種「全猜 Backpack」。

---

## 你下一步最該檢查的，不是 test，而是 adapt 當下

請直接在 adaptation loop 裡印這 4 個東西。

### A. pseudo-label 分布

```python
pseudo = probs_t.argmax(dim=1)
print("pseudo hist:", torch.bincount(pseudo, minlength=num_classes).cpu().tolist())
```

如果這裡也迅速變成幾乎全是 0，那就證明是 **pseudo-label collapse**，不是單純 test-time 問題。

---

### B. 被選到的 source prototype 類別分布

```python
selected_cls = pseudo  # 如果你是用 pseudo-label 選 prototype
print("selected proto hist:", torch.bincount(selected_cls, minlength=num_classes).cpu().tolist())
```

如果這裡全是 0，代表 adaptation 的對齊目標已經只剩 class 0。

---

### C. source model 在 target 上、adapt 前的預測分布

在 **還沒開始 adapt 前**，先測一次 target：

```python
model.eval()
# target inference
print("pre-adapt pred hist:", ...)
print("pre-adapt mean max prob:", ...)
```

這可以分辨：

* 如果 pre-adapt 就已經大幅偏向 class 0，問題可能來自 source model / label mapping / classifier bias
* 如果 pre-adapt 還正常，adapt 後才崩，問題更可能在 adaptation loss / pseudo-label pipeline

---

### D. class 0 的 logit / prototype 是否異常

```python
print("classifier bias:", model.head.bias.data)
print("classifier weight norm:", model.head.weight.data.norm(dim=1))
print("source proto feature norm:", z_s.norm(dim=1))
print("source proto pred[0:3]:", p_s[:3])
```

如果 class 0 明顯特別大，就很可疑。

---

## 目前我對你的判斷

我會這樣下結論：

* **是，模型已經 collapse**
* 更精確是 **single-class output collapse**
* 很可能是 **pseudo-label collapse 導致的自我強化**
* 而且這很可能不是 dataset 本身問題，而是 **adaptation pipeline / prototype selection / loss design** 的問題

---

## 最值得先查的三個點

優先順序我會排成：

### 第一：adaptation 時 pseudo-label 是否快速全變成 0

這最直接。

### 第二：prototype 是否真的用對了

是否每個 target sample 是拿 `pseudo-label 對應的 source prototype`，而不是：

* 拿錯 index
* broadcast 成 class 0
* target prototype 混進去
* 某個 tensor shape 導致全部對到第一列

### 第三：是否缺少 confidence filtering

你現在 `skipped_samples=0` 很危險。
如果 early pseudo-label 品質差，全部拿來訓練非常容易 collapse。

---

一句話總結：

**對，這不是普通表現差，而是明確的單一類別崩塌。**

把你 adaptation loop 裡「pseudo-label 產生 → prototype 選取 → loss 計算」那段貼出來，我可以直接幫你找是哪一行把它推成 class 0。
