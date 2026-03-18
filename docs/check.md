這組 log 很不正常，**比較像 pipeline / implementation bug，不像單純沒 tune 好**。

你的現象是：

* source pretrain 幾乎滿分：`accuracy=0.9993`
* adaptation loss 很快掉到接近 0：`0.0939 -> 0.0002`
* 但 target accuracy 不只沒上升，還掉到 `8.85% -> 4.17% -> 2.08% -> 4.69%`

這代表模型很可能正在**非常成功地優化一個錯的目標**，而不是在做有效的 domain adaptation。這和你上傳的檢查筆記裡的判斷一致：這種「loss 很漂亮、accuracy 很差」最常見是 prototype 來源、feature/prediction 空間、divergence/trust 公式，或 evaluation label mapping 出錯。

我會這樣判斷優先順序。

## 最可能的 bug 1：adaptation 時 prototype 不是純 source prototype

JFPD 的 paper 很明確：
target sample 先用 pseudo-label 決定類別，然後**只去取對應的 source prototypes** `(z^s_{ŷ_t}, p^s_{ŷ_t})`。paper 的 Algorithm 1 也是這個流程。

也就是說，adaptation 應該是：

1. target forward
2. 算 pseudo-label
3. 用 pseudo-label 去選 **source** feature prototype / **source** prediction prototype
4. 算 discrepancy

不是：

* 用 target batch 自己做 prototype
* source / target 混在一起做 prototype
* 建 target pseudo-prototype bank 再去對齊

你的 log 非常像這種錯：如果 target 跟 target 自己比，或 target prototype 被混進去，`d_feat` 和 `d_pred` 會很快接近 0，但分類能力不會真的變好。這點也在你上傳的 implementation guide 裡被特別強調。 

**你現在最該先印的 debug log：**

```python
print("prototype_domain", prototype_domain)  # 應該永遠是 source
print("target_used_in_proto", target_used_in_proto)  # 應該永遠 False
print("proto_labels_used", proto_labels_used[:10])   # 應該對應 pseudo-label
print("source_proto_count_per_class", counts)
```

## 最可能的 bug 2：feature prototype 和 prediction prototype 用錯空間

paper 的定義是：

* feature prototype = `f(x)` 的平均
* prediction prototype = `g(f(x))` 的平均，也就是 **softmax probability** 的平均，不是 logits。 

如果你用 timm/ViT 時不小心：

* 把 logits 當 feature
* `forward_features()` 之後沒做對應的 pooling / pre_logits
* feature prototype 和 prediction prototype 其實都來自同一個 tensor

那 JFPD 會直接失真。loss 可以很好看，但任務早就不是 paper 那個 objective 了。這也正是你上傳的 check.md 裡列的高風險錯誤。

**立刻檢查：**

```python
print(z_s.shape)   # 應該是 [num_classes, feature_dim]
print(p_s.shape)   # 應該是 [num_classes, num_classes]
print(feature_dim, num_classes)  # 兩者不應相等（通常）
print(p_t.sum(dim=-1)[:5])       # 應該接近 1
print(p_s.sum(dim=-1)[:5])       # 應該接近 1
```

如果 `p_s` 不是機率分布，`d_pred` 幾乎一定會錯。

## 最可能的 bug 3：divergence / trust weight 公式實作錯

paper 明確寫的是：

* `d_feat = d / (1 + d)`
* `d_pred = D / (1 + D)`
* `psi = 1 / (1 + H(p_t) + H(p^s_{ŷ_t}))`
* `phi = 1 / (1 + d_feat(...))`  

常見錯法是：

* 用 `cosine_similarity` 當 distance
* 忘了做 normalize：`d/(1+d)`
* `d_pred` 拿 KL / CE 亂代，甚至直接對 logits 算
* `phi` 寫成 `1 + d_feat`
* `psi` 只用 target entropy，沒跟 source prediction prototype 對應

你的 log 裡：

* `d_feat` 很快到 `0.0012`
* `d_pred` 幾乎 `0.0001`
* `phi` 卻幾乎 `0.999`

這種速度太可疑了。正常情況下，target accuracy 還在 2%～5% 時，不太可能 feature/prediction discrepancy 已經幾乎完美對齊。這也和你的筆記中的判斷一致。

## 也很值得查：evaluation label mapping 或單一類別崩塌

你這個 accuracy 水平很像：

* 幾乎全部預測成某一類
* 或 source / target / test 的 `class_to_idx` 不一致

你的檢查筆記也提到這點。

**立刻印：**

```python
# target test
print("pred hist:", torch.bincount(preds, minlength=num_classes))
print("label hist:", torch.bincount(labels, minlength=num_classes))
print("mean max prob:", probs.max(dim=1).values.mean().item())
print("class_to_idx source:", source_dataset.class_to_idx)
print("class_to_idx target_train:", target_train_dataset.class_to_idx)
print("class_to_idx target_test:", target_test_dataset.class_to_idx)
```

如果你看到：

* `pred hist` 幾乎集中在單一類 => model collapse
* confusion matrix 像 permutation => label mapping bug

## 設定本身也偏離 paper，但這不是 4% 的主因

你的 command 用的是 `vit_tiny_patch16_224...`，而 paper 在 Office-Home 用的是 ImageNet-pretrained ResNet-34 / ResNet-50 / ViT-B/32 / ViT-B/16；source pretraining 100 epochs，adapt 50 epochs，batch size 128，prototype 每類 32 個 source samples。 

所以你現在的設定確實和 paper 不一致。這可能讓結果變差，但**不太會讓它壞到 2%～8% 這種程度**。這種程度還是更像程式錯。

## 我建議你先做的最小 debug 順序

先不要大改整個系統，先做這四件：

1. **把 prototype 固定成 source-only**

   * 暫時不要 dynamic target-related 東西
   * 先確認 target 完全不參與 prototype 建構

2. **確認兩種 prototype 的空間**

   * `z_s_c = mean(feature)`
   * `p_s_c = mean(softmax(logits))`

3. **把 divergence / trust 單元測試化**

   * 人工餵兩個相同向量，distance 應接近 0
   * 人工餵兩個不同 one-hot 分布，JS divergence 應大於 0
   * 檢查 `phi = 1/(1+d_feat)`、`psi = 1/(1+H(pt)+H(ps_proto))`

4. **檢查 evaluation mapping**

   * `class_to_idx`
   * prediction histogram
   * confusion matrix

## 一個很實用的 smoke test

在正式 adapt 前，先做一個極小測試：

* 抽 8 個 target samples
* 抽每類 2 個 source samples 建 prototype
* 手動印出每個 sample 的：

  * pseudo-label
  * selected source class
  * `d_feat`
  * `d_pred`
  * `psi`
  * `phi`
  * final loss

像這樣：

```python
for i in range(8):
    print({
        "pseudo": yhat[i].item(),
        "d_feat": d_feat[i].item(),
        "d_pred": d_pred[i].item(),
        "psi": psi[i].item(),
        "phi": phi[i].item(),
        "loss": loss_i[i].item(),
    })
```

如果你看到一開始就幾乎全部 `d_feat≈0`、`d_pred≈0`，那幾乎可以直接確定 implementation 有問題。

## 我的結論

單看這份 log，我的排序是：

1. **prototype 來源錯**
2. **feature / prediction prototype 空間用錯**
3. **divergence / trust 公式錯**
4. **evaluation label mapping 或 collapse**
5. 設定偏離 paper

其中第 1 點最可疑，因為 paper 明確要求 adaptation 時 target sample 只去對齊 **source prototypes**，而不是 target prototypes。 

把 `compute_prototypes()`、`compute_jfpd_loss()`、還有 ViT 的 `forward_features / head` 那段貼出來，我可以直接幫你逐段抓。
