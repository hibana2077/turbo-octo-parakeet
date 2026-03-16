Yes — the best way is to **unit-test the JFPD math before training**.

For this paper, the key things to verify independently are:

1. `d_feat = d / (1 + d)`
2. `d_pred = D / (1 + D)`
3. `ψ = 1 / (1 + H(ps) + H(pt))`
4. `ϕ = 1 / (1 + d_feat)`
5. `L_JFPD = α ψ d_feat + (1-α) ϕ d_pred`
6. pseudo-label retrieval uses `argmax(pt)` and then selects the matching source prototypes `(z_s^ŷ, p_s^ŷ)` for that class. The paper also states the default implementation uses cosine distance for feature divergence and JS divergence for prediction divergence.    

## 1) One exact hand-check example

Use this exact test case:

* `ft = [1.0, 0.0]`
* `zs = [0.6, 0.8]`
* `pt = [0.7, 0.2, 0.1]`
* `ps = [0.8, 0.1, 0.1]`
* `alpha = 0.5`

Using **cosine distance** and **JS divergence**, you should get:

* cosine similarity = `0.6`
* cosine distance = `1 - 0.6 = 0.4`
* `d_feat = 0.4 / 1.4 = 0.2857142857`
* `ϕ = 1 / (1 + 0.2857142857) = 0.7777777778`

Using natural log:

* `H(ps) = 0.6390318597`
* `H(pt) = 0.8018185525`
* `ψ = 1 / (1 + H(ps) + H(pt)) = 0.4096932753`

For prediction divergence:

* `JS(pt, ps) = 0.0101628553`
* `d_pred = 0.0101628553 / (1 + 0.0101628553) = 0.0100606107`

Final loss:

* `L = 0.5 * ψ * d_feat + 0.5 * ϕ * d_pred`
* `L = 0.0624400705`

So if your code is correct, it should reproduce those numbers up to small floating-point tolerance. These steps directly match the paper’s definitions of normalized feature divergence, normalized prediction divergence, uncertainty-aware trust, semantic-alignment trust, and the final JFPD loss.   

```python
import torch

# assume you already defined:
# cosine_distance
# normalized_feature_divergence
# js_divergence
# normalized_prediction_divergence
# entropy_from_prob
# jfpd_loss

ft = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
zs = torch.tensor([[0.6, 0.8]], dtype=torch.float64)

pt = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
ps = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)

loss, stat = jfpd_loss(ft, pt, zs, ps, alpha=0.5)

print("d_feat =", stat["d_feat"])
print("d_pred =", stat["d_pred"])
print("psi    =", stat["psi"])
print("phi    =", stat["phi"])
print("loss   =", loss.item())

assert abs(stat["d_feat"] - 0.2857142857) < 1e-6
assert abs(stat["d_pred"] - 0.0100606107) < 1e-6
assert abs(stat["psi"]    - 0.4096932753) < 1e-6
assert abs(stat["phi"]    - 0.7777777778) < 1e-6
assert abs(loss.item()    - 0.0624400705) < 1e-6

print("value check passed")
```

---

## 2) Three sanity checks that should always hold

These are even more useful than a single scalar test.

### Check A: perfect match gives zero loss

If `ft == zs` and `pt == ps`, then both divergences are zero, so the final JFPD loss must be zero. That follows immediately from the paper’s equations.  

```python
ft = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
zs = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
pt = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)
ps = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)

loss, stat = jfpd_loss(ft, pt, zs, ps, alpha=0.5)
assert abs(stat["d_feat"]) < 1e-12
assert abs(stat["d_pred"]) < 1e-12
assert abs(loss.item()) < 1e-12
```

### Check B: higher entropy should reduce `ψ`

Because `ψ = 1 / (1 + H(ps) + H(pt))`, more uncertain predictions must give smaller trust.  

```python
conf = torch.tensor([[0.98, 0.01, 0.01]], dtype=torch.float64)
unif = torch.tensor([[1/3, 1/3, 1/3]], dtype=torch.float64)

psi_conf = 1.0 / (1.0 + entropy_from_prob(conf) + entropy_from_prob(conf))
psi_unif = 1.0 / (1.0 + entropy_from_prob(unif) + entropy_from_prob(unif))

assert psi_conf.item() > psi_unif.item()
```

### Check C: larger feature mismatch should reduce `ϕ`

Because `ϕ = 1 / (1 + d_feat)`, more feature discrepancy means smaller semantic trust. 

```python
ft1 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
zs1 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)   # identical

ft2 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
zs2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)   # orthogonal

d1 = normalized_feature_divergence(ft1, zs1)
d2 = normalized_feature_divergence(ft2, zs2)

phi1 = 1.0 / (1.0 + d1)
phi2 = 1.0 / (1.0 + d2)

assert d1.item() < d2.item()
assert phi1.item() > phi2.item()
```

---

## 3) Prototype indexing check

A very common bug is not the math, but selecting the **wrong prototype** after pseudo-labeling. During adaptation, the paper says you assign `ŷ_t = argmax(pt)` and retrieve the corresponding source feature/prediction prototypes for that class. 

So test that explicitly:

```python
source_feat_proto = torch.tensor([
    [1.0, 0.0],   # class 0
    [0.0, 1.0],   # class 1
    [1.0, 1.0],   # class 2
], dtype=torch.float64)

source_prob_proto = torch.tensor([
    [0.9, 0.05, 0.05],  # class 0
    [0.1, 0.8, 0.1],    # class 1
    [0.1, 0.2, 0.7],    # class 2
], dtype=torch.float64)

pt = torch.tensor([
    [0.05, 0.90, 0.05],  # pseudo-label 1
    [0.20, 0.20, 0.60],  # pseudo-label 2
], dtype=torch.float64)

pseudo = pt.argmax(dim=-1)
zs = source_feat_proto[pseudo]
ps = source_prob_proto[pseudo]

assert pseudo.tolist() == [1, 2]
assert torch.allclose(zs, torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float64))
assert torch.allclose(ps, torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float64))
```

If this fails, your pipeline is wrong even if the formulas are correct.

---

## 4) What usually goes wrong in practice

The most common implementation mistakes are:

* using **logits** instead of **probabilities** in entropy / JS divergence
* averaging source **logits** rather than averaging source **softmax probabilities** for `p_s^c`
* computing `ϕ` from raw cosine distance instead of the paper’s normalized `d_feat`
* mixing up `alpha` branches, since the paper uses `α ψ d_feat + (1-α) ϕ d_pred`
* prototype indexing off by one or misaligned class ids after dataset filtering. The source prediction prototype is defined as the average of `g(f(x_s))`, and the loss uses source prototypes matched by target label or pseudo-label.  

---

## 5) The safest debugging order

Do it in this order:

1. verify the single-sample exact numeric test above
2. verify the three invariants
3. verify pseudo-label → prototype lookup
4. run one tiny batch and print all intermediate tensors
5. only then start adaptation training

If you want, paste your current `jfpd_loss()` and prototype code, and I’ll check it line by line against the paper’s equations.
