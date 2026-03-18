#!/usr/bin/env python3

import torch

from train_jfpd import entropy_from_prob, jfpd_loss, normalized_feature_divergence


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-6) -> None:
    if abs(actual - expected) >= tol:
        raise AssertionError(f"{name} mismatch: got {actual:.10f}, expected {expected:.10f}")


def run_exact_numeric_check() -> None:
    ft = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    zs = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    pt = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
    ps = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)

    loss, stat = jfpd_loss(ft, pt, zs, ps, alpha=0.5)

    assert_close("d_feat", stat["d_feat"], 0.2857142857)
    assert_close("d_pred", stat["d_pred"], 0.0100606107)
    assert_close("psi", stat["psi"], 0.4096932753)
    assert_close("phi", stat["phi"], 0.7777777778)
    assert_close("loss", loss.item(), 0.0624400705)

    print("exact numeric check passed")


def run_loss_mode_check() -> None:
    ft = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    zs = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    pt = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
    ps = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)

    fgpd_loss, fgpd_stat = jfpd_loss(ft, pt, zs, ps, alpha=0.5, mode="fgpd")
    pgfd_loss, pgfd_stat = jfpd_loss(ft, pt, zs, ps, alpha=0.5, mode="pgfd")

    assert_close("fgpd loss", fgpd_loss.item(), fgpd_stat["phi"] * fgpd_stat["d_pred"])
    assert_close("pgfd loss", pgfd_loss.item(), pgfd_stat["psi"] * pgfd_stat["d_feat"])

    print("loss mode check passed")


def run_perfect_match_check() -> None:
    ft = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    zs = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    pt = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)
    ps = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)

    loss, stat = jfpd_loss(ft, pt, zs, ps, alpha=0.5)

    assert_close("perfect d_feat", stat["d_feat"], 0.0, tol=1e-12)
    assert_close("perfect d_pred", stat["d_pred"], 0.0, tol=1e-12)
    assert_close("perfect loss", loss.item(), 0.0, tol=1e-12)

    print("perfect match check passed")


def run_entropy_trust_check() -> None:
    conf = torch.tensor([[0.98, 0.01, 0.01]], dtype=torch.float64)
    unif = torch.tensor([[1 / 3, 1 / 3, 1 / 3]], dtype=torch.float64)

    psi_conf = 1.0 / (1.0 + entropy_from_prob(conf) + entropy_from_prob(conf))
    psi_unif = 1.0 / (1.0 + entropy_from_prob(unif) + entropy_from_prob(unif))

    if not (psi_conf.item() > psi_unif.item()):
        raise AssertionError(
            f"entropy trust check failed: confident psi={psi_conf.item():.10f}, uniform psi={psi_unif.item():.10f}"
        )

    print("entropy trust check passed")


def run_feature_trust_check() -> None:
    ft1 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    zs1 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    ft2 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    zs2 = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

    d1 = normalized_feature_divergence(ft1, zs1)
    d2 = normalized_feature_divergence(ft2, zs2)
    phi1 = 1.0 / (1.0 + d1)
    phi2 = 1.0 / (1.0 + d2)

    if not (d1.item() < d2.item()):
        raise AssertionError(f"feature divergence ordering failed: d1={d1.item():.10f}, d2={d2.item():.10f}")
    if not (phi1.item() > phi2.item()):
        raise AssertionError(f"feature trust ordering failed: phi1={phi1.item():.10f}, phi2={phi2.item():.10f}")

    print("feature trust check passed")


def run_prototype_indexing_check() -> None:
    source_feat_proto = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    source_prob_proto = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=torch.float64,
    )
    pt = torch.tensor(
        [
            [0.05, 0.90, 0.05],
            [0.20, 0.20, 0.60],
        ],
        dtype=torch.float64,
    )

    pseudo = pt.argmax(dim=-1)
    zs = source_feat_proto[pseudo]
    ps = source_prob_proto[pseudo]

    expected_zs = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    expected_ps = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float64)

    if pseudo.tolist() != [1, 2]:
        raise AssertionError(f"prototype pseudo-label check failed: got {pseudo.tolist()}")
    if not torch.allclose(zs, expected_zs):
        raise AssertionError(f"feature prototype lookup failed: got {zs.tolist()}")
    if not torch.allclose(ps, expected_ps):
        raise AssertionError(f"prediction prototype lookup failed: got {ps.tolist()}")

    print("prototype indexing check passed")


def main() -> None:
    run_exact_numeric_check()
    run_loss_mode_check()
    run_perfect_match_check()
    run_entropy_trust_check()
    run_feature_trust_check()
    run_prototype_indexing_check()
    print("all JFPD checks passed")


if __name__ == "__main__":
    main()
