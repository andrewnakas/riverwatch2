#!/usr/bin/env python3
"""LEDGER 55 TRACK 2 — forcing dropout for the NH AR-LSTM members.

WHY. From the frozen record (benchmarks/l54_FINAL_8member_test1.json), per-member
leave-one-out inside each block:

  MB-LSTM x5 (differ by FORCING)      LOO NOT ordered by solo skill
  AR-LSTM x3 (differ by HYPERPARAM)   LOO STRICTLY ordered by solo skill,
                                      on BOTH frames; on val1 nhar is -0.000055
                                      and nhar0 is -0.000009

The AR block carries skill but no distinctness. Making its members differ by
FORCING is the axis measured to produce LOO — but the literal version failed in
LEDGER 54 (single-forcing AR members lost -0.026 solo, vs the MB family's
-0.008). Forcing dropout is the middle path: keep all 15 channels and the
multi-forcing skill, but with probability p mask ONE product's 5 channels so no
seed can lean on one product.

HOW. With probability p per training sample, one randomly chosen forcing
product's columns are set to EXACTLY ZERO (post-normalisation = the channel
mean). Never NaN: NaN is dropped by _validate_samples and poisons NH members.

WHERE. `ARLSTM.forward` is wrapped, and the mask is applied to the RAW
`data["x_d"]` before `self.embedding_net(data)` runs. Two reasons:
  * NH keeps `data["x_d"]` as a dict KEYED BY FEATURE NAME (inputlayer.py:234
    concatenates `data["x_d"][k]` over the feature list), so the mask selects
    tensors BY NAME. It therefore cannot touch the autoregressive input
    `q_mm_shift1` by construction -- not by an index assumption;
  * `num_workers: 2` means anything RNG-driven inside the Dataset runs in worker
    processes, where per-worker seeding silently duplicates mask streams. Here it
    runs once, in the main process, on the batch the model actually consumes.

THE FUSED FAST PATH IS UNAFFECTED, and this is load-bearing (the queue guard
asserts `fused_batches > step_batches`). Dispatch in nh_arlstm_fast.py is on
`_ar_has_nan`, which inspects ONLY the trailing autoregressive column. This mask
touches forcing channels by name, never the AR column, and writes exact zeros via
torch.where, so it cannot manufacture a NaN anywhere. With missing_fraction 0.0
and complete 1999-2008 discharge every training batch still takes _fused_loop.

Masking is gated on `self.training`, so in-training validation is byte-identical
to the control (and still takes the step path on the 1980-89 window, which is why
that guard is `fused > step` and not `step == 0`).
"""
import json
from typing import Dict, List

_S: Dict = {
    "p": 0.0, "n_groups": 0, "F": 0, "M": None, "ones": None, "gen": None,
    "gen_seed": None, "groups": None, "group_names": None,
    "seen": 0, "masked": 0, "by_group": None, "probe": None, "epochs_logged": 0,
    "installed": False,
}


def build_groups(cfg) -> (List[List[str]], List[str]):
    """Group dynamic-input FEATURE NAMES by forcing suffix, from the config's own
    list. The autoregressive input is not in `dynamic_inputs`, so it can never
    land in a group."""
    dyn = list(cfg.dynamic_inputs)
    order, by_name = [], {}
    for i, name in enumerate(dyn):
        if "_" not in name:
            raise AssertionError(
                f"dynamic input {name!r} has no forcing suffix — forcing dropout needs a "
                "multi-forcing dataset (nh_data_multi), not a single-forcing one")
        suffix = name.rsplit("_", 1)[1]
        if suffix not in by_name:
            by_name[suffix] = []
            order.append(suffix)
        by_name[suffix].append(name)
    groups = [by_name[s] for s in order]
    sizes = {len(g) for g in groups}
    assert len(groups) >= 2, f"need >=2 forcing products, found {order}"
    assert sizes == {5}, f"expected 5 vars per product, got sizes {sizes} for {order}"
    flat = sorted(n for g in groups for n in g)
    assert flat == sorted(dyn), "groups do not tile the dynamic inputs"
    return groups, order


def patch(cfg, p: float, device) -> None:
    """Install the wrapper. Refuses p <= 0 so the control is literally the
    untouched code path and can never be confused with a silent no-op."""
    import torch
    from neuralhydrology.modelzoo import arlstm

    assert p > 0, ("forcing dropout refuses p <= 0: the control is the unpatched "
                   "trainer (scripts/nh_arlstm_train.py), not p=0 here")
    assert not _S["installed"], "patch() called twice"

    groups, names = build_groups(cfg)
    n_dyn = len(cfg.dynamic_inputs)
    n_ar = len(cfg.autoregressive_inputs or [])
    F = n_dyn + n_ar

    gen_seed = 0x5D0000 + int(cfg.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(gen_seed)

    _S.update(p=float(p), n_groups=len(groups), F=F, gen=gen, gen_seed=gen_seed,
              groups=groups, group_names=names, by_group=[0] * len(groups),
              installed=True)

    inner = arlstm.ARLSTM.forward           # already nh_arlstm_fast._forward_fast

    def _forward_fd(self, data, h_0=None, c_0=None):
        if self.training and _S["p"] > 0:
            xd = data["x_d"]
            assert isinstance(xd, dict), (
                f"x_d is {type(xd).__name__}, expected a dict keyed by feature name "
                "(neuralhydrology/modelzoo/inputlayer.py:234) — refusing to mask blind")
            missing = [n for g in _S["groups"] for n in g if n not in xd]
            assert not missing, (
                f"dynamic inputs absent from the batch: {missing[:4]} — the config's "
                "feature list and the batch disagree, refusing to mask blind")
            any_name = _S["groups"][0][0]
            B = xd[any_name].shape[0]
            dev = xd[any_name].device
            hit = torch.rand(B, device=dev, generator=_S["gen"]) < _S["p"]
            g = torch.randint(0, _S["n_groups"], (B,), device=dev, generator=_S["gen"])
            new_xd = dict(xd)
            for gi, gnames in enumerate(_S["groups"]):
                sel = hit & (g == gi)                       # [B] rows losing this product
                if not bool(sel.any()):
                    continue
                for nm in gnames:
                    t = new_xd[nm]
                    keep = (~sel).view(-1, *([1] * (t.dim() - 1)))
                    # torch.where, NOT t*mask: a multiply propagates 0*NaN = NaN,
                    # which would flip the AR fused-path dispatch. where() writes 0.0.
                    new_xd[nm] = torch.where(keep, t, t.new_zeros(()))
            if _S["probe"] is None and bool(hit.any()):
                _S["probe"] = _probe(xd, new_xd, hit, g)
            _S["seen"] += int(B)
            _S["masked"] += int(hit.sum().item())
            bg = torch.bincount(g[hit], minlength=_S["n_groups"]).tolist()
            _S["by_group"] = [a + b for a, b in zip(_S["by_group"], bg)]
            data = {**data, "x_d": new_xd}      # shallow copy: y/x_s/scaling untouched
        return inner(self, data, h_0, c_0)

    def _probe(xd, new_xd, hit, g):
        """One structural check, once per run, on a row that was actually masked:
        the chosen product's channels went to exactly zero, NOTHING ELSE changed
        (the AR input included), and no NaN was introduced."""
        import torch as _t
        b = int(_t.nonzero(hit)[0].item())
        gi = int(g[b].item())
        gnames = _S["groups"][gi]
        pre = max(float(xd[n][b].abs().max().item()) for n in gnames)
        post = max(float(new_xd[n][b].abs().max().item()) for n in gnames)
        changed = sorted(n for n in xd if bool((new_xd[n][b] != xd[n][b]).any().item()))
        nan_after = int(sum(int(_t.isnan(t).sum().item()) for t in new_xd.values()))
        nan_before = int(sum(int(_t.isnan(t).sum().item()) for t in xd.values()))
        ok = (pre > 0) and (post == 0.0) and (changed == sorted(gnames)) \
            and (nan_after == nan_before)
        rec = {"row": b, "group": _S["group_names"][gi], "features": gnames,
               "pre_absmax": pre, "post_absmax": post, "changed_features": changed,
               "nan_before": nan_before, "nan_after": nan_after, "ok": ok}
        print(f"[fdrop] PROBE {'ok' if ok else 'FAILED'}: row={b} "
              f"group={rec['group']} feats={gnames} pre|max|={pre:.4g} post|max|={post:.4g} "
              f"changed={changed} nan {nan_before}->{nan_after}", flush=True)
        assert ok, f"forcing-dropout probe FAILED: {rec}"
        return rec

    _orig_train = arlstm.ARLSTM.train

    def _train_fd(self, mode: bool = True):
        if mode and _S["seen"] > 0:
            log_epoch()
        return _orig_train(self, mode)

    arlstm.ARLSTM.forward = _forward_fd
    arlstm.ARLSTM.train = _train_fd
    arlstm.ARLSTM._l55_fd_patched = True
    print(f"[fdrop] installed p={p} groups=" +
          " ".join(f"{n}:{len(g)}" for n, g in zip(names, groups)) +
          f" n_dyn={n_dyn} n_ar={n_ar} gen_seed={hex(gen_seed)}", flush=True)


def log_epoch() -> None:
    n, k = _S["seen"], _S["masked"]
    if n == 0:
        return
    _S["epochs_logged"] += 1
    per = " ".join(f"{nm}={c}" for nm, c in zip(_S["group_names"], _S["by_group"]))
    print(f"[fdrop] cumulative after epoch {_S['epochs_logged']}: samples={n} "
          f"masked={k} rate={k / n:.5f} (target {_S['p']}) by_group[{per}]", flush=True)


def summary() -> Dict:
    n, k = _S["seen"], _S["masked"]
    return {"p": _S["p"], "groups": dict(zip(_S["group_names"] or [], _S["groups"] or [])),
            "gen_seed": _S["gen_seed"], "seen": n, "masked": k,
            "rate": (k / n) if n else None,
            "by_group": _S["by_group"],
            "by_group_frac": [c / k for c in _S["by_group"]] if k else None,
            "first_batch_probe": _S["probe"]}


def assert_applied(tol_rate: float = 0.005, tol_group: float = 0.02) -> Dict:
    """Hard end-of-run checks. Run BEFORE any dump can be produced, so a silently
    unmasked run can never enter the scored pool."""
    s = summary()
    assert _S["installed"], "forcing dropout was never installed"
    assert s["masked"] > 0, "NO SAMPLE WAS EVER MASKED — the patch is a no-op"
    assert s["first_batch_probe"] is not None, "the structural probe never ran"
    assert s["first_batch_probe"]["ok"], f"probe failed: {s['first_batch_probe']}"
    assert abs(s["rate"] - s["p"]) < tol_rate, (
        f"masking rate {s['rate']:.5f} != p {s['p']} (tol {tol_rate})")
    k = 1.0 / _S["n_groups"]
    for nm, fr in zip(_S["group_names"], s["by_group_frac"]):
        assert abs(fr - k) < tol_group, (
            f"product {nm} took {fr:.4f} of the masks, expected {k:.4f} (tol {tol_group})")
    print(f"[fdrop] ASSERTIONS PASSED {json.dumps({kk: s[kk] for kk in ('rate','by_group_frac')})}",
          flush=True)
    return s
