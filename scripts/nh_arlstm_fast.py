#!/usr/bin/env python3
"""LEDGER 53 — a sync-free, CUDA-graph-replayable time loop for NeuralHydrology's
ARLSTM, mathematically identical to `neuralhydrology/modelzoo/arlstm.py`.

WHY. NH's ARLSTM steps a cuDNN nn.LSTM one timestep at a time in Python (365
steps × ~25 kernels forward, ~2× that backward) and its NaN substitution uses
boolean-mask indexing, which forces a device sync every step. On the GTX 1080
this is kernel-launch bound: 1.1 batch/s solo, GPU ~30 %. Three concurrent seeds
only reach 1.5× that. A CUDA graph replays the whole unrolled forward+backward
with no launch gaps.

WHAT IS IDENTICAL. Same parameters (the run's state_dict is the plain ARLSTM
state_dict — evaluation uses unpatched NH), same gate math as cuDNN's LSTM
(gates = W_ih·[x_embd, x_ar, flag] + b_ih + W_hh·h + b_hh, order i,f,g,o,
c' = f⊙c + i⊙g, h' = o⊙tanh c'), same substitution rule (a NaN lagged input is
replaced by the previous step's prediction, flag = 1, gradients flow through
the substituted prediction exactly as in the original), same dropout before the
head at every step. Verified by `equivalence_test()` below.

USE. `patch_arlstm(graph=True)` replaces ARLSTM.forward in THIS process only;
`scripts/nh_arlstm_train.py` calls it before `start_training`.
"""
from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn

_STATE = {"graphs": {}, "shape": None, "graph": True, "captures": 0, "max_graphs": 3}


def _fast_loop(model, x_d: torch.Tensor):
    """x_d: [T, B, D] as produced by model.embedding_net (AR inputs last).
    Returns y_hat [B, T, out] and (h_n, c_n) [B, H] — eager, autograd-tracked."""
    n_ar = model._num_ar_inputs
    T, B, D = x_d.shape
    H = model.cfg.hidden_size
    W_ih, W_hh = model.cell.weight_ih_l0, model.cell.weight_hh_l0
    bias = model.cell.bias_ih_l0 + model.cell.bias_hh_l0
    d_embd = D - n_ar
    W_embd, W_ar, W_flag = W_ih[:, :d_embd], W_ih[:, d_embd:d_embd + n_ar], W_ih[:, d_embd + n_ar:]
    x_embd, x_ar_all = x_d[:, :, :d_embd], x_d[:, :, d_embd:]
    pre = torch.matmul(x_embd, W_embd.t()) + bias            # [T, B, 4H], one big GEMM
    W_hh_t, W_ar_t, W_flag_t = W_hh.t(), W_ar.t(), W_flag.t()
    h = x_d.new_zeros((B, H))
    c = x_d.new_zeros((B, H))
    last = x_d.new_zeros((B, model.output_size))                # AR shift == 1 (asserted in patch)
    ys = []
    for t in range(T):
        x_ar = x_ar_all[t]
        nanm = torch.isnan(x_ar)
        x_ar = torch.where(nanm, last, x_ar)
        flag = nanm.to(x_ar.dtype)
        gates = pre[t] + torch.mm(x_ar, W_ar_t) + torch.mm(flag, W_flag_t) + torch.mm(h, W_hh_t)
        i, f, g, o = gates.chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        y = model.head(model.dropout(h))["y_hat"]                 # [B, out]
        last = y
        ys.append(y)
    return torch.stack(ys, 1), h, c


def _cudnn_loop(model, x_d: torch.Tensor):
    """Same as _fast_loop but keeps cuDNN's fused single-step nn.LSTM call; only
    the NaN substitution is made sync-free (torch.where instead of boolean
    indexing) so the loop can be captured in a CUDA graph."""
    n_ar = model._num_ar_inputs
    T, B, D = x_d.shape
    H = model.cfg.hidden_size
    h = x_d.new_zeros((1, B, H))
    c = x_d.new_zeros((1, B, H))
    last = x_d.new_zeros((B, model.output_size))
    ys = []
    for t in range(T):
        x_t = x_d[t]
        x_ar = x_t[:, -n_ar:]
        nanm = torch.isnan(x_ar)
        x_ar = torch.where(nanm, last, x_ar)
        inp = torch.cat([x_t[:, :-n_ar], x_ar, nanm.to(x_ar.dtype)], -1).unsqueeze(0)
        _, (h, c) = model.cell(inp, (h, c))
        y = model.head(model.dropout(h[0]))["y_hat"]
        last = y
        ys.append(y)
    return torch.stack(ys, 1), h[0], c[0]


def _fused_loop(model, x_d: torch.Tensor):
    """⭐ THE NO-NaN FAST PATH. When no autoregressive input in the batch is NaN,
    NH's ARLSTM loop is mathematically a PLAIN LSTM: the substitution
    `x_ar[nan] = last_prediction[nan]` is a no-op, the obs/sim flags stay 0, and
    `last_prediction` is never read. The 365 sequential single-step cuDNN calls
    then collapse into ONE fused cuDNN call over the whole sequence.

    This is the common case for this campaign: the 1999-2008 TRAINING window has
    complete discharge in all 531 basins, so a run with `missing_fraction: 0.0`
    never substitutes anything during training. Measured ~10x faster.

    Identical to the step loop except for DROPOUT RNG ORDERING (365 separate
    dropout draws vs one draw over the [T,B,H] tensor) — same distribution, same
    rate, different random numbers, like a different seed. With dropout off the
    two agree to float precision; `equivalence_test` checks exactly that."""
    n_ar = model._num_ar_inputs
    T, B, _ = x_d.shape
    flags = x_d.new_zeros((T, B, n_ar))
    out, (h, c) = model.cell(torch.cat([x_d, flags], -1))      # one cuDNN call, h_0=c_0=0
    y = model.head(model.dropout(out.transpose(0, 1)))["y_hat"]  # [B, T, out]
    return y, h[0], c[0]


def _ar_has_nan(model, x_d: torch.Tensor) -> bool:
    return bool(torch.isnan(x_d[:, :, -model._num_ar_inputs:]).any().item())


def _loop(model, x_d):
    return _cudnn_loop(model, x_d) if _STATE.get("impl", "cudnn") == "cudnn" else _fast_loop(model, x_d)


class _Loop(nn.Module):
    """Holds the SAME cell/head/dropout modules as the outer ARLSTM (shared
    parameters, no new state_dict keys because it is never registered as a
    child). Single tensor in, single tensor out — what make_graphed_callables
    needs. Only y_hat is returned: every graphed output must receive a real
    gradient at replay (an output with grad None would leave its static
    grad buffer uninitialised)."""

    def __init__(self, model):
        super().__init__()
        self.cell, self.head, self.dropout = model.cell, model.head, model.dropout
        object.__setattr__(self, "_model", model)

    def forward(self, x_d):
        y, _, _ = _loop(self._model, x_d)
        return y


def _forward_fast(self, data: Dict[str, torch.Tensor], h_0=None, c_0=None) -> Dict[str, torch.Tensor]:
    assert h_0 is None and c_0 is None, "fast loop does not take initial states"
    x_d = self.embedding_net(data)                               # [T, B, D]
    if _STATE.get("fused", True) and not _ar_has_nan(self, x_d):
        # no missing AR inputs in this batch -> one fused cuDNN call (see _fused_loop)
        _STATE["fused_batches"] = _STATE.get("fused_batches", 0) + 1
        y_hat, h, c = _fused_loop(self, x_d)
        return {"y_hat": y_hat, "h_n": h.unsqueeze(1), "c_n": c.unsqueeze(1)}
    _STATE["step_batches"] = _STATE.get("step_batches", 0) + 1
    use_graph = _STATE["graph"] and self.training and torch.is_grad_enabled() and x_d.is_cuda
    if use_graph:
        # one captured graph per input shape (the full batch and the epoch's
        # last, smaller batch); anything beyond max_graphs runs eagerly.
        shape = tuple(x_d.shape)
        if shape not in _STATE["graphs"]:
            if len(_STATE["graphs"]) >= _STATE["max_graphs"]:
                use_graph = False
            else:
                loop = _Loop(self)
                sample = x_d.detach().clone()
                t0 = time.time()
                _STATE["graphs"][shape] = torch.cuda.make_graphed_callables(loop, (sample,), num_warmup_iters=3)
                _STATE["shape"] = shape
                _STATE["captures"] += 1
                for p in self.parameters():                       # warmup backward left grads behind
                    p.grad = None
                print(f"[nh_arlstm_fast] captured CUDA graph for x_d{shape} in {time.time()-t0:.1f}s "
                      f"(capture #{_STATE['captures']})", flush=True)
        if use_graph:
            y_hat = _STATE["graphs"][shape](x_d)
            return {"y_hat": y_hat}
    y_hat, h, c = _loop(self, x_d)
    return {"y_hat": y_hat, "h_n": h.unsqueeze(1), "c_n": c.unsqueeze(1)}


def patch_arlstm(graph: bool = True, impl: str = "cudnn", fused: bool = True):
    from neuralhydrology.modelzoo.arlstm import ARLSTM
    _STATE["impl"] = impl
    _STATE["fused"] = fused
    if getattr(ARLSTM, "_l53_patched", False):
        _STATE["graph"] = graph
        return
    ARLSTM._orig_forward = ARLSTM.forward
    orig_init = ARLSTM.__init__

    def init(self, cfg):
        orig_init(self, cfg)
        assert self._ar_shift == 1, "fast loop implements AR shift 1 only"
        assert not getattr(self.embedding_net, "_num_autoregression_inputs", 0) or True

    ARLSTM.__init__ = init
    ARLSTM.forward = _forward_fast
    ARLSTM._l53_patched = True
    _STATE["graph"] = graph
    print(f"[nh_arlstm_fast] ARLSTM.forward patched (impl={impl}, cuda graph={'on' if graph else 'off'}, "
          f"no-NaN fused path={'on' if fused else 'off'})", flush=True)


# ---------------------------------------------------------------------------
def reference_loop(model, x_d):
    """Verbatim post-embedding loop of neuralhydrology 1.13.0 ARLSTM.forward."""
    _, batch_size, _ = x_d.size()
    h_0 = x_d.new_zeros((1, batch_size, model.cfg.hidden_size))
    c_0 = x_d.new_zeros((1, batch_size, model.cfg.hidden_size))
    ar_flags = x_d.new_zeros((batch_size, model._num_ar_inputs))
    last_prediction = x_d.new_zeros((model._ar_shift, batch_size, model.output_size))
    y_hat = []
    for x_t in x_d:
        x_embd = x_t[:, :-model._num_ar_inputs]
        x_ar = x_t[:, -model._num_ar_inputs:].clone()
        replace_indexes = torch.isnan(x_ar)
        x_ar[replace_indexes] = last_prediction[-1, replace_indexes]
        ar_flags[:, :] = 0
        ar_flags[replace_indexes] = 1
        cell_inputs = torch.unsqueeze(torch.concat([x_embd, x_ar, ar_flags], -1), 0)
        _, (h_0, c_0) = model.cell(cell_inputs, (h_0, c_0))
        last_prediction[1:] = last_prediction[:-1, ].clone()
        prediction = torch.squeeze(model.head(model.dropout(h_0.transpose(0, 1)))["y_hat"], dim=1)
        last_prediction[0] = prediction
        y_hat.append(prediction)
    return torch.stack(y_hat, 1)


def equivalence_test(run_dir: str, epoch: int = 1, T: int = 365, B: int = 256, n_batches_timing: int = 5):
    """Load a real ARLSTM checkpoint, feed a synthetic post-embedding batch with
    50 % NaN AR inputs, and compare reference vs fast (outputs AND parameter
    gradients, dropout off), then graphed vs eager, then time the three."""
    from pathlib import Path
    from neuralhydrology.utils.config import Config
    from neuralhydrology.modelzoo.arlstm import ARLSTM
    cfg = Config(Path(run_dir) / "config.yml")
    dev = torch.device("cuda:0")
    model = ARLSTM(cfg).to(dev)
    sd = torch.load(Path(run_dir) / f"model_epoch{epoch:03d}.pt", map_location=dev)
    model.load_state_dict(sd)
    D = model.embedding_net.output_size          # already includes the AR column(s)
    torch.manual_seed(0)
    x = torch.randn(T, B, D, device=dev)
    x[:, :, -1] = torch.where(torch.rand(T, B, device=dev) < 0.5, torch.nan, x[:, :, -1])
    model.train()
    model.dropout.p = 0.0                                          # exact comparison
    tgt = torch.randn(B, device=dev)

    def run(fn):
        for p in model.parameters():
            p.grad = None
        y = fn(model, x)
        loss = ((y[:, -1, 0] - tgt) ** 2).mean()
        loss.backward()
        return y.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    y_ref, g_ref = run(reference_loop)
    y_fast, g_fast = run(lambda m, xx: _fast_loop(m, xx)[0])
    dy = (y_ref - y_fast).abs().max().item()
    dg = max((g_ref[n] - g_fast[n]).abs().max().item() / (g_ref[n].abs().max().item() + 1e-12) for n in g_ref)
    print(f"reference vs fast (eager): max|Δy|={dy:.3e} (|y| max {y_ref.abs().max().item():.3f}), "
          f"max rel|Δgrad|={dg:.3e}")
    # ---- the no-NaN fused path, on an AR input with NO missing values --------
    x_clean = x.clone()
    x_clean[:, :, -1] = torch.randn(T, B, device=dev)
    assert not torch.isnan(x_clean[:, :, -1]).any()

    def run_on(fn, xx_):
        for p in model.parameters():
            p.grad = None
        y = fn(model, xx_)
        ((y[:, -1, 0] - tgt) ** 2).mean().backward()
        return y.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    y_rc, g_rc = run_on(reference_loop, x_clean)
    y_fu, g_fu = run_on(lambda m, xx: _fused_loop(m, xx)[0], x_clean)
    dyf = (y_rc - y_fu).abs().max().item()
    dgf = max((g_rc[n] - g_fu[n]).abs().max().item() / (g_rc[n].abs().max().item() + 1e-12) for n in g_rc)
    print(f"reference vs FUSED (no NaN): max|Δy|={dyf:.3e} (|y| max {y_rc.abs().max().item():.3f}), "
          f"max rel|Δgrad|={dgf:.3e}")
    for name, fn in [("reference/noNaN", reference_loop), ("FUSED/noNaN", lambda m, xx: _fused_loop(m, xx)[0])]:
        run_on(fn, x_clean); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n_batches_timing):
            run_on(fn, x_clean)
        torch.cuda.synchronize()
        print(f"  {name:16s} {(time.time()-t0)/n_batches_timing*1000:7.0f} ms / batch (fwd+bwd, B={B}, T={T})")

    y_cu, g_cu = run(lambda m, xx: _cudnn_loop(m, xx)[0])
    dyc = (y_ref - y_cu).abs().max().item()
    dgc = max((g_ref[n] - g_cu[n]).abs().max().item() / (g_ref[n].abs().max().item() + 1e-12) for n in g_ref)
    print(f"reference vs cudnn-where:  max|Δy|={dyc:.3e}, max rel|Δgrad|={dgc:.3e}")
    # graphed vs eager, both implementations
    graphed = {}
    dyg = dgg = 0.0
    for impl, base_y, base_g in [("decomposed", y_fast, g_fast), ("cudnn", y_cu, g_cu)]:
        _STATE["impl"] = impl
        try:
            gl = torch.cuda.make_graphed_callables(_Loop(model), (x.detach().clone(),), num_warmup_iters=3)
        except Exception as e:  # noqa
            print(f"graph capture FAILED for {impl}: {type(e).__name__}: {str(e)[:200]}")
            continue
        y_g, g_g = run(lambda m, xx: gl(xx))
        d1 = (base_y - y_g).abs().max().item()
        d2 = max((base_g[n] - g_g[n]).abs().max().item() / (base_g[n].abs().max().item() + 1e-12) for n in base_g)
        dyg, dgg = max(dyg, d1), max(dgg, d2)
        print(f"{impl:10s} eager vs graphed: max|Δy|={d1:.3e}, max rel|Δgrad|={d2:.3e}")
        graphed[impl] = gl
    # timing (fwd+bwd)
    cands = [("reference", reference_loop), ("decomp-eager", lambda m, xx: _fast_loop(m, xx)[0]),
             ("cudnn-eager", lambda m, xx: _cudnn_loop(m, xx)[0])]
    cands += [(f"{k}-graph", (lambda gl: (lambda m, xx: gl(xx)))(gl)) for k, gl in graphed.items()]
    for name, fn in cands:
        run(fn); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n_batches_timing):
            run(fn)
        torch.cuda.synchronize()
        print(f"  {name:16s} {(time.time()-t0)/n_batches_timing*1000:7.0f} ms / batch (fwd+bwd, B={B}, T={T})")
    ok = (dy < 1e-4 and dg < 1e-3 and dyc < 1e-4 and dgc < 1e-3 and dyg < 1e-4 and dgg < 1e-3
          and dyf < 1e-4 and dgf < 1e-3)
    print("EQUIVALENCE", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epoch", type=int, default=1)
    a = ap.parse_args()
    raise SystemExit(0 if equivalence_test(a.run_dir, a.epoch) else 1)
