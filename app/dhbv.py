"""δHBV hybrid model: an LSTM parameter-net wrapping the differentiable HBV core.

Li/Shen 2025 (HESS 29:6829) — the CAMELS-531 record. An LSTM ingests the same
normalized forcings + static attributes as the MB-LSTM, but instead of emitting
a flow distribution it emits HBV PARAMETERS: static params from its final state,
plus a few DYNAMIC (per-timestep) params. The differentiable HBV core
(app/hbv.py) then runs those params over the raw physical forcing sequence to
produce streamflow. The whole thing is trained end-to-end with the basin-NSE
loss — gradients flow flow-loss → HBV → params → LSTM.

Design (matches the plan / the record recipe):
- Reuse the SAME encoder+decoder LSTM stack as MBLSTMNet (identical enc_in/
  dec_in sizing) so a δHBV run can optionally warm-start from a trained LSTM
  checkpoint (shape-filtered strict=False load).
- Static HBV params ← encoder final hidden state → static_head.
- Dynamic params (BETA, K0) ← per-step LSTM outputs over the full 365+14
  sequence → dynamic_head → sigmoid-squashed to their physical ranges.
- The HBV core runs on RAW PHYSICAL precip/tmean/PET (threaded separately by the
  trainer — NOT reconstructed from the z-scored x_enc), warmed up over the 365
  context days, and only the last HORIZON days are returned. Output is physical
  cfs (mm/day × area / 2.44658); the trainer z-transforms it into the target
  space before the loss.
"""
from __future__ import annotations

from app import hbv

CFS_PER_MM_DAY = 2.44658  # q_cfs = q_mm_day * area_km2 / CFS_PER_MM_DAY  (verified)


def build_dhbv_model(cfg: dict):
    """Construct the δHBV module from a checkpoint cfg dict. Lives here so
    training and serving share one architecture (the MB-LSTM invariant)."""
    import torch
    import torch.nn as nn

    # Same input dims as MBLSTMNet so the LSTM stack is warm-start-compatible.
    enc_in = len(cfg["enc_vars"]) + 2 + 2 + len(cfg["static_feats"])   # +q,+qmask,+doy
    dec_in = len(cfg["dec_vars"]) + 2 + 1 + len(cfg["static_feats"])   # +doy,+lead
    hidden = int(cfg["hidden"])
    horizon = int(cfg.get("horizon", 14))
    n_static = hbv.N_HBV_PARAMS
    # Optional dynamic-γ routing: also predict ROUTN/ROUTK per-timestep so the
    # unit hydrograph varies in time (off by default — existing ckpts unaffected).
    dyn_route = bool(cfg.get("dynamic_routing", False))
    dyn_names = tuple(hbv.DYNAMIC_PARAMS) + (("ROUTN", "ROUTK") if dyn_route else ())
    n_dyn = len(dyn_names)
    # Optional forcing-error correction (B5): a per-timestep learned multiplier
    # on raw precip (the dominant forcing error), bounded and mass-aware, so the
    # LSTM cancels systematic input bias before HBV. Off by default.
    forcing_corr = bool(cfg.get("forcing_correction", False))
    # max multiplicative factor: precip is scaled in [1/F, F] via F**tanh(.)
    corr_max = float(cfg.get("forcing_corr_max", 2.0))
    # nmul: number of parallel HBV component instances per basin (Li/Shen use 16).
    # Each component has its own param set; the 16 land discharges are AVERAGED,
    # then routed once. Default 1 = single instance (backward-compatible).
    nmul = int(cfg.get("nmul", 1))
    # BETAET (16th param, 3rd dynamic) shipped in the same commit as nmul.
    # Checkpoints from before it (no "nmul" key in cfg) have 15-logit static
    # heads and 2-param dynamic heads — infer the gate so they still load.
    betaet = bool(cfg.get("betaet", "nmul" in cfg))
    param_names = [n for n in hbv.PARAM_NAMES if betaet or n != "BETAET"]
    n_static = len(param_names)
    if not betaet:
        dyn_names = tuple(n for n in dyn_names if n != "BETAET")
        n_dyn = len(dyn_names)

    class DHBVNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(enc_in, hidden, batch_first=True)
            self.decoder = nn.LSTM(dec_in, hidden, batch_first=True)
            # static params: nmul component sets from the encoder final state
            self.static_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, n_static * nmul))
            # dynamic (per-timestep) params, nmul component sets per step
            self.dynamic_head = nn.Linear(hidden, n_dyn * nmul)
            self.horizon = horizon
            self.dyn_route = dyn_route
            self.dyn_names = dyn_names
            self.nmul = nmul
            self.forcing_corr = forcing_corr
            self.corr_max = corr_max
            if forcing_corr:
                # per-step precip log-multiplier; zero-init so it starts as ×1
                self.precip_corr_head = nn.Linear(hidden, 1)
                nn.init.zeros_(self.precip_corr_head.weight)
                nn.init.zeros_(self.precip_corr_head.bias)

        def forward(self, x_enc, x_dec, raw_precip, raw_tmean, raw_pet, area_km2):
            """x_enc (B,365,enc_in), x_dec (B,H,dec_in) z-scored — drive the LSTM.
            raw_precip/tmean/pet (B, 365+H) physical — drive HBV. area_km2 (B,).
            Returns q_cfs (B, H) physical streamflow for the H horizon days."""
            B = x_enc.shape[0]
            M = self.nmul
            enc_out, (h, c) = self.encoder(x_enc)          # enc_out (B,365,hid)
            dec_out, _ = self.decoder(x_dec, (h, c))       # (B,H,hid)
            seq_hidden = torch.cat([enc_out, dec_out], dim=1)   # (B, 365+H, hid)
            Tseq = seq_hidden.shape[1]
            # static params → nmul component sets: (B, n_static*M) → (B*M, n_static)
            static_logits = self.static_head(h[-1]).view(B, M, n_static).reshape(B * M, n_static)
            params = hbv.map_params(static_logits, torch, names=param_names)  # dict{name:(B*M,)}
            # dynamic params over the sequence, nmul sets: (B,T,n_dyn*M)→(B*M,T,n_dyn)
            dsig = torch.sigmoid(self.dynamic_head(seq_hidden)).view(B, Tseq, M, n_dyn)
            dsig = dsig.permute(0, 2, 1, 3).reshape(B * M, Tseq, n_dyn)  # (B*M,T,n_dyn)
            dyn = {}
            for i, name in enumerate(self.dyn_names):
                lo, hi = hbv.PARAM_RANGES[name]
                dyn[name] = lo + (hi - lo) * dsig[..., i]     # (B*M, Tseq)
            hbv_dyn = {k: v for k, v in dyn.items() if k in hbv.DYNAMIC_PARAMS}
            # expand raw forcings over the M components
            rp, rt, rpet = raw_precip, raw_tmean, raw_pet
            if self.forcing_corr:
                logm = torch.tanh(self.precip_corr_head(seq_hidden)[..., 0])  # (B,Tseq)
                rp = rp * self.corr_max ** logm
            if M > 1:
                rp = rp.repeat_interleave(M, dim=0)          # (B*M, Tseq)
                rt = rt.repeat_interleave(M, dim=0)
                rpet = rpet.repeat_interleave(M, dim=0)
            # run HBV per component, then AVERAGE the M land discharges
            q_land = hbv.hbv_forward(rp, rt, rpet, params, torch,
                                     n_warmup=0, dyn=hbv_dyn)   # (B*M, Tseq) mm/day
            if M > 1:
                q_land = q_land.view(B, M, Tseq).mean(dim=1)   # (B, Tseq)
            # routing params (component-0 set is fine; take mean over M): (B,)
            def _rp(name):
                v = params[name]
                return v.view(B, M).mean(dim=1) if M > 1 else v
            if self.dyn_route:
                routn = dyn["ROUTN"].view(B, M, Tseq).mean(1) if M > 1 else dyn["ROUTN"]
                routk = dyn["ROUTK"].view(B, M, Tseq).mean(1) if M > 1 else dyn["ROUTK"]
            else:
                routn, routk = _rp("ROUTN"), _rp("ROUTK")     # (B,)
            q_routed = hbv.gamma_uh(q_land, routn, routk, torch)
            q_mm = q_routed[:, -self.horizon:]                # (B, H) mm/day
            q_cfs = q_mm * area_km2[:, None] / CFS_PER_MM_DAY  # (B, H) cfs
            return q_cfs

    return DHBVNet()
