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
    n_dyn = len(hbv.DYNAMIC_PARAMS)

    class DHBVNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(enc_in, hidden, batch_first=True)
            self.decoder = nn.LSTM(dec_in, hidden, batch_first=True)
            # static params from the encoder's final hidden state
            self.static_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, n_static))
            # dynamic (per-timestep) params from every LSTM output step
            self.dynamic_head = nn.Linear(hidden, n_dyn)
            self.horizon = horizon

        def forward(self, x_enc, x_dec, raw_precip, raw_tmean, raw_pet, area_km2):
            """x_enc (B,365,enc_in), x_dec (B,H,dec_in) z-scored — drive the LSTM.
            raw_precip/tmean/pet (B, 365+H) physical — drive HBV. area_km2 (B,).
            Returns q_cfs (B, H) physical streamflow for the H horizon days."""
            enc_out, (h, c) = self.encoder(x_enc)          # enc_out (B,365,hid)
            dec_out, _ = self.decoder(x_dec, (h, c))       # (B,H,hid)
            # static params from the final encoder state
            static_logits = self.static_head(h[-1])        # (B, n_static)
            params = hbv.map_params(static_logits, torch)  # dict{name:(B,)}
            # dynamic params over the full 365+H sequence
            seq_hidden = torch.cat([enc_out, dec_out], dim=1)   # (B, 365+H, hid)
            dyn_sig = torch.sigmoid(self.dynamic_head(seq_hidden))  # (B, Tseq, n_dyn)
            dyn = {}
            for i, name in enumerate(hbv.DYNAMIC_PARAMS):
                lo, hi = hbv.PARAM_RANGES[name]
                dyn[name] = lo + (hi - lo) * dyn_sig[..., i]     # (B, Tseq)
            # run HBV over the whole sequence; keep only the horizon days
            q_land = hbv.hbv_forward(raw_precip, raw_tmean, raw_pet, params,
                                     torch, n_warmup=0, dyn=dyn)   # (B, Tseq) mm/day
            # learnable gamma-UH routing (ROUTN shape, ROUTK scale, both static)
            q_routed = hbv.gamma_uh(q_land, params["ROUTN"], params["ROUTK"], torch)
            q_mm = q_routed[:, -self.horizon:]                # (B, H) mm/day
            q_cfs = q_mm * area_km2[:, None] / CFS_PER_MM_DAY  # (B, H) cfs
            return q_cfs

    return DHBVNet()
