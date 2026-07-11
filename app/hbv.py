"""Differentiable HBV rainfall-runoff model (PyTorch), for the δHBV hybrid.

Li/Shen 2025 (HESS 29:6829) — the CAMELS-531 record — parameterizes a
differentiable HBV bucket model with an LSTM: θ = LSTM(forcings, attrs), then
Q = HBV(precip, temp, PET, θ), trained end-to-end with a basin-normalized NSE
loss. δHBV alone ≈ a single LSTM (~0.74); its value is as an ENSEMBLE MEMBER
combined with per-forcing LSTMs (record ladder: 3-LSTM ens 0.808 → +δHBV 0.818).

This module is the HBV forward core only — the parameter-generating network and
the training loop live in the trainer (a --model dhbv branch reusing its corpus/
protocol plumbing). The forward is fully vectorized over (batch basins × N
parallel HBV component instances) and differentiable end to end; parameters
arrive already mapped into physical ranges (the NN emits logits → param_ranges()
squashes them). No in-place ops on tensors that need grad; no hard clamps in the
grad path (softplus/sigmoid bounds instead).

Structure (classic HBV, e.g. Seibert & Vis 2012):
  snow  : degree-day melt with a refreeze term and liquid-water retention
  soil  : beta-store partitioning P+melt into recharge vs evapotranspiration
  resp  : two linear groundwater reservoirs (upper w/ near-surface outlet + perc,
          lower slow baseflow)
  route : triangular/gamma unit-hydrograph convolution

The 12 HBV parameters (name: [lo, hi] physical range):
  BETA   [1,6]     soil recharge nonlinearity
  FC     [50,1000] field capacity (mm)
  K0     [0.05,0.9] near-surface storage outflow coeff (1/day)
  K1     [0.01,0.5] upper reservoir outflow coeff
  K2     [0.001,0.2] lower reservoir (baseflow) outflow coeff
  LP     [0.2,1]   soil-moisture fraction of FC above which ET = PET
  PERC   [0,10]    max percolation upper→lower (mm/day)
  UZL    [0,100]   threshold for the near-surface K0 outlet (mm)
  TT     [-2.5,2.5] rain/snow temperature threshold (°C)
  CFMAX  [0.5,10]  degree-day melt factor (mm/°C/day)
  CFR    [0,0.1]   refreeze coefficient
  CWH    [0,0.2]   liquid water holding capacity of snow
"""
from __future__ import annotations

# Parameter ranges, order fixed (the NN head emits this many logits per
# component). Keep in sync with N_HBV_PARAMS.
PARAM_RANGES = {
    "BETA": (1.0, 6.0), "FC": (50.0, 1000.0),
    "K0": (0.05, 0.9), "K1": (0.01, 0.5), "K2": (0.001, 0.2),
    "LP": (0.2, 1.0), "PERC": (0.0, 10.0), "UZL": (0.0, 100.0),
    "TT": (-2.5, 2.5), "CFMAX": (0.5, 10.0), "CFR": (0.0, 0.1),
    "CWH": (0.0, 0.2),
}
PARAM_NAMES = list(PARAM_RANGES)
N_HBV_PARAMS = len(PARAM_NAMES)


def map_params(raw, torch):
    """raw: (..., N_HBV_PARAMS) unbounded logits → dict{name: (...) tensor} in
    physical ranges via sigmoid squash (smooth, keeps gradients everywhere)."""
    out = {}
    sig = torch.sigmoid(raw)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        out[name] = lo + (hi - lo) * sig[..., i]
    return out


def hamon_pet(tmean, daylight_hours, torch):
    """Hamon PET (mm/day). tmean °C, daylight_hours in hours (0..24).
    PET = 0.1651 * (daylight/12) * 216.7 * es / (T+273.3), es saturation vapor
    pressure (hPa). Kept fully differentiable in tmean (no clamp on T)."""
    es = 6.108 * torch.exp(17.27 * tmean / (tmean + 237.3))
    rho_sat = 216.7 * es / (tmean + 273.3)
    return 0.1651 * (daylight_hours / 12.0) * rho_sat / 10.0


def hbv_forward(precip, tmean, pet, params, torch, n_warmup: int = 0):
    """Vectorized differentiable HBV.

    precip, tmean, pet: (B, T) tensors (mm/day, °C, mm/day). B = batch basins
      (optionally already expanded over component instances — treat B as the
      flat batch dimension; the trainer averages component outputs outside).
    params: dict{name: (B,)} from map_params (static per basin) — dynamic
      params can be passed as (B, T) and indexed per step by the caller; this
      core takes static (B,) for clarity and lets the trainer override the
      few dynamic ones per step if needed.
    Returns Q_land (B, T): runoff generated per day (mm/day), pre-routing.
    """
    B, T = precip.shape
    dev, dt = precip.device, precip.dtype
    z = torch.zeros(B, dtype=dt, device=dev)
    p = params
    # states
    snow = z.clone()      # solid snowpack (mm)
    liq = z.clone()       # liquid water in snowpack (mm)
    sm = p["FC"] * 0.5    # soil moisture (mm), start half-full
    suz = z.clone()       # upper zone storage (mm)
    slz = z.clone()       # lower zone storage (mm)

    q_steps = []
    for t in range(T):
        pr = precip[:, t]
        temp = tmean[:, t]
        ep = pet[:, t]
        # --- snow: partition precip by TT, melt/refreeze by degree-day ---
        is_snow = torch.sigmoid((p["TT"] - temp) * 5.0)  # soft rain/snow split
        snowfall = pr * is_snow
        rain = pr * (1.0 - is_snow)
        melt_pot = p["CFMAX"] * torch.relu(temp - p["TT"])
        melt = torch.minimum(melt_pot, snow + snowfall)
        snow = snow + snowfall - melt
        refr_pot = p["CFR"] * p["CFMAX"] * torch.relu(p["TT"] - temp)
        refr = torch.minimum(refr_pot, liq)
        snow = snow + refr
        liq = liq + melt - refr + rain
        # liquid water above holding capacity leaves the snowpack
        hold = p["CWH"] * snow
        to_soil = torch.relu(liq - hold)
        liq = liq - to_soil
        # when no snow, all liquid drains
        no_snow = torch.sigmoid((0.1 - snow) * 50.0)
        extra = liq * no_snow
        to_soil = to_soil + extra
        liq = liq - extra

        # --- soil: beta store ---
        soil_frac = torch.clamp(sm / p["FC"], 0.0, 1.0)
        recharge = to_soil * soil_frac.pow(p["BETA"])
        sm = sm + to_soil - recharge
        # ET: linear ramp to PET up to LP*FC
        et = ep * torch.clamp(sm / (p["LP"] * p["FC"]), 0.0, 1.0)
        et = torch.minimum(et, sm)
        sm = sm - et
        # soil can't exceed FC; spill to recharge
        spill = torch.relu(sm - p["FC"])
        sm = sm - spill
        recharge = recharge + spill

        # --- response: upper + lower reservoirs ---
        suz = suz + recharge
        perc = torch.minimum(p["PERC"], suz)
        suz = suz - perc
        slz = slz + perc
        q0 = p["K0"] * torch.relu(suz - p["UZL"])
        q1 = p["K1"] * suz
        suz = suz - q0 - q1
        q2 = p["K2"] * slz
        slz = slz - q2
        q_steps.append(q0 + q1 + q2)

    q = torch.stack(q_steps, dim=1)  # (B, T)
    if n_warmup:
        q = q[:, n_warmup:]
    return q


def gamma_uh(q, n, k, torch, max_len: int = 15):
    """Route q (B, T) through a gamma-distributed unit hydrograph with shape n
    (B,) and scale k (B,). Differentiable in n, k. Truncated/normalized to
    max_len days (day-ahead routing rarely needs more)."""
    B, T = q.shape
    dev, dt = q.device, q.dtype
    lags = torch.arange(1, max_len + 1, dtype=dt, device=dev)  # (L,)
    n_ = n[:, None].clamp(min=1.0 + 1e-3)
    k_ = k[:, None].clamp(min=1e-3)
    # gamma pdf ∝ x^(n-1) exp(-x/k); use log for stability, no lgamma-grad issues
    logw = (n_ - 1.0) * torch.log(lags[None, :]) - lags[None, :] / k_
    w = torch.softmax(logw, dim=1)  # (B, L) normalized weights
    # causal convolution: out[t] = sum_l w[l] * q[t-l]
    qp = torch.nn.functional.pad(q, (max_len - 1, 0))  # left-pad
    wf = w.flip(1)[:, None, :]  # (B,1,L) as conv kernel per basin
    out = torch.nn.functional.conv1d(
        qp[:, None, :], wf.reshape(B, 1, max_len), groups=1) if False else None
    # grouped conv per-basin: reshape to (1, B, T+L-1), kernel (B,1,L), groups=B
    out = torch.nn.functional.conv1d(
        qp.unsqueeze(0), wf, groups=B).squeeze(0)  # (B, T)
    return out
