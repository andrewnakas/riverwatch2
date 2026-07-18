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
    # δHBV1.1p capillary-rise: water drawn from the upper groundwater zone back
    # into the soil store when the soil is dry (Li/Shen 2025 addition).
    "CAPRISE": (0.0, 3.0),
    # BETAET: ET-shape exponent (Li/Shen's "γ", the 3rd DYNAMIC param in the
    # code-verified δHBV1.1p). ET = PET · (SM/FC)^BETAET (was a linear ramp).
    "BETAET": (0.3, 5.0),
    # Gamma unit-hydrograph routing: shape ROUTN, scale ROUTK (both learnable).
    "ROUTN": (1.0, 5.0), "ROUTK": (0.5, 5.0),
}
PARAM_NAMES = list(PARAM_RANGES)
N_HBV_PARAMS = len(PARAM_NAMES)

# The parameters δHBV1.1p predicts DYNAMICALLY (per-timestep). The code-verified
# record recipe (mhpi/hydrodl2 hbv_1_1p) uses BETA + K0 + BETAET.
DYNAMIC_PARAMS = ("BETA", "K0", "BETAET")


def map_params(raw, torch, names=None):
    """raw: (..., len(names or PARAM_NAMES)) unbounded logits → dict{name: (...)
    tensor} in physical ranges via sigmoid squash (smooth, keeps gradients
    everywhere). `names` lets pre-BETAET checkpoints map their 15 logits."""
    out = {}
    sig = torch.sigmoid(raw)
    for i, name in enumerate(names or PARAM_NAMES):
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


def extraterrestrial_radiation(lat_rad, doy, np):
    """FAO-56 extraterrestrial radiation Ra (MJ/m^2/day) from latitude (radians)
    and day-of-year. Pure numpy (computed once per station at corpus load, not
    in the grad path). lat_rad, doy broadcast; returns same shape."""
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)          # inv earth-sun dist
    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)       # solar declination
    # sunset hour angle; clip the arccos argument to [-1,1] (polar day/night)
    x = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    ws = np.arccos(x)
    ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.sin(ws))
    return np.maximum(ra, 0.0)


def hargreaves_pet(tmean, tmax, tmin, lat_rad, doy, np):
    """Hargreaves (1985) PET (mm/day), the method Li/Shen 2025 use for δHBV.
    PET = 0.0023 * Ra * (tmean + 17.8) * sqrt(max(tmax - tmin, 0)).
    Pure numpy — computed per station at corpus load (not differentiable; the
    LSTM learns HBV params, not PET). tmean/tmax/tmin °C, lat_rad radians."""
    ra = extraterrestrial_radiation(lat_rad, doy, np)
    trange = np.sqrt(np.maximum(tmax - tmin, 0.0))
    pet = 0.0023 * ra * (tmean + 17.8) * trange
    return np.maximum(pet, 0.0)


def hbv_forward(precip, tmean, pet, params, torch, n_warmup: int = 0, dyn=None):
    """Vectorized differentiable HBV.

    precip, tmean, pet: (B, T) tensors (mm/day, °C, mm/day). B = batch basins
      (optionally already expanded over component instances — treat B as the
      flat batch dimension; the trainer averages component outputs outside).
    params: dict{name: (B,)} from map_params (static per basin).
    dyn: optional dict{name: (B, T)} of TIME-VARYING params (δHBV1.1p predicts
      BETA, K0 per timestep). A name present in `dyn` overrides `params[name]`
      per step; anything not in `dyn` uses the static `params` value.
    Returns Q_land (B, T): runoff generated per day (mm/day), pre-routing.
    """
    B, T = precip.shape
    dev, dt = precip.device, precip.dtype
    z = torch.zeros(B, dtype=dt, device=dev)
    p = params
    dyn = dyn or {}
    if "BETAET" not in p and "BETAET" not in dyn:
        # pre-BETAET checkpoints: exponent 1 == the original linear-ramp ET
        p = dict(p, BETAET=z + 1.0)

    def par(name, t):  # per-step value: dynamic (B,) slice if given, else static
        return dyn[name][:, t] if name in dyn else p[name]

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
        beta_t = par("BETA", t)
        k0_t = par("K0", t)
        betaet_t = par("BETAET", t)
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

        # --- soil: beta store (BETA may be per-step dynamic) ---
        soil_frac = torch.clamp(sm / p["FC"], 0.0, 1.0)
        recharge = to_soil * soil_frac.pow(beta_t)
        sm = sm + to_soil - recharge
        # ET: PET scaled by (SM/LP*FC)^BETAET (Li/Shen ET-shape exponent; was a
        # linear ramp, i.e. BETAET=1). Clamp base to [eps,1]: d/dx x^b → ∞ at x=0
        # for b<1, which makes the backward pass emit inf grads (every batch then
        # trips the skip-guard). The eps floor keeps the gradient finite.
        et_base = torch.clamp(sm / (p["LP"] * p["FC"]), 1e-4, 1.0)
        et = ep * et_base.pow(betaet_t)
        et = torch.minimum(et, sm)
        sm = sm - et
        # soil can't exceed FC; spill to recharge
        spill = torch.relu(sm - p["FC"])
        sm = sm - spill
        recharge = recharge + spill

        # --- response: upper + lower reservoirs ---
        suz = suz + recharge
        # δHBV1.1p capillary rise: upper zone → soil when soil is below capacity,
        # scaled by the dryness deficit (drawn only from available suz water).
        caprise = torch.minimum(p["CAPRISE"] * (1.0 - soil_frac), suz)
        suz = suz - caprise
        sm = sm + caprise
        perc = torch.minimum(p["PERC"], suz)
        suz = suz - perc
        slz = slz + perc
        q0 = k0_t * torch.relu(suz - p["UZL"])
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
    and scale k. Differentiable in n, k. Truncated/normalized to max_len days
    (day-ahead routing rarely needs more).

    n, k may be either STATIC per-basin (B,) — the classic δHBV routing — or
    TIME-VARYING (B, T), in which case each output day t is routed with its own
    UH kernel (dynamic-γ routing; a per-window-mean of the dynamic values is the
    cheaper approximation the trainer can pass instead of full per-step). Both
    modes are differentiable in n, k.
    """
    B, T = q.shape
    dev, dt = q.device, q.dtype
    lags = torch.arange(1, max_len + 1, dtype=dt, device=dev)  # (L,)

    if n.dim() == 2:  # time-varying UH: (B, T) shape/scale → per-day kernels
        n_ = n.clamp(min=1.0 + 1e-3)[:, :, None]                 # (B,T,1)
        k_ = k.clamp(min=1e-3)[:, :, None]                       # (B,T,1)
        logw = (n_ - 1.0) * torch.log(lags[None, None, :]) - lags[None, None, :] / k_
        w = torch.softmax(logw, dim=2)                           # (B,T,L)
        # out[t] = sum_l w[t,l] * q[t-l]  (causal, each day's own kernel)
        qp = torch.nn.functional.pad(q, (max_len - 1, 0))        # (B, T+L-1)
        # gather the L past values ending at each t: (B,T,L)
        idx = torch.arange(T, device=dev)[:, None] + torch.arange(max_len, device=dev)[None, :]
        past = qp[:, idx]                                        # (B,T,L), oldest→newest
        return (past * w.flip(2)).sum(dim=2)                     # align w (lag1..L) to newest→oldest

    n_ = n[:, None].clamp(min=1.0 + 1e-3)
    k_ = k[:, None].clamp(min=1e-3)
    # gamma pdf ∝ x^(n-1) exp(-x/k); use log for stability, no lgamma-grad issues
    logw = (n_ - 1.0) * torch.log(lags[None, :]) - lags[None, :] / k_
    w = torch.softmax(logw, dim=1)  # (B, L) normalized weights
    # causal convolution: out[t] = sum_l w[l] * q[t-l]
    qp = torch.nn.functional.pad(q, (max_len - 1, 0))  # left-pad
    wf = w.flip(1)[:, None, :]  # (B,1,L) as conv kernel per basin
    # grouped conv per-basin: input (1, B, T+L-1), kernel (B,1,L), groups=B
    out = torch.nn.functional.conv1d(
        qp.unsqueeze(0), wf, groups=B).squeeze(0)  # (B, T)
    return out
