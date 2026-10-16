"""
v_diversion_and_resale.py
=========================
Estimation framework for smuggled AI compute reaching China.

Two independent estimates of the same underlying quantity (total smuggled
chips reaching China), combined via weighted blend:

  1. Diversion (export-side): chips redirected toward China, observed
     primarily through indictments, investigations, and trade anomalies.
     Built from 9 case-level anchors, then scaled by detection rate.

  2. Resale (China-side): chips flowing through China's gray-market
     distribution network, modeled structurally as N_vendors x volume
     per vendor, then scaled by 1/marketplace_fraction to account for
     non-marketplace channels (direct deals, etc.). Based on WSJ/NYT
     reporting on marketplace sellers, with marketplace modeling
     approach influenced by CNAS (2024).

The two sides are NOT additive — they estimate the same thing from
different vantage points. Combined via a 60/40 weighted average per MC
draw, tilted toward diversion (case-anchored, less sensitive to
marketplace extrapolation).
"""

import csv

import numpy as np
from scipy import stats

N   = 200_000
RNG = np.random.default_rng(42)

# ── Helpers ──────────────────────────────────────────────────────────────────
def lognormal_from_ci(lo, hi, credibility=90):
    """Draw N samples from a lognormal whose [credibility]% CI is [lo, hi]."""
    z  = stats.norm.ppf(0.5 + credibility / 200)
    mu = (np.log(lo) + np.log(hi)) / 2
    sg = (np.log(hi) - np.log(lo)) / (2 * z)
    return RNG.lognormal(mu, sg, N)

def pct(arr):
    return np.percentile(arr, [5, 50, 95])

def lnparams(lo, hi, credibility=90):
    """Return (mu, sigma) for lognormal with given CI."""
    z = stats.norm.ppf(0.5 + credibility / 200)
    return (np.log(lo) + np.log(hi)) / 2, (np.log(hi) - np.log(lo)) / (2 * z)

# ── Chip conversion ──────────────────────────────────────────────────────────
H100E = {"A100": 0.315, "H100": 1.0, "H200": 1.0, "H800": 1.0, "B200": 2.527}
PRICE = {"A100": 12_000, "H100": 25_000, "H200": 25_000, "H800": 30_000, "B200": 37_000}


# ════════════════════════════════════════════════════════════════════════════
#  DIVERSION SIDE: export-side cases
# ════════════════════════════════════════════════════════════════════════════
# For each case we estimate chips allegedly diverted, with per-case
# uncertainty about quantities and chip mix. We then apply two global
# scaling parameters:
#   (a) accuracy: are the allegations correct / did chips reach China?
#   (b) detection rate: what fraction of all diversion do we observe?

# ── Case D1: Supermicro / Liaw indictment ────────────────────────────────
# $2.5B in servers sold to Company-1 (SE Asia). Indictment says
# "substantial portion" diverted to China, not all.
# Source: US v. Liaw, 1:26-cr-00100-ER (SDNY), filed Mar 17, 2026
# $2.5B total to Company-1, but uncertain how much reached China.
# Indictment says "substantial portion" / "most". Range $1.25B-$2.5B
# captures both the "how much was diverted" and "how much reached China."
d1_diverted_value = lognormal_from_ci(1.25e9, 2.5e9)
# Per-GPU price range anchored to indictment data: 3 data points in ¶20 at
# $300k per 8-GPU B200 server ($37.5k/GPU); H-class bulk at ~$200k/server
# ($25k/GPU; Hao Global plea had H100 at $22.5k, H200 at $23-24k bulk).
d1_price_per_gpu = lognormal_from_ci(25_000, 40_000)
d1_chips = d1_diverted_value / d1_price_per_gpu
# Chip mix varies by year: 2024 was mostly H100/H200 (B200 just emerging);
# 2025 was heavily B200 ($510M sprint Apr-May explicitly "B200 and H200 GPUs",
# "854 b200 labels", "512 x B200 by Feb").
d1_b200_2024 = np.clip(RNG.normal(0.10, 0.05, N), 0, 1)   # B200 barely available
d1_b200_2025 = np.clip(RNG.normal(0.50, 0.10, N), 0, 1)   # sprint was B200-dominated
# Blended fraction for total H100e (used in per-source output)
# Q4 FY24 alone (Apr-Jun 2024) was $99.7M; activity ramped through 2024
# before the dominant Apr-May 2025 $510M sprint. Chip share in 2024
# ~0.28 lands ~20% of H100e in 2024 (2025 chips weigh more via B200).
d1_frac_2024 = np.clip(RNG.normal(0.28, 0.08, N), 0, 1)
d1_b200 = d1_frac_2024 * d1_b200_2024 + (1 - d1_frac_2024) * d1_b200_2025
d1_h100e = d1_chips * (d1_b200 * H100E["B200"] + (1 - d1_b200) * H100E["H100"])
# Per-year H100e (used in year-split section)
d1_h100e_2024 = d1_chips * d1_frac_2024 * (
    d1_b200_2024 * H100E["B200"] + (1 - d1_b200_2024) * H100E["H100"])
d1_h100e_2025 = d1_chips * (1 - d1_frac_2024) * (
    d1_b200_2025 * H100E["B200"] + (1 - d1_b200_2025) * H100E["H100"])

# ── Case D2: Megaspeed — alleged 50k chip gap ───────────────────────────
# 136k imported via Singapore minus ~86k accounted for at data centers.
# Source: Bloomberg, Oct/Dec 2025
d2_chips = RNG.uniform(0, 50_000, N)
# Jan-Nov 2025: B200 availability ramped quickly; Bloomberg reports Megaspeed
# ordered heavily during ban period. $2.4B of unaccounted-for chips were
# Bianca boards (GB200/GB300) — which are exclusively Blackwell — so much
# of the gap is B200. Center at ~80% B200 by chip count.
d2_b200 = np.clip(RNG.normal(0.80, 0.08, N), 0, 1)
d2_h100e = d2_chips * (d2_b200 * H100E["B200"] + (1 - d2_b200) * H100E["H100"])

# ── Case D3: Aperia / Wei — $250M in Dell+Supermicro servers via Singapore ──
# Two defendants (Wei, Woon, Lim). BT Singapore source does not specify chip
# type; we model an A100/H100 mix (H100 from Supermicro AI servers, A100 the
# older-generation alternative for Dell servers).
# Source: Singapore charges, Feb-Mar 2025
d3_dollar = lognormal_from_ci(200e6, 300e6)
d3_price_per_gpu = lognormal_from_ci(12_000, 30_000)   # A100-to-H100 range
d3_chips = d3_dollar / d3_price_per_gpu
d3_h100_frac = np.clip(RNG.normal(0.50, 0.15, N), 0, 1)
d3_h100e = d3_chips * (d3_h100_frac * H100E["H100"]
                       + (1 - d3_h100_frac) * H100E["A100"])

# ── Case D4: Li Ming / Luxuriate Your Life — $140M in SMCI servers ──────
# Separate scheme from Aperia, same investigation sweep.
# Source: Singapore charges, Feb-Mar 2025
d4_dollar = lognormal_from_ci(100e6, 180e6)
# H-class bulk Supermicro servers at ~$200k/server ($25k/GPU); Hao Global plea
# had H100 at $22.5k, H200 at $23-24k bulk. Narrower range than Supermicro D1
# because no B200 component in this case.
d4_price_per_gpu = lognormal_from_ci(20_000, 30_000)
d4_chips = d4_dollar / d4_price_per_gpu
d4_h100e = d4_chips * H100E["H100"]   # assume H100-class

# ── Case D5: Operation Gatekeeper / Hao Global ──────────────────────────
# 7,032 GPUs in purchase agreement; $53M paid; ~1,776 likely reached China.
# 480 H100 confirmed + 1,296 H200 likely via Canada.
# Source: US v. Hao Global, 4:25-cr-00510 (SDTX), plea Oct 10, 2025
d5_h100_chips = lognormal_from_ci(400, 600)       # 480 confirmed
d5_h200_chips = lognormal_from_ci(500, 1_500)     # 1,296 likely, uncertain
d5_h100e = d5_h100_chips * H100E["H100"] + d5_h200_chips * H100E["H200"]

# ── Case D6: Janford / Ho (Tampa) — 400 A100s confirmed shipped ─────────
# Plus attempted H100/H200 shipments seized. Only A100s confirmed delivered.
# Source: DOJ, M.D. Florida, Nov 2025
d6_chips = lognormal_from_ci(350, 500)
d6_h100e = d6_chips * H100E["A100"]

# ── Case D7: Other smaller cases ────────────────────────────────────────
# Leslie Zhou, college student, Shenzhen merchant, etc.
# Aggregate: maybe 500-5,000 chips total, mostly H100/A100
d7_chips = lognormal_from_ci(500, 5_000)
d7_h100e = d7_chips * H100E["H100"]

# ── Case D8: ALX Solutions / Geng & Yang (CDCA) ───────────────────────
# El Monte, CA company purchasing from Supermicro and MiTAC, Oct 2022–Jul 2025.
# Source: Criminal complaint 2:25-mj-04821-DUTY, filed Aug 1, 2025
# Complaint says "approximately 205 NVIDIA H100 GPUs" from Supermicro
# (Aug 2023–Jul 2024), but the $28.4M invoice lists 106 units of 8-GPU
# HGX baseboards at $208k each. If "205" counts baseboards, that's
# ~1,640 H100 chips; if individual chips, ~205. We model the range.
# Also: 40 H200s (separate invoice), plus RTX 4090/5090/A100 in smaller
# quantities (low H100e contribution, omitted).
# NOTE: Zheng/Kelly/English (N.D. Georgia, Mar 2026) is NOT modeled here —
# zero chips were delivered (conspiracy charge; both attempts blocked by
# Supermicro/Nvidia compliance).
d8_h100_chips = lognormal_from_ci(205, 1_640, credibility=90)  # key ambiguity
d8_h200_chips = lognormal_from_ci(30, 60)                       # 40 on invoice
d8_h100e = d8_h100_chips * H100E["H100"] + d8_h200_chips * H100E["H200"]

# ── Case D9: William & Mike (The Information, Aug 2024) ────────────────
# Two brokers profiled by The Information based on interviews:
# - William (Malaysia): arranged single order of 2,400 H100s ($120M) for
#   Chinese appliance company. Still actively doing deals.
# - Mike (Hong Kong): warehouse held 4,800 H100s when visited (May 2024).
#   Valued at ~$180M, sold for ~$230M. Uses shell companies.
# These are point-in-time snapshots, not annual totals. Both are "Both"
# evidence: they describe both the diversion setup (shell companies,
# distributor relationships) and the resale to Chinese buyers.
# Conservative: model just the observed quantities, not annual throughput.
d9_william_chips = lognormal_from_ci(2_000, 3_000)   # single order ~2,400
d9_mike_chips = lognormal_from_ci(4_000, 6_000)      # warehouse snapshot ~4,800
d9_h100e = (d9_william_chips + d9_mike_chips) * H100E["H100"]  # all H100

# ── Year splits ────────────────────────────────────────────────────────
# Each case spans a time window; allocate to 2024 vs 2025 with uncertainty.

# D1 Supermicro: year split and B200 fractions handled above (per-year H100e).

# D2 Megaspeed: gap is Jan-Nov 2025 specifically.
d2_frac_2024 = np.zeros(N)

# D3 Aperia: Nov 2023 – Feb 2025. Bulk probably in 2024.
d3_frac_2024 = np.clip(RNG.normal(0.70, 0.10, N), 0, 1)

# D4 Li Ming: fraud alleged in 2023. Mostly pre-2025.
d4_frac_2024 = np.clip(RNG.normal(0.80, 0.10, N), 0, 1)

# D5 Gatekeeper: Oct 2024 – May 2025. First shipment Nov 2024 (480 H100s).
# Bulk of H200 shipments in early 2025 (Feb-Mar).
d5_frac_2024 = np.clip(RNG.normal(0.30, 0.10, N), 0, 1)

# D6 Janford: 400 A100s shipped Oct 2024 – Jan 2025. Straddles years.
d6_frac_2024 = np.clip(RNG.normal(0.60, 0.15, N), 0, 1)

# D7 Other small: various. Rough even split.
d7_frac_2024 = np.clip(RNG.normal(0.50, 0.15, N), 0, 1)

# D8 ALX Solutions: Oct 2022–Jul 2025. Supermicro purchases Aug 2023–Jul 2024;
# H200 invoice Jan 2024; post-seizure shipments Mar-Apr 2025.
# Bulk of value in the Supermicro H100 purchases (2023-2024).
# NOTE: "2024" bucket here means pre-2025 (model only tracks 2024/2025).
# Activity spans back to Oct 2022, but nearly all chip value is the
# Supermicro H100s (Aug 2023–Jul 2024) and H200 invoice (Jan 2024).
# Only the 9 small UPS shipments (Mar-Apr 2025) and phone-chat A100/Quanta
# activity fall in 2025 — maybe ~5-15% of total volume.
d8_frac_2024 = np.clip(RNG.normal(0.90, 0.05, N), 0, 1)

# D9 William/Mike: Information article published Aug 2024, describing activity
# in May-Aug 2024. Point-in-time snapshots, all in 2024.
d9_frac_2024 = np.ones(N)

# Observed diversion by year
diversion_2024_observed = (d1_h100e_2024 + d2_h100e * d2_frac_2024
                           + d3_h100e * d3_frac_2024 + d4_h100e * d4_frac_2024
                           + d5_h100e * d5_frac_2024 + d6_h100e * d6_frac_2024
                           + d7_h100e * d7_frac_2024 + d8_h100e * d8_frac_2024
                           + d9_h100e * d9_frac_2024)
diversion_2025_observed = (d1_h100e_2025 + d2_h100e * (1 - d2_frac_2024)
                           + d3_h100e * (1 - d3_frac_2024) + d4_h100e * (1 - d4_frac_2024)
                           + d5_h100e * (1 - d5_frac_2024) + d6_h100e * (1 - d6_frac_2024)
                           + d7_h100e * (1 - d7_frac_2024) + d8_h100e * (1 - d8_frac_2024)
                           + d9_h100e * (1 - d9_frac_2024))
diversion_observed = diversion_2024_observed + diversion_2025_observed

# ── Diversion scaling parameters ────────────────────────────────────────

# (a) Allegation accuracy / success rate
# Are the allegations true? Did attempted diversions succeed?
# Most major cases have strong evidence (plea deals, indictments with
# surveillance footage, seized shipments). But not all proven in court.
# Already partially handled per-case (uncertain dollar ranges, chip mixes,
# year splits), so this is a residual correction for cases that may be overstated.
diversion_accuracy = RNG.beta(16, 2, N)  # median ~90%

# (b) Detection rate: P(diversion is detected | diversion occurred)
# Key multiplier. "Detection" here means eventually surfacing publicly
# (indictment, investigative reporting, customs analysis) — not seizure.
# Bounds anchored on fraud/illicit-finance detection literature:
# Dyck/Morse/Zingales (2024) estimate ~33% of US corporate frauds detected;
# Kacperczyk/Pagnotta (2019) estimate ~15% of insider trading detected;
# FATF/Silverado put sophisticated sanctions-evasion detection well under 50%.
# 10-80% brackets this evidence with room for the possibility that chip
# diversion is more visible than typical covert finance (large dollar values,
# concentrated hardware supply chain).
diversion_detection = lognormal_from_ci(0.10, 0.80, credibility=90)
diversion_detection = np.clip(diversion_detection, 0.05, 1.0)

# Total diversion estimate (H100e) — cumulative and per-year
diversion_total = diversion_observed * diversion_accuracy / diversion_detection
diversion_2024_total = diversion_2024_observed * diversion_accuracy / diversion_detection
diversion_2025_total = diversion_2025_observed * diversion_accuracy / diversion_detection


# ════════════════════════════════════════════════════════════════════════════
#  RESALE SIDE: China-side marketplace model
# ════════════════════════════════════════════════════════════════════════════
# Model the marketplace structure directly:
#   Marketplace chips = N_vendors × chips_per_vendor_per_year
#
# Vendor count: WSJ (Jul 2024) found 70+ sellers on Chinese platforms
# (Baidu Tieba, JD.com, Taobao). Lognormal(38, 150) at 90% CI for 2024,
# following CNAS (2024) parameterization of this evidence.
#
# Chips per vendor per year: anchored to known dealers.
#   William (~1,200/yr, small), Mike (~2,400/yr, medium),
#   Hao Global peak (~6,000/yr), Gate of Era (~40k+/yr annualized — outlier).
# Two regimes: base (non-peak periods) and ban-period (Apr-Jul 2025 surge,
# with Gate of Era-scale vendors at p99). No need to cap: the diversion and
# resale sides are not additive, so large vendors appearing in both estimates
# do not cause double-counting.
#
# IMPORTANT: This captures marketplace-observable activity only.
# Direct company-to-company deals and government procurement are NOT included.
# Those channels are captured by the diversion side. marketplace_fraction
# scales the marketplace observation up to total compute reaching China.

# ── Hierarchical vendor model ────────────────────────────────────────
RHO_VENDOR = 0.5  # intra-class correlation: how much vendors move together
                   # (market-wide shocks vs individual variation)

def marketplace_model(mu_v, sg_v, mu_a, sg_a, months=12):
    """
    Hierarchical vendor model for China's gray-market chip marketplace.
    Each MC draw: sample N vendors, each with correlated annual volume.
    RHO_VENDOR controls market-wide vs individual variation in log-volume.
    """
    n_vendors = np.maximum(np.round(RNG.lognormal(mu_v, sg_v, N)).astype(np.int64), 1)
    total = int(n_vendors.sum())
    sg_market = sg_a * np.sqrt(RHO_VENDOR)
    sg_indiv = sg_a * np.sqrt(1 - RHO_VENDOR)
    market_shocks = RNG.normal(0, sg_market, N) if sg_market > 0 else np.zeros(N)
    indiv_z = RNG.normal(0, sg_indiv, total) if sg_indiv > 0 else np.zeros(total)
    market_repeated = np.repeat(market_shocks, n_vendors)
    all_vols = np.exp(mu_a + market_repeated + indiv_z)
    starts = np.concatenate([[0], np.cumsum(n_vendors[:-1])])
    return np.add.reduceat(all_vols, starts) * (months / 12)

# ── Per-vendor volume parameters by period ────────────────────────────
# Gate of Era's ~40k/yr annualized rate was a brief peak during the ban
# quarter (Apr-Jul 2025). Other periods use a lower upper bound reflecting
# typical large vendors (Hao Global peak ~6k/yr, Mike ~2.4k/yr).
MU_A_BASE, SG_A_BASE = lnparams(204, 12_000, credibility=95)  # non-peak periods
MU_A_BAN,  SG_A_BAN  = lnparams(130, 40_000, credibility=98)  # ban: Gate of Era at p99

# ── Resale 2024 ──────────────────────────────────────────────────────
# Full calendar year. No Gate of Era-scale activity in 2024.
MU_V_24, SG_V_24 = lnparams(38, 150, credibility=90)    # WSJ: 70+ found

resale_2024_chips = marketplace_model(MU_V_24, SG_V_24, MU_A_BASE, SG_A_BASE, months=12)

# Chip mix 2024: beta(5,1) ≈ 83% H100, rest A100 (before B200 widespread)
_r24_h100_frac = RNG.beta(5, 1, N)
resale_2024_h100e = resale_2024_chips * (
    _r24_h100_frac * H100E["H100"] + (1 - _r24_h100_frac) * H100E["A100"])

# ── Resale 2025 (period-based) ────────────────────────────────────────
# Marketplace articles are from mid-2024. For 2025 we use a wider vendor
# range [20, 200] — reflecting uncertainty about market evolution:
# more vendors drawn by higher margins vs enforcement pressure.
# Year-level shock correlates all three sub-periods.
SG_YEAR = 0.5
resale_year_shock = np.exp(RNG.normal(0, SG_YEAR, N))

MU_V_25, SG_V_25 = lnparams(20, 200, credibility=90)

# Pre-ban (Jan 1 – Apr 8, 3.25 months): baseline volumes
resale_pre = marketplace_model(MU_V_25, SG_V_25, MU_A_BASE, SG_A_BASE, months=3.25) * resale_year_shock

# Ban period (Apr 9 – Jul 27, 3.50 months): peak marketplace activity
# Gate of Era-scale vendors active. FT found >$1B — calibration check.
resale_ban = marketplace_model(MU_V_25, SG_V_25, MU_A_BAN, SG_A_BAN, months=3.50) * resale_year_shock

# Post-ban (Jul 28 – Dec 31, 5.25 months): back to baseline volumes
resale_post = marketplace_model(MU_V_25, SG_V_25, MU_A_BASE, SG_A_BASE, months=5.25) * resale_year_shock

resale_2025_chips = resale_pre + resale_ban + resale_post

# Chip mixes by 2025 period
_b200_pre = RNG.uniform(0.12, 0.22, N)  # B200 still ramping in Q1 2025; narrow band
_b200_ban = RNG.uniform(0.45, 0.65, N)  # FT: "dominant"; Bloomberg: "more than half"
resale_pre_h100e = resale_pre * (_b200_pre * H100E["B200"] + (1 - _b200_pre) * H100E["H100"])
resale_ban_h100e = resale_ban * (_b200_ban * H100E["B200"] + (1 - _b200_ban) * H100E["H100"])
resale_post_h100e = resale_post * (_b200_ban * H100E["B200"] + (1 - _b200_ban) * H100E["H100"])
resale_2025_h100e = resale_pre_h100e + resale_ban_h100e + resale_post_h100e

# ── Marketplace fraction ────────────────────────────────────────────
# Not all smuggled chips pass through the observable gray market.
# Some go through direct company-to-company deals that bypass marketplace
# vendors entirely. But the boundary is blurry: even the Supermicro/Liaw
# pipeline used brokers that likely overlap with marketplace sellers
# (gray market provides plausible deniability). So this fraction is high.
marketplace_fraction = RNG.beta(8, 2, N)  # median ~82%

# ── Resale totals (H100e) ────────────────────────────────────────────
resale_2024_total = resale_2024_h100e / marketplace_fraction
resale_2025_total = resale_2025_h100e / marketplace_fraction
resale_total = resale_2024_total + resale_2025_total

# ── Calibration values (for output) ─────────────────────────────────
# FT ban-quarter: model's ban-period chip estimate × gray-market price
_ban_blended_price = _b200_ban * 55_000 + (1 - _b200_ban) * 30_000
resale_ban_value = resale_ban * _ban_blended_price

# Percentile calibration: where do William/Mike fall in the base distribution?
from scipy.stats import lognorm as _lognorm_dist
_w_pct = _lognorm_dist.cdf(2_400, s=SG_A_BASE, scale=np.exp(MU_A_BASE)) * 100
_m_pct = _lognorm_dist.cdf(4_800, s=SG_A_BASE, scale=np.exp(MU_A_BASE)) * 100


# ════════════════════════════════════════════════════════════════════════════
#  COMBINING TWO ESTIMATES OF THE SAME QUANTITY
# ════════════════════════════════════════════════════════════════════════════
# Both sides attempt to estimate the *total* smuggled compute reaching China.
# Diversion scales up observed export-side cases by 1/detection_rate.
# Resale models marketplace vendors then scales by 1/marketplace_fraction.
# They are two views of the same underlying quantity, not additive streams.
#
# Some evidence is shared across streams (e.g., The Information describes
# both the shell-company diversion setup and the resale to Chinese buyers),
# so the estimates are positively correlated — combining them should NOT
# narrow uncertainty as much as two independent estimates would.
#
# Approach: 60/40 weighted average per MC draw (60% diversion, 40% resale).
# Tilts toward diversion: case-anchored, less sensitive to marketplace
# extrapolation, and the resale upper tail is harder to discipline empirically.

w_diversion = 0.6  # 60% diversion / 40% resale

# Blend: simple average of the two estimates per MC draw.
# Always falls between diversion and resale per draw.
combined = w_diversion * diversion_total + (1 - w_diversion) * resale_total

# Per-year: apply the same weighting
combined_2024 = w_diversion * diversion_2024_total + (1 - w_diversion) * resale_2024_total
combined_2025 = w_diversion * diversion_2025_total + (1 - w_diversion) * resale_2025_total


# ════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ════════════════════════════════════════════════════════════════════════════
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("Diversion & Resale Model")
print("=" * 70)

# Per-source scaled diversion estimates
diversion_scale = diversion_accuracy / diversion_detection
d1_scaled = d1_h100e * diversion_scale
d2_scaled = d2_h100e * diversion_scale
d3_scaled = d3_h100e * diversion_scale
d4_scaled = d4_h100e * diversion_scale
d5_scaled = d5_h100e * diversion_scale
d6_scaled = d6_h100e * diversion_scale
d7_scaled = d7_h100e * diversion_scale
d8_scaled = d8_h100e * diversion_scale
d9_scaled = d9_h100e * diversion_scale

# ── DIVERSION SIDE OUTPUT ────────────────────────────────────────────
print("\n--- Diversion side: observed cases (H100e, before scaling) ---")
print("  Direct chip estimates from indictments, investigations, trade data.")
cases_observed = [
    ("D1: Supermicro/Liaw",    d1_h100e, "$1.25-2.5B diverted (\"substantial portion\" of $2.5B to Company-1)"),
    ("D2: Megaspeed gap",      d2_h100e, "Uniform(0, 50k) chip gap between imports and accounted-for"),
    ("D3: Aperia/Wei",         d3_h100e, "$200-300M Dell+Supermicro servers via Singapore; A100/H100 mix"),
    ("D4: Li Ming",            d4_h100e, "$100-180M SMCI servers; separate scheme, same investigation"),
    ("D5: Gatekeeper/Hao",     d5_h100e, "480 H100 confirmed + ~1,296 H200 likely via Canada"),
    ("D6: Janford/Ho",         d6_h100e, "~400 A100s confirmed shipped (Tampa)"),
    ("D7: Other small",        d7_h100e, "Zhou, Shenzhen merchant, etc. — aggregate"),
    ("D8: ALX Solutions/Geng", d8_h100e, "205-1,640 H100s from Supermicro + 40 H200s (CDCA complaint)"),
    ("D9: William+Mike",       d9_h100e, "2,400 + 4,800 H100s (Information, Aug 2024; point-in-time)"),
]
for label, arr, note in cases_observed:
    p5, p50, p95 = pct(arr)
    print(f"  {label:<25s}  median {p50:>10,.0f}  [{p5:>10,.0f} - {p95:>10,.0f}]")
    print(f"    {note}")
print(f"\n  {'Total observed':<25s}  median {pct(diversion_observed)[1]:>10,.0f}"
      f"  [{pct(diversion_observed)[0]:>10,.0f} - {pct(diversion_observed)[2]:>10,.0f}]")

print("\n--- Diversion side: scaling ---")
print("  observed × accuracy / detection_rate → estimate of ALL diverted chips")
print("  accuracy: are allegations correct / did chips reach China?")
print("  detection_rate: what fraction of all diversion do we observe?")
print("    (BIS has ~1 officer for all of SE Asia; cases take months/years to surface)")
for label, arr in [
    ("Accuracy",     diversion_accuracy),
    ("Detection rate",    diversion_detection),
]:
    p5, p50, p95 = pct(arr)
    print(f"  {label:<25s}  median {p50:>8.3f}  [{p5:>8.3f} - {p95:>8.3f}]")
p5, p50, p95 = pct(diversion_scale)
print(f"  {'Effective multiplier':<25s}  median {p50:>8.2f}x [{p5:>8.2f}x - {p95:>8.2f}x]")

print("\n--- Diversion side: scaled per-source (H100e) ---")
print("  Each case scaled to estimate total smuggling.")
print("  These are NOT additive with resale — both estimate the same quantity.")
for label, arr in [
    ("D1: Supermicro/Liaw",    d1_scaled),
    ("D2: Megaspeed gap",      d2_scaled),
    ("D3: Aperia/Wei",         d3_scaled),
    ("D4: Li Ming",            d4_scaled),
    ("D5: Gatekeeper/Hao",     d5_scaled),
    ("D6: Janford/Ho",         d6_scaled),
    ("D7: Other small",        d7_scaled),
    ("D8: ALX Solutions/Geng", d8_scaled),
    ("D9: William+Mike",       d9_scaled),
]:
    p5, p50, p95 = pct(arr)
    print(f"    {label:<25s}  median {p50:>10,.0f}  [{p5:>10,.0f} - {p95:>10,.0f}]")
p5, p50, p95 = pct(diversion_total)
print(f"    {'Sum (diversion est.)':<25s}  median {p50:>10,.0f}  [{p5:>10,.0f} - {p95:>10,.0f}]")

print("\n--- Diversion side: per-case year allocation (observed H100e) ---")
print("  How each case's chips split between 2024 and 2025.")
div_year_cases = [
    ("D1: Supermicro/Liaw", d1_h100e_2024, d1_h100e_2025,
     "~10% B200 in 2024, ~50% B200 in 2025 ($510M sprint)"),
    ("D2: Megaspeed gap",   d2_h100e * d2_frac_2024, d2_h100e * (1 - d2_frac_2024),
     "Gap is Jan-Nov 2025 specifically"),
    ("D3: Aperia/Wei",      d3_h100e * d3_frac_2024, d3_h100e * (1 - d3_frac_2024),
     "Nov 2023 - Feb 2025; bulk in 2024"),
    ("D4: Li Ming",         d4_h100e * d4_frac_2024, d4_h100e * (1 - d4_frac_2024),
     "Fraud alleged from 2023; mostly pre-2025"),
    ("D5: Gatekeeper/Hao",  d5_h100e * d5_frac_2024, d5_h100e * (1 - d5_frac_2024),
     "First shipment Nov 2024; bulk H200s in early 2025"),
    ("D6: Janford/Ho",      d6_h100e * d6_frac_2024, d6_h100e * (1 - d6_frac_2024),
     "A100s shipped Oct 2024 - Jan 2025; straddles year"),
    ("D7: Other small",     d7_h100e * d7_frac_2024, d7_h100e * (1 - d7_frac_2024),
     "Various; rough even split"),
    ("D8: ALX Solutions",   d8_h100e * d8_frac_2024, d8_h100e * (1 - d8_frac_2024),
     "Supermicro purchases Aug 2023-Jul 2024; bulk in 2024"),
    ("D9: William+Mike",   d9_h100e * d9_frac_2024, d9_h100e * (1 - d9_frac_2024),
     "Information article Aug 2024; all 2024"),
]
print(f"  {'Case':<25s}  {'2024 H100e':>12s}  {'2025 H100e':>12s}  Note")
for label, h24, h25, note in div_year_cases:
    m24, m25 = np.median(h24), np.median(h25)
    print(f"  {label:<25s}  {m24:>12,.0f}  {m25:>12,.0f}  {note}")
m_2024_tot = np.median(diversion_2024_observed)
m_2025_tot = np.median(diversion_2025_observed)
print(f"  {'Total observed':<25s}  {m_2024_tot/(m_2024_tot+m_2025_tot):>9.0%}"
      f"  {m_2024_tot:>12,.0f}  {m_2025_tot:>12,.0f}")

# ── RESALE SIDE OUTPUT ───────────────────────────────────────────────
print("\n--- Resale side: marketplace model ---")
print("  Structural model: N_vendors x chips_per_vendor_per_year / marketplace_fraction")
print("  Vendors: WSJ found 70+ on Chinese platforms (Baidu Tieba, JD, Taobao).")
print("  Volume anchors (point-in-time, not annual rates):")
print("    William: single order of 2,400 H100s ($120M) for one client (Information, Aug 2024)")
print("    Mike: 4,800 H100s in warehouse at time of visit ($180-230M) (Information, May 2024)")
print("    Gate of Era: ~40k/yr annualized rate during ban quarter (FT, 2025)")
print("  Uses hierarchical model: vendors share market-wide shocks (rho=0.5)")
print("  but also have individual variation.")
print("  Marketplace fraction scales up for non-marketplace channels (direct deals, etc.).")
print()
_v24_med = np.exp((np.log(38) + np.log(150)) / 2)
_v25_med = np.exp((np.log(20) + np.log(200)) / 2)
print(f"  Parameters:")
print(f"    Vendors 2024:   [38 – 150], median ~{_v24_med:.0f}  (WSJ: 70+ found; CNAS parameterization)")
print(f"    Vendors 2025:   [20 – 200], median ~{_v25_med:.0f}  (wider: uncertain market evolution)")
_a_base_med = np.exp((np.log(204) + np.log(12_000)) / 2)
_a_ban_med = np.exp((np.log(130) + np.log(40_000)) / 2)
print(f"    Chips/vendor/yr (base):  [204 – 12,000], median ~{_a_base_med:.0f}  (2024, pre/post-ban)")
print(f"    Chips/vendor/yr (ban):   [130 – 40,000], median ~{_a_ban_med:.0f}  (ban quarter; Gate of Era at p99)")
print(f"    Intra-class rho: {RHO_VENDOR}  (vendor correlation)")
print(f"    Year shock 2025: sigma={SG_YEAR}  (correlated across 2025 sub-periods)")
_mf_med = np.median(marketplace_fraction)
print(f"    Marketplace fraction: Beta(8,2), median ~{_mf_med:.0%}  (scales up for direct deals)")
print()
print(f"  2024 (full year):")
p5, p50, p95 = pct(resale_2024_chips)
print(f"    Chips:  median {p50:>10,.0f}  [{p5:>10,.0f} - {p95:>10,.0f}]")
p5, p50, p95 = pct(resale_2024_total)
print(f"    H100e:  median {p50:>10,.0f}  [{p5:>10,.0f} - {p95:>10,.0f}]")
print(f"  2025 (period-based: pre-ban / ban / post-ban):")
for plabel, parr in [
    ("Pre-ban  (Jan-Apr,  3.25mo)", resale_pre_h100e),
    ("Ban      (Apr-Jul,  3.50mo)", resale_ban_h100e),
    ("Post-ban (Jul-Dec,  5.25mo)", resale_post_h100e),
]:
    p5, p50, p95 = pct(parr)
    print(f"    {plabel:<30s}  median {p50:>10,.0f}  [{p5:>10,.0f} - {p95:>10,.0f}]")
p5, p50, p95 = pct(resale_2025_total)
print(f"    {'Total 2025 H100e':<30s}  median {p50:>10,.0f}  [{p5:>10,.0f} - {p95:>10,.0f}]")
p5, p50, p95 = pct(resale_total)
print(f"  {'Cumulative H100e':<27s}  median {p50:>10,.0f}  [{p5:>10,.0f} - {p95:>10,.0f}]")

# Calibration checks
print()
print(f"  Calibration checks:")
p5, p50, p95 = pct(resale_ban_value)
print(f"    FT ban-quarter: model ban-period → median ${p50/1e9:.1f}B"
      f"  [{p5/1e9:.1f}B - {p95/1e9:.1f}B]")
print(f"      FT reported >$1B in Apr-Jul 2025 — {'consistent' if p50 >= 0.8e9 else 'LOW'}")
print(f"    Implied chips/vendor 2024: median ~{_a_base_med:,.0f}/yr (parametric)")
print(f"      William single order 2,400 = p{_w_pct:.0f}; Mike inventory 4,800 = p{_m_pct:.0f}")

# ── COMBINATION OUTPUT ───────────────────────────────────────────────
print("\n--- Combination ---")
print("  Both sides estimate the same underlying quantity (total smuggled chips).")
print("  Diversion: scales up known cases by 1/detection_rate.")
print("  Resale: marketplace model scaled by 1/marketplace_fraction.")
print("  Combined = 0.6 * diversion + 0.4 * resale (per MC draw).")
print("  Tilt toward diversion: case-anchored, less sensitive to marketplace extrapolation.")

print("\n--- Scaled totals (H100e) — cumulative ---")
for label, arr in [
    ("Diversion total",  diversion_total),
    ("Resale total",     resale_total),
    ("Combined",         combined),
]:
    p5, p50, p95 = pct(arr)
    print(f"  {label:<25s}  median {p50:>12,.0f}  [{p5:>12,.0f} - {p95:>12,.0f}]")

print("\n--- By year (H100e) ---")
print("  2024:")
for label, arr in [
    ("  Diversion",  diversion_2024_total),
    ("  Resale",     resale_2024_total),
    ("  Combined",   combined_2024),
]:
    p5, p50, p95 = pct(arr)
    print(f"  {label:<25s}  median {p50:>12,.0f}  [{p5:>12,.0f} - {p95:>12,.0f}]")
print("  2025:")
for label, arr in [
    ("  Diversion",  diversion_2025_total),
    ("  Resale",     resale_2025_total),
    ("  Combined",   combined_2025),
]:
    p5, p50, p95 = pct(arr)
    print(f"  {label:<25s}  median {p50:>12,.0f}  [{p5:>12,.0f} - {p95:>12,.0f}]")

# ── Sanity check: what does the combined look like in chips and dollars? ──
# Approximate back-conversion using median B200 fraction
approx_b200_frac = 0.40  # rough overall
approx_h100e_per_chip = approx_b200_frac * H100E["B200"] + (1 - approx_b200_frac) * H100E["H100"]
approx_price_per_chip = approx_b200_frac * PRICE["B200"] + (1 - approx_b200_frac) * PRICE["H100"]

combined_chips_approx = combined / approx_h100e_per_chip
combined_value_approx = combined_chips_approx * approx_price_per_chip

print("\n--- Approximate combined (rough back-conversion, ~40% B200) ---")
p5, p50, p95 = pct(combined_chips_approx)
print(f"  Chips:  median {p50:>12,.0f}  [{p5:>12,.0f} - {p95:>12,.0f}]")
p5, p50, p95 = pct(combined_value_approx)
print(f"  Value:  median ${p50/1e9:>10.1f}B  [${p5/1e9:>10.1f}B - ${p95/1e9:>10.1f}B]")

print()


# ════════════════════════════════════════════════════════════════════════════
#  CSV EXPORT — quarterly cumulative by chip type, [Owners] timelines format
# ════════════════════════════════════════════════════════════════════════════
# For each chip type (A100, H100, H200, B200), produces 8 cumulative rows
# (one per quarter, 2024 Q1 through 2025 Q4). Plus 8 "Nvidia total" rows
# summing across chip types per draw. Each row reports H100e (median, p5,
# p95), chip count (median, p5, p95), and total TDP. Owner = "China(smuggled)".

TDP = {"A100": 400, "H100": 700, "H200": 700, "B200": 1000}

# 2024 quarterly weights — revenue ramp through the year
Q24_FRACS = np.array([0.700, 0.913, 1.076, 1.312])
Q24_FRACS = Q24_FRACS / Q24_FRACS.sum()

# 2025 diversion quarterly weights — heavier mid-year (matches Liaw sprint timing)
Q25_DIV_FRACS = np.array([0.18, 0.30, 0.28, 0.24])
Q25_DIV_FRACS = Q25_DIV_FRACS / Q25_DIV_FRACS.sum()

# Resale 2025 sub-period to calendar quarter: each row is the share of
# pre-ban / ban / post-ban months falling in each calendar quarter.
RP_PRE  = np.array([3.00/3.25, 0.25/3.25, 0,         0])
RP_BAN  = np.array([0,         2.75/3.50, 0.75/3.50, 0])
RP_POST = np.array([0,         0,         2.25/5.25, 3.00/5.25])

# ── Diversion side: per-draw chip-count arrays by chip type and year ────
# Observed (pre-scaling) chip counts. Diversion scaling applied later.

# D1 Supermicro/Liaw: H100 + B200 by year
d1_chips_2024 = d1_chips * d1_frac_2024
d1_chips_2025 = d1_chips * (1 - d1_frac_2024)
d1_b200_chips_2024 = d1_chips_2024 * d1_b200_2024
d1_h100_chips_2024 = d1_chips_2024 * (1 - d1_b200_2024)
d1_b200_chips_2025 = d1_chips_2025 * d1_b200_2025
d1_h100_chips_2025 = d1_chips_2025 * (1 - d1_b200_2025)

# D2 Megaspeed: B200 + H100, all 2025
d2_b200_chips_2025 = d2_chips * d2_b200
d2_h100_chips_2025 = d2_chips * (1 - d2_b200)

# D3 Aperia: H100 + A100 by year
d3_chips_2024 = d3_chips * d3_frac_2024
d3_chips_2025 = d3_chips * (1 - d3_frac_2024)
d3_h100_chips_2024 = d3_chips_2024 * d3_h100_frac
d3_a100_chips_2024 = d3_chips_2024 * (1 - d3_h100_frac)
d3_h100_chips_2025 = d3_chips_2025 * d3_h100_frac
d3_a100_chips_2025 = d3_chips_2025 * (1 - d3_h100_frac)

# D4 Li Ming: H100 by year
d4_h100_chips_2024 = d4_chips * d4_frac_2024
d4_h100_chips_2025 = d4_chips * (1 - d4_frac_2024)

# D5 Gatekeeper: H100 + H200 by year
d5_h100_chips_2024 = d5_h100_chips * d5_frac_2024
d5_h100_chips_2025 = d5_h100_chips * (1 - d5_frac_2024)
d5_h200_chips_2024 = d5_h200_chips * d5_frac_2024
d5_h200_chips_2025 = d5_h200_chips * (1 - d5_frac_2024)

# D6 Janford: A100 by year
d6_a100_chips_2024 = d6_chips * d6_frac_2024
d6_a100_chips_2025 = d6_chips * (1 - d6_frac_2024)

# D7 Other small: H100 by year
d7_h100_chips_2024 = d7_chips * d7_frac_2024
d7_h100_chips_2025 = d7_chips * (1 - d7_frac_2024)

# D8 ALX: H100 + H200 by year
d8_h100_chips_2024 = d8_h100_chips * d8_frac_2024
d8_h100_chips_2025 = d8_h100_chips * (1 - d8_frac_2024)
d8_h200_chips_2024 = d8_h200_chips * d8_frac_2024
d8_h200_chips_2025 = d8_h200_chips * (1 - d8_frac_2024)

# D9 William+Mike: H100, all 2024
d9_h100_chips_2024 = d9_william_chips + d9_mike_chips

# Aggregate observed chip counts by type and year
div_2024_observed = {
    "A100": d3_a100_chips_2024 + d6_a100_chips_2024,
    "H100": (d1_h100_chips_2024 + d3_h100_chips_2024 + d4_h100_chips_2024
             + d5_h100_chips_2024 + d7_h100_chips_2024 + d8_h100_chips_2024
             + d9_h100_chips_2024),
    "H200": d5_h200_chips_2024 + d8_h200_chips_2024,
    "B200": d1_b200_chips_2024,
}
div_2025_observed = {
    "A100": d3_a100_chips_2025 + d6_a100_chips_2025,
    "H100": (d1_h100_chips_2025 + d2_h100_chips_2025 + d3_h100_chips_2025
             + d4_h100_chips_2025 + d5_h100_chips_2025 + d7_h100_chips_2025
             + d8_h100_chips_2025),
    "H200": d5_h200_chips_2025 + d8_h200_chips_2025,
    "B200": d1_b200_chips_2025 + d2_b200_chips_2025,
}

# ── Resale side: per-draw chip-count arrays by chip type and period ─────
res_2024_h100 = resale_2024_chips * _r24_h100_frac
res_2024_a100 = resale_2024_chips * (1 - _r24_h100_frac)

res_pre_b200  = resale_pre  * _b200_pre
res_pre_h100  = resale_pre  * (1 - _b200_pre)
res_ban_b200  = resale_ban  * _b200_ban
res_ban_h100  = resale_ban  * (1 - _b200_ban)
res_post_b200 = resale_post * _b200_ban
res_post_h100 = resale_post * (1 - _b200_ban)

res_scale = 1.0 / marketplace_fraction
# diversion_scale (= diversion_accuracy / diversion_detection) defined above

# ── Combine into per-quarter chip-count arrays by chip type ─────────────
# Use the same 60/40 diversion/resale blend as the headline combined estimate.
chip_types = ["A100", "H100", "H200", "B200"]
quarterly = {ct: [None]*8 for ct in chip_types}

for ct in chip_types:
    div_y24 = div_2024_observed[ct]
    div_y25 = div_2025_observed[ct]
    # 2024 quarters
    for qi in range(4):
        div_q = div_y24 * Q24_FRACS[qi] * diversion_scale
        if ct == "H100":
            res_q = res_2024_h100 * Q24_FRACS[qi] * res_scale
        elif ct == "A100":
            res_q = res_2024_a100 * Q24_FRACS[qi] * res_scale
        else:
            res_q = np.zeros_like(div_q)
        quarterly[ct][qi] = w_diversion * div_q + (1 - w_diversion) * res_q
    # 2025 quarters
    for qi in range(4):
        div_q = div_y25 * Q25_DIV_FRACS[qi] * diversion_scale
        if ct == "B200":
            res_q = (res_pre_b200  * RP_PRE[qi]
                   + res_ban_b200  * RP_BAN[qi]
                   + res_post_b200 * RP_POST[qi]) * res_scale
        elif ct == "H100":
            res_q = (res_pre_h100  * RP_PRE[qi]
                   + res_ban_h100  * RP_BAN[qi]
                   + res_post_h100 * RP_POST[qi]) * res_scale
        else:
            res_q = np.zeros_like(div_q)
        quarterly[ct][qi + 4] = w_diversion * div_q + (1 - w_diversion) * res_q

# Cumulate across quarters
cumulative = {ct: [None]*8 for ct in chip_types}
for ct in chip_types:
    running = np.zeros_like(quarterly[ct][0])
    for qi in range(8):
        running = running + quarterly[ct][qi]
        cumulative[ct][qi] = running.copy()

# ── Write CSV ───────────────────────────────────────────────────────────
end_dates = ["3/31/2024", "6/30/2024", "9/30/2024", "12/31/2024",
             "3/31/2025", "6/30/2025", "9/30/2025", "12/31/2025"]
q_labels  = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024",
             "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]

header = ["Name", "Chip manufacturer", "Owner", "Chip type",
          "Start date", "End date",
          "Compute estimate in H100e (median)",
          "H100e (5th percentile)", "H100e (95th percentile)",
          "Number of Units (median)",
          "Number of Units (5th percentile)",
          "Number of Units (95th percentile)",
          "Source / Link", "Notes",
          "Last Modified By", "Last Modified", "Incomplete", "Created",
          "Total TDP (W)", "Total TDP (W) (5th percentile)",
          "Total TDP (W) (95th percentile)", "Last Modified 2"]

note = "Estimates from Epoch AI Monte Carlo model of compute smuggled to China (diversion + resale, 60/40 weighted)."
source = "https://docs.google.com/document/d/184mWLWpPNQkiLZDpAm82A-D9jgUpBnyEI9DpRoMgQNM"

rows = []
for ct in chip_types:
    for qi in range(8):
        chips = cumulative[ct][qi]
        c5, c50, c95 = pct(chips)
        h5, h50, h95 = c5*H100E[ct], c50*H100E[ct], c95*H100E[ct]
        t5, t50, t95 = c5*TDP[ct], c50*TDP[ct], c95*TDP[ct]
        name = f"China(smuggled) {ct} cumulative through {q_labels[qi]}"
        rows.append([
            name, "Nvidia", "China(smuggled)", ct,
            "1/1/2024", end_dates[qi],
            round(h50), round(h5), round(h95),
            round(c50), round(c5), round(c95),
            source, note,
            "", "", "", "",
            round(t50), round(t5), round(t95), ""
        ])

# Nvidia total: per-draw sums across chip types so percentiles capture
# joint variation correctly (sum-of-medians ≠ median-of-sum for skewed dists).
for qi in range(8):
    total_chips_arr = sum(cumulative[ct][qi] for ct in chip_types)
    total_h100e_arr = sum(cumulative[ct][qi] * H100E[ct] for ct in chip_types)
    total_tdp_arr   = sum(cumulative[ct][qi] * TDP[ct]   for ct in chip_types)
    c5, c50, c95 = pct(total_chips_arr)
    h5, h50, h95 = pct(total_h100e_arr)
    t5, t50, t95 = pct(total_tdp_arr)
    name = f"China(smuggled) Nvidia total cumulative through {q_labels[qi]}"
    rows.append([
        name, "Nvidia", "China(smuggled)", "Nvidia total",
        "1/1/2024", end_dates[qi],
        round(h50), round(h5), round(h95),
        round(c50), round(c5), round(c95),
        source, note,
        "", "", "", "",
        round(t50), round(t5), round(t95), ""
    ])

out_path = "China_smuggled_cumulative_by_chip_type.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(f"Wrote {out_path} ({len(rows)} rows)")

# Summary
print()
print(f"{'Chip':<14}  {'Quarter':<14}  {'H100e (median)':>16}  {'Chips (median)':>16}")
print("-" * 68)
for ct in chip_types:
    chips = cumulative[ct][7]
    c5, c50, c95 = pct(chips)
    print(f"{ct:<14}  {'2025 Q4 cumul':<14}  {c50*H100E[ct]:>16,.0f}  {c50:>16,.0f}")
total_chips_q4 = sum(cumulative[ct][7] for ct in chip_types)
total_h100e_q4 = sum(cumulative[ct][7] * H100E[ct] for ct in chip_types)
print(f"{'Nvidia total':<14}  {'2025 Q4 cumul':<14}  "
      f"{np.median(total_h100e_q4):>16,.0f}  {np.median(total_chips_q4):>16,.0f}")
