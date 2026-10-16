# Amazon Trainium in 2026: evidence and volume estimates

*Compiled 2026-09-25. Research notes for extending `trainium_estimates.ipynb` past Q4 2025. Nothing here is in the model yet.*

## Bottom line

| | H1 2026 | H2 2026 | **FY 2026** |
|---|---|---|---|
| **Chips produced** | ~0.4–0.8M | ~1.1–1.5M | **~1.5–2.2M (central ~1.8M)** |
| **Chips landed** at AWS | ~0.4–0.7M | ~1.0–1.4M | **~1.4–2.0M** |
| **Chips added to service** | ~0.5–0.7M | ~0.5–0.8M | **~1.0–1.4M** |
| Generation mix | Mostly Trn2 / Trn2.5 | Mostly Trn3 | Trn2 ~0.3–0.5M, Trn3 ~1.1–1.7M |

- **Installed base at end of 2026:** about 2.7–3.3M Trainium chips landed in total. That's roughly 1.7–1.8M Trn2 over its whole life plus 1.0–1.5M Trn3.
- **Revenue:** Trainium revenue run rate grew about **2.2x from December 2025 to June 2026** (range 1.9–2.5x). Chips in service grew about 1.6–1.9x over the same period.
- **By customer:** Anthropic takes most of the volume. OpenAI's 2 GW of Trainium ["begins ramping in 2027"](https://www.sec.gov/Archives/edgar/data/1018724/000101872426000012/amzn-20260331xex991.htm), so it adds essentially nothing in 2026.

The three rows measure different things, and it helps to keep them apart:

- **Produced:** chips out of TSMC and packaging. This is what CoWoS allocations and Alchip revenue track.
- **Landed:** chips delivered to AWS data centers. This is what Amazon's "chips landed" figures count.
- **In service:** chips powered on and earning money. This is what GW commitments, "chips in use" and revenue track.

Chips land within weeks of being made. Landed chips seem to go into service about one quarter later. Alchip says it takes 6–7 months from wafer start to shipped chip.

## 1. Amazon's own chip counts (strongest evidence)

| Date | Disclosure |
|---|---|
| Oct 29–30, 2025 | Project Rainier live with ["nearly 500,000" Trn2](https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster). Anthropic expected on [">1M Trn2 by year-end"](https://www.fool.com/earnings/call-transcripts/2025/10/31/amazon-amzn-q3-2025-earnings-call-transcript/). |
| Dec 2–3, 2025 | [">1M Trainium chips landed"](https://www.cnbc.com/video/2025/12/02/aws-ceo-matt-garman-weve-landed-more-than-a-million-trainium-chips.html) (Garman); ["1M+ chips in production"](https://techcrunch.com/2025/12/03/andy-jassy-says-amazons-nvidia-competitor-chip-is-already-a-multi-billion-dollar-business) (Jassy) |
| Feb 5, 2026 | ["Trainium2 is fully subscribed with 1.4 million chips landed"](https://s2.q4cdn.com/299287126/files/doc_earnings/2025/q4/earnings-result/AMZN-Q4-2025-Earnings-Release.pdf) |
| Apr 20, 2026 | Anthropic: ["over one million Trainium2 chips"](https://www.anthropic.com/news/anthropic-amazon-compute) in use |
| Apr 29, 2026 | ["Landed 2.1 million+ AI chips over the past 12 months, more than half of which were Trainium"](https://www.sec.gov/Archives/edgar/data/1018724/000101872426000012/amzn-20260331xex991.htm) |
| Jul 30, 2026 | [No chip count given](https://www.sec.gov/Archives/edgar/data/1018724/000101872426000024/amzn-20260630xex991.htm) |

**What this tells us:**

- 1.4M by Feb 2026 exactly matches [Morgan Stanley's production figures](https://www.capitalfutures.com.tw/zh-tw/financial/globalarticle?contentid=c25071000280&category=fx): ~300k Trn2 made in 2024 plus ~1.1M in 2025. So chips land soon after they're made, and MS's wafer-based method checks out.
- The Q1 2026 figure means over 1.05M Trainium landed from April 2025 to March 2026. The rest of the 2.1M, about 0.8–1.0M, is Nvidia GPUs.
- Going from ~1M (Dec 2) to 1.4M (Feb 5) implies about 200k chips landed per month over the winter.
- About 0.4M Trainium had probably landed by March 2025. Combined with the 12-month figure, that puts **Q1 2026 landings at about 0.25–0.5M**, almost all Trn2 or Trn2.5.

## 2. Power commitments → chips in service

**What Amazon and its customers have said:**

- Anthropic: [up to 5 GW](https://www.anthropic.com/news/anthropic-amazon-compute) across Trn2 to Trn4, and "nearly 1GW total of Trainium2 and Trainium3 capacity coming online by the end of 2026." Also "significant Trainium2 capacity is coming online in Q2 and scaled Trainium3 capacity is expected to come online later this year."
- OpenAI: [~2 GW of Trainium](https://www.aboutamazon.com/news/aws/amazon-open-ai-strategic-partnership-investment), covering Trn3 and Trn4, starting in 2027.
- AWS overall (not just AI): +3.9 GW in 2025, and plans to [double total power capacity by end of 2027](https://www.aboutamazon.com/news/company-news/amazon-ceo-andy-jassy-2025-letter-to-shareholders).

**Turning that into chips:**

- Anthropic's "nearly 1 GW" has to mean *new* capacity. The 1M+ Trn2 already in use is already close to 1 GW.
- Power per chip, including CPUs, networking and cooling at the rack:
  - Trn2: about 0.75–1 kW ([SemiAnalysis](https://newsletter.semianalysis.com/p/amazons-ai-self-sufficiency-trainium2-architecture-networking): 48 kW per 64-chip UltraServer).
  - Trn3: about 1–1.4 kW. There's no official figure; an AWS exec [implied >1 kW per chip](https://www.datacenterdynamics.com/en/news/aws-trainium3-chips-likely-to-consume-1kw-of-power-company-to-adopt-cold-plate-cooling/).
- That works out to **~0.8–1.1M chips added for Anthropic in 2026**: about 300–400k Trn2 in H1 and 500–700k Trn3 in H2.
- Epoch's [New Carlisle tracker](https://epoch.ai/data/ai-data-centers/directory/anthropic-amazon-new-carlisle) shows ~400 MW of growth in 2026 at that one campus, which fits.
- Other customers (Bedrock, Uber, Apple, startups) probably add 20–30% on top. That gives **~1.0–1.4M chips added to service in total**.

## 3. Supply chain: CoWoS, Alchip and analyst unit counts

### Trn3 comes from two separate designs

This is the key finding. [SemiAnalysis's Trn3 deep dive](https://newsletter.semianalysis.com/p/aws-trainium3-deep-dive-a-potential) says there are two versions of Trainium3:

- **"Anita"** is owned by Alchip, which buys the wafers and packaging from TSMC itself.
- **"Mariana"** is owned by Amazon's chip team (Annapurna), which buys directly. SemiAnalysis: "The majority of volumes will be for Mariana."

[36Kr](https://eu.36kr.com/en/p/3631956080985095) separately reports that Annapurna "directly reserves CoWoS capacity from TSMC." So Alchip's revenue only shows the smaller share of Trn3.

### Calibrating on Trn2

Morgan Stanley's [July 2025 note](https://www.capitalfutures.com.tw/zh-tw/financial/globalarticle?contentid=c25071000280&category=fx):

- 1.1M Trn2 / Trn2.5 in 2025.
- 1.9M over the chip's whole life (2H24–1H26), including **~500k "Trainium2.5" in H1 2026**. Trn2.5 is the same N5 chip with bigger HBM3E stacks.
- Packaging split 70% TSMC / 30% ASE.
- Orders ran 30–40% above production because of problems at the circuit-board stage.

Wafer math:

- Marvell got about 55k CoWoS wafers a year ([MS via Astute](https://www.astutegroup.com/news/industrial/advanced-packaging-demand-soars-nvidia-secures-60-of-cowos-capacity/)). Add the ASE share and 1.1M chips means **~14 Trn2 chips per wafer**.
- [Global Technology Research](https://globaltechresearch.substack.com/p/marvell-mrvl-us-vs-alchip-3661-tt) independently says "12 Trainium 2" per wafer.

### Trn3 packaging

- Each Trn3 is ["two CoWoS-R assemblies, rather than one large interposer"](https://newsletter.semianalysis.com/p/aws-trainium3-deep-dive-a-potential). Each assembly holds one N3P compute chip and two HBM3E stacks.
- CoWoS-R uses an organic interposer, not silicon, so Trn3 doesn't compete with Nvidia for the scarce silicon-interposer capacity.
- The real limits are:
  - N3 wafers: "the N3 capacity is fully booked in the industry," per [Alchip's Q4'25 call](https://www.alphaspread.com/security/twse/3661/investor-relations/earnings-call/q4-2025).
  - HBM3E and ABF substrate: Morgan Stanley says these, not CoWoS, are [the binding constraints](https://finance.biggo.com/news/kmUbeJ0Bh5an-7GhgJ7Z).
- Rough conversion: **~8–12 Trn3 chips per wafer**.

### Alchip (Anita): the one hard revenue trail

- Alchip charges for everything except the HBM, which Amazon supplies directly (["except the HBM, I think is consigned"](https://www.alphaspread.com/security/twse/3661/investor-relations/earnings-call/q1-2026)).
- A US broker puts its price at [~$4,600 per chip, with 500k units in 2026](https://wantrich.chinatimes.com/news/20260320900245-420101).
- [Monthly revenue](https://www.wantgoo.com/stock/3661/financial-statements/monthly-revenue) (NT$bn): ~1.4–2.1 each month from Dec to May, then **Jun 3.57, Jul 7.43, Aug 8.78** ([Aug report](https://finance.technews.tw/2026/09/10/alchip-3661-202608-financial-report)). The normal baseline is about 1.5–2 a month.
- After removing the baseline, that's about 10k chips in June and 40–50k a month in July and August.
- Adding [UBS's forecast](https://udn.com/news/story/6848/9732499) of ~NT$30B in Q3 and ~NT$36.5B in Q4 gives **~0.42–0.55M Anita chips in 2026**.
- Alchip says 2026 volume can't go up because production takes 6–7 months ([Q2'26 call](https://ca.investing.com/news/transcripts/earnings-call-transcript-alchip-q2-2026-tops-eps-view-as-ai-ramp-drives-growth-93CH-4800784)).

### Analyst totals

| Source | Estimate |
|---|---|
| Morgan Stanley, Dec 2025 ([CommonWealth](https://english.cw.com.tw/article/article.action?id=4491)) | >1.5M Trainium in 2026. Alchip CoWoS 5k wafers (2025) → 70k (2026) |
| Morgan Stanley, Jan 2026 ([via X](https://x.com/jukan05/status/2008827432665845815), unverified) | Trn3 by maker: Alchip 500–600k, Annapurna 800–900k, Marvell 100–200k |
| Morgan Stanley CoWoS table ([36Kr](https://eu.36kr.com/en/p/3580962946874242)) | 2026: Nvidia 595k wafers (60%), Amazon via Alchip 50k, Marvell 55k (AWS + Microsoft) |
| 36Kr pre-order tally ([link](https://eu.36kr.com/en/p/3631956080985095)) | Alchip 60k wafers (+200% YoY). Marvell flat YoY |
| UBS, Sep 2026 ([UDN](https://udn.com/news/story/6848/9732499)) | Trn3: 1.8M (2026), 2.8M (2027), 4.6M (2028) |
| TrendForce ([Oct 2025](https://www.trendforce.com/presscenter/news/20251013-12741.html), [Feb 2026](https://www.trendforce.com/presscenter/news/20260225-12934.html)) | AWS ASIC shipments +20% in 2026. Trn3 ramps from Q2, stronger in H2 |
| Broker rumor, Oct 2025 ([via X](https://x.com/Jukanlosreve/status/1973640639263941036)) | 3–4M Trn3. Nothing since supports this; ignore |

### Production by maker, 2026

| | Maker, process | 2026 units | Timing | How solid |
|---|---|---|---|---|
| Trn2 / Trn2.5 | Marvell, N5 | 0.3–0.5M | H1, tailing off by Q2–Q3 | Medium |
| Trn3 Anita | Alchip, N3P | 0.42–0.55M | Mass production late May | Good |
| Trn3 Mariana | Annapurna direct, N3P | 0.6–0.9M | Maybe some from Q1, mostly H2 | Weak |
| Trn3 Marvell | Marvell, N3 | 0.1–0.2M | H2 | Weak |
| **Total** | | **1.5–2.2M** | | |

Marvell says it has ["purchase orders covering the entirety of this fiscal year for this next-generation program"](https://www.fool.com/earnings/call-transcripts/2026/03/05/marvell-mrvl-q4-2026-earnings-call-transcript/) (Feb 2026 to Jan 2027). Analysts disagree on its Trn3 role:

- [Benchmark](https://sherwood.news/markets/marvell-sinks-after-benchmark-cuts-company-saying-that-it-lost-its-amazon/) and SemiAnalysis say it lost the lead to Alchip.
- [RBC](https://www.itiger.com/hant/news/2603198493) says Amazon uses both suppliers.

## 4. Revenue

Amazon reports one combined "chips business" figure: Graviton CPUs, Trainium and, from April, Nitro network cards.

| Disclosed | What it covers | Run rate |
|---|---|---|
| [Feb 5, 2026](https://s2.q4cdn.com/299287126/files/doc_earnings/2025/q4/earnings-result/AMZN-Q4-2025-Earnings-Release.pdf) | Trainium + Graviton | ">$10B", growing triple digits YoY |
| [Apr 29, 2026](https://www.fool.com/earnings/call-transcripts/2026/04/29/amazon-amzn-q1-2026-earnings-call-transcript/) | + Nitro | ">$20B", "nearly 40% quarter-over-quarter growth in Q1" |
| [Jul 30, 2026](https://www.sec.gov/Archives/edgar/data/1018724/000101872426000024/amzn-20260630xex991.htm) | same | ">$25B", growing triple digits YoY |

**Taking Graviton and Nitro out:**

- The jump from >$10B to >$20B wasn't 100% growth in a quarter. The Feb figure was a loose floor and didn't include Nitro. Working back from the "nearly 40%" growth, the real December figure was about $14–15B.
- Graviton was "multibillion" and growing >50% a year ([Q4'25 call](https://www.investing.com/news/transcripts/earnings-call-transcript-amazons-q4-2025-revenue-beats-forecast-stock-dips-93CH-4489218)). Say $4–8B. Nitro is probably $1–2B.
- That leaves **Trainium at about $7–9B in December and $16–20B in June, roughly 2.2x**.
- Chips in service grew only ~1.6–1.9x over the same period. The rest comes from more revenue per chip: newly landed chips reaching full use, Bedrock inference growth (Bedrock mostly runs on Trainium) and early Trn3.

**Revenue if Amazon sold the chips instead:**

- Jassy: ["If our chips business was a stand-alone business, and sold chips produced this year to AWS and other third parties … our annual run rate would be ~$50 billion."](https://www.aboutamazon.com/news/company-news/amazon-ceo-andy-jassy-2025-letter-to-shareholders)
- Take out Graviton and Nitro and Trainium is about $35–45B.
- At Nvidia-like prices of $25–35k per chip, that's **1.0–1.8M chips produced in 2026**. This is a soft check, but it agrees with the supply-chain numbers.

**Contracted demand:** Amazon has [">$225 billion in revenue commitments for Trainium"](https://www.fool.com/earnings/call-transcripts/2026/04/29/amazon-amzn-q1-2026-earnings-call-transcript/). Most of that is Anthropic (>$100B over ten years) and OpenAI. It mostly tells us about 2027 and later.

## 5. Price per chip-hour

| | $ per chip-hour | Source |
|---|---|---|
| Trn2 list (Capacity Blocks, Ohio; the only public price) | **$2.24** (was $2.79 at launch) | [AWS](https://aws.amazon.com/ec2/capacityblocks/pricing/), [re:Invent 2024 slides](https://d1.awsstatic.com/onedam/marketing-channels/website/aws/en_US/events/approved/reinvent-2025/reinvent/2024/slides/cmp/CMP333-NEW_AWS-Trainium2-for-breakthrough-AI-training-and-inference-performance.pdf) |
| Trn2 3-year reserved | $2.15 | same slides |
| Trn2 spot | ~$0.78 | [Vantage](https://instances.vantage.sh/aws/ec2/trn2.48xlarge) |
| Large committed customers | ~$0.50–1.00 | [UncoverAlpha expert calls](https://www.uncoveralpha.com/p/amazon-trainium-scaling-ai-without); [The Information](https://finance.yahoo.com/news/amazon-takes-nvidia-discounted-ai-153343703.html) ("H100 performance at 25% of the cost") |
| Anthropic (implied from $100B / 5 GW) | ~$0.3–0.6 | Derived from [Futurum's](https://futurumgroup.com/insights/is-anthropics-100-billion-pact-for-aws-silicon-a-bargain-in-a-supply-constrained-market/) ~$20B per GW |
| Blended rate implied by revenue ÷ chips | ~$1.2–1.6 | Derived |
| For comparison: H100 / H200 / B200 on AWS Capacity Blocks | $5.19 / $5.97 / $12.36 | [AWS](https://aws.amazon.com/ec2/capacityblocks/pricing/) |

- AWS publishes no on-demand price for Trn2 and no price at all for Trn3.
- SemiAnalysis's $/hour figures are only in its paid [AI Cloud TCO Model](https://semianalysis.com/ai-cloud-tco-model/).
- Because the price level is so uncertain, the revenue *multiplier* (section 4) is a more useful input than dollar levels.

## 6. Trn3 timing

- **Dec 2, 2025:** Trn3 [announced as generally available](https://press.aboutamazon.com/2025/12/trainium3-ultraservers-now-available-enabling-customers-to-train-and-deploy-ai-models-faster-at-lower-cost), with Bedrock already running production work on it. No regions or prices were published.
- **Feb 2026:** Amazon expected "nearly all Trainium3 supply committed by mid-2026."
- **Apr 2026:** Trn3 ["just started shipping at the start of 2026"](https://www.aboutamazon.com/news/company-news/amazon-ceo-andy-jassy-2025-letter-to-shareholders) and is "nearly fully-subscribed."
- **Mar 2026:** Digitimes ran a [rumor that Amazon might scale back Trn3](https://www.digitimes.com/news/a20260317PD224/market-amazon-production-shipments-growth.html). Suppliers denied it.
- **Jul 2026:** AWS [raised Q3 Trn3 server targets 20–30%](https://www.tradingkey.com/analysis/stocks/us-stocks/262013038-amazon-aws-hiking-q3-asic-server-shipment-targets-up-30-tradingkey). Full-rack assembly started mass production in July.
- **Most 2026 Trn3 is the air-cooled 64-chip version.** The liquid-cooled 144-chip UltraServer is a smaller share ([SemiAnalysis](https://newsletter.semianalysis.com/p/aws-trainium3-deep-dive-a-potential)).
- **Trainium4:** ["about 18 months from broad availability"](https://www.aboutamazon.com/news/company-news/amazon-ceo-andy-jassy-2025-letter-to-shareholders) as of April 2026, so around Q4 2027.

## 7. What moves the numbers most

1. **Mariana volume (0.6–0.9M).** There's no public data trail, and it drives most of the full-year range.
2. **Trn3 power per chip (1.0 vs 1.4 kW).** This changes the GW → chips conversion by about 40%.
3. **Whether Trn3 shipped before June.** Bedrock ran on Trn3 in December and Jassy said it "began shipping early 2026," which suggests some early Mariana chips. Alchip's revenue shows nothing before June.
4. **Server and rack counts overstate chips.** In 2025, chip orders ran 30–40% above chips actually produced ([MS](https://www.capitalfutures.com.tw/zh-tw/financial/globalarticle?contentid=c25071000280&category=fx)). Counts from server or rack makers overstate chips; wafer-based and Alchip-revenue-based counts don't.
