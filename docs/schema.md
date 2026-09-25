# Canonical schema: AI chip sales and owners

*Agreed 2026-09-24. The contract between the models, `pipeline/chip_schema.py`, `staging/`
and the push tool that writes to Airtable. A spelling not in this file should never reach
Airtable or the public data. The decisions behind it are listed at the end.*

## Principles

- **One name per quantity**, in code and in public. Code uses snake_case. Airtable and the
  public CSVs use display names produced by one translation table (section 6), shared by all
  six tables.
- **Every uncertain number is a triplet**: 5th percentile, median, 95th percentile, suffixed
  `_p5`, `_med`, `_p95`. That is a 90% credible interval around the median. A triplet is
  three separate columns everywhere the data goes; the tables below list the three on one
  line only for brevity.
- **Every table carries the same four metrics**: units, H100e, power and cost. No table lacks
  a metric another has.
- **Calendar quarters and ISO dates.** Quarters read `Q3 2025`, dates `2025-07-01`. Fiscal
  quarters stay inside the model modules.
- **`Name` is generated, never typed.** It is the Airtable primary field and the key the push
  tool matches on.
- **No formula, lookup or linked fields in the results tables.** The repo computes every
  number and the push tool writes plain values. Today several columns are Airtable lookups
  through a `Chip type (linked)` field; all of those go.

## 1. Tables

Two datasets, three grains each. An owners table is the matching sales table plus an `owner`
column.

| Table | One row is |
|---|---|
| `sales_quarterly_by_chip` | chips of one type a designer shipped in one quarter |
| `sales_cumulative_by_chip` | chips of one type a designer shipped from the series start through one quarter |
| `sales_cumulative_by_designer` | the same, summed over the designer's chips |
| `owners_quarterly_by_chip` | chips of one type an owner acquired in one quarter |
| `owners_cumulative_by_chip` | an owner's stock of one chip type at the end of one quarter |
| `owners_cumulative_by_designer` | an owner's stock of one designer's chips at the end of one quarter |

## 2. Quarterly and cumulative rows

The metric columns are identical in both kinds of table. Only the columns saying what span a
row covers differ.

A **quarterly** row covers one quarter. It carries `start_date` and `end_date`, that quarter's
first and last day. Its metrics are flows: what shipped or changed hands within the quarter.

A **cumulative** row covers everything from the start of a designer's series through one
quarter. It carries `series_start`, the date the running total begins, and `end_date`, the last
day of its last quarter. Series start at different times, for example 2022 for Nvidia and 2024
for AMD, so `series_start` belongs on the row. Its metrics are stocks.

There is no separate quarter column: the dates fix the quarter, and the Name (section 5) spells
out the quarter, or the first and last quarter, for readers.

The two kinds are redundant, and deliberately so. A cumulative row is the running sum of the
quarterly rows before it. Its interval, though, comes from summing Monte Carlo samples, not
from adding up quarterly percentiles, and the website cannot redo that sum. Both are
published.

Today's exports blur this. Cumulative rows put the series start in a column called
`Start date` with nothing to say so, and one of the two cumulative tables repeats it in the
Name.

## 3. Columns

Q marks quarterly tables only, C cumulative only, O owners only.

| Column | Type | On | Meaning |
|---|---|---|---|
| `name` | text | all | see section 5 |
| `designer` | text | all | who designed the chip (section 4) |
| `owner` | text | O | who holds it (section 4) |
| `chip_type` | text | by-chip | section 4 |
| `start_date` | date | Q | first day of the quarter |
| `series_start` | date | C | first day of the running total |
| `end_date` | date | all | last day of the quarter or through-quarter |
| `units_p5`, `units_med`, `units_p95` | integer | all | chip count |
| `h100e_p5`, `h100e_med`, `h100e_p95` | number | all | units × the chip's 8-bit TOPS ÷ 1,979 |
| `power_mw_p5`, `power_mw_med`, `power_mw_p95` | number | all | units × the chip's TDP, in MW |
| `cost_usd_p5`, `cost_usd_med`, `cost_usd_p95` | number | all | units × the chip's price, in USD, computed in the repo |
| `incomplete` | true/false | all | the source data covers only part of the period |
| `source` | text | all | a human note or link; empty for model output |
| `notes` | text | all | `Generated on YYYY-MM-DD HH:MM`, written by the exporter |

Dropped, and why:

- **`Total TDP (W)`** carries the same information as power, and its median was unlabelled.
- **`Chip type (linked)`** and the lookups through it, **`Chip TDP (W)`, `Chip H100e`,
  `Chip Cost (USD)`, `Cost Estimate (USD)`**, repeat per-chip facts on every row. The facts
  belong on `chip_types`, and the repo now computes the derived totals.
- **`Last Modified By`, `Last Modified`, `Last Modified 2`, `Created`** are Airtable
  bookkeeping. They stay in Airtable, hidden from the published views.

Chip specs, meaning 8-bit OP/s, TDP and price, come from the Airtable `chip_types` table. The
repo keeps a read-only copy in `staging/chip_types.csv`, refreshed from Airtable before each
build, and reaches specs only through the chip-type map in section 4. A model with its own price
estimates may pass them instead (the TPU model does); nothing else overrides the table. As of
2026-09-24 the specs every model has used match the table exactly for all 22 chips.

## 4. Vocabularies

**Designers:** `Nvidia`, `AMD`, `Google`, `Amazon`, `Huawei`, `Cambricon`. The components base
writes `NVIDIA` and should change to match.

**Chip types**, one spelling each, without a manufacturer prefix:

- Nvidia: `A100`, `A800`, `H100/H200`, `H800`, `H20`, `B200`, `B300`
- AMD: `MI250X`, `MI300A`, `MI300X`, `MI308X`, `MI325X`, `MI350X`, `MI355X`
- Google: `TPU v4`, `TPU v4i`, `TPU v5e`, `TPU v5p`, `TPU v6e`, `TPU v7`
- Amazon: `Trainium1`, `Trainium2`
- Huawei: `Ascend 910B`, `Ascend 910C`
- Cambricon: `Siyuan 590`

**Owners:** `Google`, `Microsoft`, `Meta`, `Amazon`, `Oracle`, `xAI`, `CoreWeave`, `China`,
`China (smuggled)`, `Other`.

Spellings that exist today are mapped on the way in and never written out: `H100`,
`Instinct MI300X` and its siblings, `Trainium1/Inferentia`, `China(smuggled)` and
`China (official)`.

### The chip-type map

Some chip types in the results are synthetic: they do not match one row in `chip_types`. Today
there is one, `H100/H200`, which the models report as a single line. In Airtable its linked
field points at the `H100` row, so it borrows H100's TDP and price. The two chips have identical
specs in `chip_types`, so the numbers come out right, but nothing records that choice.

A repo CSV, `pipeline/chip_type_map.csv`, makes it explicit. It has one row for every chip
type that may appear in results:

| Column | Meaning | `H100/H200` row |
|---|---|---|
| `chip_type` | the canonical results label | `H100/H200` |
| `designer` | canonical designer | `Nvidia` |
| `spec_chip` | the `chip_types` row that supplies TOPS, TDP and price | `H100` |
| `members` | the real chips the label covers, `;`-separated | `H100;H200` |
| `aliases` | old spellings mapped to this label on the way in, `;`-separated | `H100` |

Most rows are one-to-one, such as `MI300X` with spec chip `MI300X` and alias `Instinct MI300X`.
The map is the only place names are translated. The model modules read specs through it, and
the exporter refuses to write a chip type that is missing from it. The `members` column lets a
later model split H100 from H200 without renaming anything.

`chip_types` itself needs a cleanup: it holds both `MI300X` and `Instinct MI300X`, and likewise
for all seven AMD chips.

## 5. Names

One template per table. The designer appears in every one, so names stay unique when an owner
holds several designers' chips.

| Table | Template | Example |
|---|---|---|
| `sales_quarterly_by_chip` | designer chip quarter | `Nvidia H100/H200 Q3 2025` |
| `sales_cumulative_by_chip` | designer chip `cumulative` first quarter `through` last quarter | `AMD MI300X cumulative Q1 2024 through Q3 2025` |
| `sales_cumulative_by_designer` | designer `cumulative` first `through` last | `Google cumulative Q4 2022 through Q3 2025` |
| `owners_quarterly_by_chip` | owner designer chip quarter | `Microsoft Nvidia H100/H200 Q3 2025` |
| `owners_cumulative_by_chip` | owner designer chip `cumulative` first `through` last | `Meta AMD MI300X cumulative Q1 2024 through Q3 2025` |
| `owners_cumulative_by_designer` | owner designer `cumulative` first `through` last | `China Huawei cumulative Q1 2024 through Q3 2025` |

These replace every Name in use today. That costs nothing, because the new names go into new
Airtable tables (section 7); curated rows get the new names once, during the transition. A
cumulative Name includes its series' first quarter, so if a series start moves, every row in
that series gets a new Name and the push replaces those rows rather than updating them.

## 6. Display names

One rule: quantity, then unit, then statistic, with the statistic always spelled out. The
median is never implied by leaving the label off.

| Canonical | Airtable and public |
|---|---|
| `name` | `Name` |
| `designer` | `Designer` (today `Chip manufacturer`) |
| `owner` | `Owner` |
| `chip_type` | `Chip type` |
| `series_start` | `Series start date` |
| `start_date`, `end_date` | `Start date`, `End date` |
| `units_*` | `Number of units (median)`, `(5th percentile)`, `(95th percentile)` |
| `h100e_*` | `H100e (median)`, … |
| `power_mw_*` | `Power (MW) (median)`, … |
| `cost_usd_*` | `Cost (USD) (median)`, … |
| `incomplete` | `Incomplete` |
| `source` | `Source` |
| `notes` | `Notes` |

## 7. Airtable tables and published files

The six tables are new Airtable tables, created for this schema alongside the ones the site
reads today; the website switches to them in one change, and switching back is the rollback.
Tools address tables by ID, which Airtable assigns when they are created and the push tool
records in `pipeline/airtable_tables.json`. The names below are only what a table is called when
first created; rename them in Airtable at any time.

| Canonical | Airtable table | Replaces (current ID) | Published as |
|---|---|---|---|
| `sales_quarterly_by_chip` | Sales: quarterly by chip | `tblyL1FwqiMRzCkeX` | `ai_chip_sales.zip` → `quarterly_by_chip.csv` |
| `sales_cumulative_by_chip` | Sales: cumulative by chip | `tblJWvrfdWJJNXlSl` | `ai_chip_sales.zip` → `cumulative_by_chip.csv` |
| `sales_cumulative_by_designer` | Sales: cumulative by designer | `tblb8LAUvR9OufJBl` | `ai_chip_sales.zip` → `cumulative_by_designer.csv` |
| `owners_quarterly_by_chip` | Owners: quarterly by chip | `tblf8X5qu4r1Hcfjy` | `ai_chip_owners.zip` → `quarterly_by_chip.csv` |
| `owners_cumulative_by_chip` | Owners: cumulative by chip | `tbl7p2wcXksNLzbjt` | `ai_chip_owners.zip` → `cumulative_by_chip.csv` |
| `owners_cumulative_by_designer` | Owners: cumulative by designer | `tblU5qHjez8cgXkSE` | `ai_chip_owners.zip` → `cumulative_by_designer.csv` |

The file names inside the zips change with the columns, in one breaking release with a
changelog entry. `chip_types` and `organizations` keep their tables and file names.

## 8. Who writes what

The repo writes only the rows its models produce: every row whose designer, or for owners
tables whose owner and designer, a model family covers. Everything else in the base is
Airtable's, edited there and never overwritten by the repo:

- **Curated rows**: Huawei, Cambricon, xAI, smuggled China, and any future row no model
  produces.
- **`chip_types` and `organizations`**: the repo only reads them.

The one exception is the transition. The new tables start empty, so the curated rows, cleaned to
this schema (new Names, fixed typos such as the Cambricon Q2 2024 row), are written into them
once from `staging/curated/`. After that, `staging/curated/` becomes a read-only copy pulled
from Airtable, like `staging/chip_types.csv`, and the push tool refuses to write those rows.

## Decisions (2026-09-24)

| # | Question | Decision |
|---|---|---|
| 1 | How cumulative rows state their start | A `Series start date` column, and the first quarter in the Name; no quarter label columns (revised 2026-09-24) |
| 2 | Where cost is computed | In the repo, per sample |
| 3 | Where chip specs live | Airtable `chip_types`, read by the repo |
| 4 | Name templates | The new ones in section 5 |
| 5 | `Nvidia` or `NVIDIA` | `Nvidia` |
| 6 | Designer column's public name | `Designer` |
| 7 | Files inside the public zips | Renamed to match the new tables |
| 8 | Who writes curated rows, chip types and organizations | Airtable only, after a one-time transition (section 8) |

## Implementation notes

- **Cost with a missing price.** Trainium1 and Siyuan 590 have no price in `chip_types`. Their
  own cost is blank, and so is any designer total that includes them, rather than an
  undercount. The sales hub sums priced chips for its designer cost view, as the live site does.
- **Rows with no samples behind them** (the Nvidia "Other" remainder, CoreWeave, the curated
  rows) are converted from percentiles by `pipeline/build_staging.py`. Their designer-level cost is the
  sum of per-chip costs, and the row's Notes say so. Point estimates keep blank percentiles.
- **Incomplete flags** follow `chip_schema.incomplete_quarters`: a quarterly row is incomplete
  when its quarter starts before or ends after the source data; a cumulative row only when it
  ends after it.
