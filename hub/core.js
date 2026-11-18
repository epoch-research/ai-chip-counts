/* Shared machinery for the AI Chip Sales and AI Chip Owners explorers.
 *
 * Reads the canonical tables from ../staging/ (the stand-in for Airtable) and renders
 * a stacked bar or area chart, its legend, a settings sidebar and a table view. The two
 * explorer scripts only decide which rows go into which series.
 */
/* global d3 */
'use strict';

const Hub = (() => {
  // ---------------------------------------------------------------- constants
  const STAGING = '../staging/';

  const COLORS = {
    teal: '#00A5A6', pink: '#E03D90', orange: '#FC6538', purple: '#6A3ECB', blue: '#0058DC',
    yellow: '#EA8D00', green: '#279E27', yellow2: '#E1C700', turquoise: '#3CC8C8', darkBlue: '#015D90',
    red: '#EA4831', lightGreen: '#5BC236', lightBlue: '#009AF1', lightPurple: '#B087F4',
    other: '#AAB1B1', trend: '#3E555E',
  };
  const DESIGNER_COLORS = {
    Nvidia: COLORS.teal, AMD: COLORS.orange, Google: COLORS.pink,
    Huawei: COLORS.purple, Amazon: COLORS.blue, Cambricon: COLORS.yellow,
  };
  const CATEGORICAL = [COLORS.teal, COLORS.pink, COLORS.orange, COLORS.purple, COLORS.blue, COLORS.yellow,
    COLORS.green, COLORS.yellow2, COLORS.turquoise, COLORS.darkBlue, COLORS.lightGreen, COLORS.lightBlue,
    COLORS.lightPurple];

  // Metric id -> canonical column stem, display labels, and a scale factor from the stored unit.
  const METRICS = {
    h100e: { stem: 'H100e', label: 'Compute capacity (H100e)', scale: 1,
      tooltip: 'Compute capacity in the equivalent number of Nvidia H100s, based on dense 8-bit operations per second.' },
    power: { stem: 'Power (MW)', label: 'Chip power (GW)', scale: 1 / 1000,
      tooltip: 'Total chip power in gigawatts, in terms of thermal design power. Excludes server and data center overheads.' },
    cost: { stem: 'Cost (USD)', label: 'Cost (USD)', scale: 1,
      tooltip: 'Estimated cost of the chips in US dollars, excluding servers, networking and facilities.' },
    units: { stem: 'Number of units', label: 'Number of units', scale: 1, tooltip: 'Number of chip units.' },
  };
  const STATS = { med: 'median', p5: '5th percentile', p95: '95th percentile' };

  // ---------------------------------------------------------------- data
  // Standalone builds embed every table as CSV text in window.HUB_DATA, so they open
  // from disk; served pages fetch the files from staging/.
  const EMBEDDED = window.HUB_DATA || null;
  const PATHS = { chip_type_map: '../pipeline/chip_type_map.csv' };
  const cache = {};
  function loadCsv(name) {
    if (!cache[name]) {
      cache[name] = EMBEDDED && EMBEDDED[name] != null
        ? Promise.resolve(d3.csvParse(EMBEDDED[name]))
        : d3.csv(PATHS[name] || `${STAGING}${name}.csv`);
    }
    return cache[name];
  }

  function parseDate(s) {
    const [y, m, d] = s.split('-').map(Number);
    return new Date(Date.UTC(y, m - 1, d));
  }
  function endOfDay(date) {
    return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate(), 23, 59, 59, 999));
  }
  function quarterLabel(date) { return `${date.getUTCFullYear()} Q${Math.floor(date.getUTCMonth() / 3) + 1}`; }

  /* One staging row -> {designer, owner, chip, start, end, incomplete, m: {metric: {med, p5, p95}}}.
   * Quarterly rows span their quarter; cumulative rows span series start to end. */
  function parseRow(r) {
    const num = (v) => (v === '' || v == null ? null : +v);
    const m = {};
    for (const [id, spec] of Object.entries(METRICS)) {
      m[id] = {};
      for (const [s, word] of Object.entries(STATS)) {
        const v = num(r[`${spec.stem} (${word})`]);
        m[id][s] = v == null ? null : v * spec.scale;
      }
    }
    const start = parseDate(r['Start date'] || r['Series start date']);
    return {
      name: r.Name, designer: r.Designer, owner: r.Owner || null, chip: r['Chip type'] || null,
      start, end: endOfDay(parseDate(r['End date'])), incomplete: r.Incomplete === 'true', m, raw: r,
    };
  }

  async function loadTable(name) { return (await loadCsv(name)).map(parseRow); }

  async function loadChipTypes() {
    const rows = await loadCsv('chip_types');
    const map = await loadCsv('chip_type_map');
    const byName = new Map(rows.map((r) => [r.Name, r]));
    // Canonical results label -> release date of the chip that supplies its specs.
    const release = new Map();
    for (const r of map) {
      const spec = byName.get(r.spec_chip);
      release.set(r.chip_type, spec && spec['Release date'] ? parseDate(spec['Release date']) : new Date(0));
    }
    return { rows, release };
  }

  // ---------------------------------------------------------------- time
  function quarterIntervals(start, end) {
    const out = [];
    let y = start.getUTCFullYear(); let q = Math.floor(start.getUTCMonth() / 3);
    for (;;) {
      const from = new Date(Date.UTC(y, q * 3, 1));
      if (from >= end) break;
      const to = new Date(Date.UTC(y + (q === 3 ? 1 : 0), (q * 3 + 3) % 12, 1) - 1);
      out.push({ from, to, label: `${y} Q${q + 1}`, long: `${y} Q${q + 1} (${monthShort(from)} – ${monthShort(to)})` });
      q += 1; if (q > 3) { q = 0; y += 1; }
    }
    return out;
  }
  function yearIntervals(start, end) {
    const out = [];
    for (let y = start.getUTCFullYear(); ; y += 1) {
      const from = new Date(Date.UTC(y, 0, 1));
      if (from >= end) break;
      out.push({ from, to: new Date(Date.UTC(y + 1, 0, 1) - 1), label: `${y}`, long: `${y} (Jan – Dec)` });
    }
    return out;
  }
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function monthShort(d) { return MONTHS[d.getUTCMonth()]; }
  function inInterval(date, iv) { return date > iv.from && date <= iv.to; }

  // ---------------------------------------------------------------- formatting
  function formatNumber(v, sig = 3) {
    if (v == null || !isFinite(v)) return '–';
    const a = Math.abs(v);
    const units = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
    for (const [k, s] of units) {
      if (a >= k) return `${+(v / k).toPrecision(sig)}${s}`;
    }
    if (a === 0) return '0';
    return a < 1 ? `${+v.toPrecision(2)}` : `${+v.toPrecision(sig)}`;
  }
  function formatAxis(v) { return formatNumber(v, 3); }
  function formatShare(v) { return v > 1 - 1e-6 ? '100%' : `${(v * 100).toFixed(1)}%`; }
  function lowerFirst(s) { return s.charAt(0).toLowerCase() + s.slice(1); }

  // ---------------------------------------------------------------- palettes
  /* Shades of one base colour, perceptually spaced in Lab lightness, light to dark.
   * Mirrors the website's palette helper: make n + 4 shades and drop the extremes. */
  function sequentialPalette(base, n) {
    if (n <= 1) return [base];
    const lab = d3.lab(base);
    const count = n + 4;
    const range = 100 * (0.95 - 1 / count);
    const step = range / (count - 1);
    const lo = (100 - range) / 2;
    const levels = d3.range(count).map((i) => lo + i * step);
    let offset = Infinity;
    for (const l of levels) if (Math.abs(lab.l - l) < Math.abs(offset)) offset = lab.l - l;
    const colors = levels.map((l) => d3.lab(l + offset, lab.a, lab.b).formatHex()).reverse();
    return colors.slice(2, count - 2);
  }

  // ---------------------------------------------------------------- trend
  /* Least-squares exponential fit y = a·e^(bt) with a bootstrap interval on the growth rate. */
  function fitExponential(points) {
    const pts = points.filter((p) => p.y > 0);
    if (pts.length < 2) return null;
    const fit = (arr) => {
      const n = arr.length;
      let sx = 0; let sy = 0; let sxy = 0; let sxx = 0;
      for (const p of arr) { const ly = Math.log(p.y); sx += p.x; sy += ly; sxy += p.x * ly; sxx += p.x * p.x; }
      const den = n * sxx - sx * sx;
      if (Math.abs(den) < 1e-10) return null;
      const b = (n * sxy - sx * sy) / den;
      return { a: Math.exp((sy - b * sx) / n), b };
    };
    const best = fit(pts);
    if (!best) return null;
    const msYear = 365.25 * 864e5;
    const boots = [];
    if (pts.length >= 3) {
      let seed = 42;
      const rand = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
      for (let i = 0; i < 1000; i += 1) {
        const sample = pts.map(() => pts[Math.floor(rand() * pts.length)]);
        const f = fit(sample);
        if (f) boots.push(f.b * msYear);
      }
      boots.sort((x, y) => x - y);
    }
    const q = (p) => boots[Math.floor(p * (boots.length - 1))];
    const perYear = best.b * msYear;
    return { ...best, perYear, ci: boots.length ? [q(0.05), q(0.95)] : null };
  }

  /* Trend over per-period totals. In cumulative mode the fit is to the increments
   * between bars, since fitting stocks directly is biased upward. Trailing incomplete
   * periods are left out of the fit. Returns the line and summary stats. */
  function trendLine(totals, cumulative, projectUntil) {
    const trailing = (i) => totals[i].incomplete && !totals.slice(i + 1).some((t) => !t.incomplete);
    let endIdx = totals.length - 1;
    while (endIdx >= 0 && trailing(endIdx)) endIdx -= 1;
    if (endIdx < 1) return null;
    const step = totals[endIdx].x - totals[endIdx - 1].x;
    const future = [];
    if (projectUntil) for (let t = totals[endIdx].x + step; t <= projectUntil.getTime() + 1; t += step) future.push(t);

    if (cumulative) {
      const diffs = [];
      for (let i = 1; i <= endIdx; i += 1) {
        if (totals[i].y > totals[i - 1].y) diffs.push({ x: totals[i].x, y: totals[i].y - totals[i - 1].y });
      }
      const f = fitExponential(diffs);
      if (!f) return null;
      let acc = totals[0].y;
      const pts = [{ x: totals[0].x, y: acc }];
      for (let i = 1; i <= endIdx; i += 1) { acc += f.a * Math.exp(f.b * totals[i].x); pts.push({ x: totals[i].x, y: acc }); }
      for (const t of future) { acc += f.a * Math.exp(f.b * t); pts.push({ x: t, y: acc, projected: true }); }
      return { ...f, points: pts, title: 'Trend in quarterly additions', lastReal: totals[endIdx].x };
    }
    const f = fitExponential(totals.slice(0, endIdx + 1).map((t) => ({ x: t.x, y: t.y })));
    if (!f) return null;
    const xs = [...totals.slice(0, endIdx + 1).map((t) => t.x), ...future];
    return { ...f, points: xs.map((x) => ({ x, y: f.a * Math.exp(f.b * x), projected: x > totals[endIdx].x })),
      title: 'Trend', lastReal: totals[endIdx].x };
  }

  function trendSummary(t) {
    if (!t) return '';
    const x = (r) => Math.exp(r).toFixed(1);
    const dbl = (r) => (r > 0 ? ((Math.log(2) / r) * 12).toFixed(1) : '–');
    const ci = t.ci ? ` (90% CI ${x(t.ci[0])}–${x(t.ci[1])}×)` : '';
    return `${t.title}: ${x(t.perYear)}× per year${ci}, doubling every ${dbl(t.perYear)} months`;
  }

  // ---------------------------------------------------------------- chart
  let hatchCount = 0;

  /* Draw a stacked chart.
   * spec: {
   *   periods: [{label, long, x?}],   // one column per period (x: timestamp for trends)
   *   series:  [{id, label, color, values: [{value, low, high, incomplete}]}], bottom to top
   *   type: 'bar' | 'area', share, yLabel, yTooltip,
   *   trend: result of trendLine(), projected periods drawn after the real ones,
   *   showCi(seriesId) -> bool, xFormat(label) }
   */
  function drawChart(el, spec) {
    el.innerHTML = '';
    const width = Math.max(320, el.clientWidth);
    const height = Math.max(300, Math.min(560, el.clientHeight || 460));
    const margin = { top: 16, right: 16, bottom: 44, left: 64 };
    const svg = d3.select(el).append('svg').attr('width', width).attr('height', height).attr('class', 'chart-svg');
    const hatchId = `hatch-${hatchCount += 1}`;
    const pattern = svg.append('defs').append('pattern').attr('id', hatchId).attr('patternUnits', 'userSpaceOnUse')
      .attr('width', 6).attr('height', 6).attr('patternTransform', 'rotate(45)');
    pattern.append('rect').attr('width', 6).attr('height', 6).attr('fill', 'rgba(255,255,255,0.35)');
    pattern.append('line').attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 6).attr('stroke', 'rgba(255,255,255,0.85)').attr('stroke-width', 2.5);

    const projected = spec.trend ? spec.trend.points.filter((p) => p.projected) : [];
    const labels = [...spec.periods.map((p) => p.label), ...projected.map((p) => `${quarterLabel(new Date(p.x))}*`)];
    const x = d3.scaleBand().domain(labels).range([margin.left, width - margin.right]).padding(spec.type === 'bar' ? 0.18 : 0);

    // Stack: cumulative sums bottom to top per period.
    const stacks = spec.periods.map((_, i) => {
      let acc = 0;
      return spec.series.map((s) => { const v = Math.max(0, s.values[i]?.value || 0); const y0 = acc; acc += v; return [y0, acc]; });
    });
    const totals = stacks.map((st) => (st.length ? st[st.length - 1][1] : 0));
    const trendMax = spec.trend ? d3.max(spec.trend.points, (p) => p.y) : 0;
    const yMax = spec.share ? 1 : Math.max(d3.max(totals) || 1, trendMax || 0) * 1.05;
    const y = d3.scaleLinear().domain([0, yMax]).nice().range([height - margin.bottom, margin.top]);

    if (projected.length) {
      const first = x(labels[spec.periods.length]);
      svg.append('rect').attr('class', 'projection-bg').attr('x', first - (x.step() * x.padding()) / 2)
        .attr('y', margin.top).attr('width', width - margin.right - first + (x.step() * x.padding()) / 2)
        .attr('height', height - margin.bottom - margin.top);
    }
    svg.append('g').attr('class', 'grid').attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(6).tickSize(-(width - margin.left - margin.right)).tickFormat(''));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(6).tickFormat(spec.share ? (v) => `${Math.round(v * 100)}%` : formatAxis));
    const every = Math.ceil(labels.length / Math.max(2, Math.floor((width - margin.left) / 64)));
    svg.append('g').attr('class', 'axis x-axis').attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickValues(labels.filter((_, i) => i % every === 0)).tickFormat(spec.xFormat || ((d) => d)));

    const plot = svg.append('g');
    if (spec.type === 'area') {
      const cx = (i) => x(labels[i]) + x.bandwidth() / 2;
      spec.series.forEach((s, si) => {
        const area = d3.area().x((_, i) => cx(i)).y0((_, i) => y(stacks[i][si][0])).y1((_, i) => y(stacks[i][si][1]));
        plot.append('path').attr('d', area(spec.periods)).attr('fill', s.color).attr('fill-opacity', 0.85);
      });
    } else {
      spec.series.forEach((s, si) => {
        spec.periods.forEach((p, i) => {
          const [y0, y1] = stacks[i][si];
          if (y1 - y0 <= 0) return;
          const attrs = { x: x(p.label), y: y(y1), width: x.bandwidth(), height: Math.max(0.5, y(y0) - y(y1)) };
          plot.append('rect').attr('x', attrs.x).attr('y', attrs.y).attr('width', attrs.width).attr('height', attrs.height)
            .attr('fill', s.color).attr('fill-opacity', 0.85);
          if (s.values[i]?.incomplete) {
            plot.append('rect').attr('x', attrs.x).attr('y', attrs.y).attr('width', attrs.width).attr('height', attrs.height)
              .attr('fill', `url(#${hatchId})`);
          }
        });
      });
    }

    if (spec.trend) {
      const px = (p) => {
        const idx = spec.periods.findIndex((q) => q.x === p.x);
        const lbl = idx >= 0 ? labels[idx] : `${quarterLabel(new Date(p.x))}*`;
        return x(lbl) + x.bandwidth() / 2;
      };
      const line = d3.line().x(px).y((p) => y(p.y)).curve(d3.curveMonotoneX);
      svg.append('path').attr('class', 'trend').attr('d', line(spec.trend.points.filter((p) => !p.projected)));
      const proj = spec.trend.points.filter((p) => p.projected || p.x === spec.trend.lastReal);
      if (proj.length > 1) svg.append('path').attr('class', 'trend projected').attr('d', line(proj));
    }

    // Hover: one column at a time.
    const tip = el.parentElement.querySelector('.tooltip') || el.parentElement.appendChild(Object.assign(document.createElement('div'), { className: 'tooltip' }));
    tip.hidden = true;
    const hover = svg.append('rect').attr('class', 'hover-col').attr('y', margin.top).attr('height', height - margin.bottom - margin.top).attr('width', x.step()).attr('opacity', 0);
    svg.append('rect').attr('x', margin.left).attr('y', margin.top).attr('width', width - margin.left - margin.right)
      .attr('height', height - margin.top - margin.bottom).attr('fill', 'transparent')
      .on('mousemove', (ev) => {
        const [mx] = d3.pointer(ev);
        const i = Math.floor((mx - margin.left) / x.step());
        if (i < 0 || i >= spec.periods.length) { tip.hidden = true; hover.attr('opacity', 0); return; }
        hover.attr('x', x(labels[i]) - (x.step() * x.padding()) / 2).attr('opacity', 1);
        tip.innerHTML = tooltipHtml(spec, i, totals[i]);
        tip.hidden = false;
        const box = el.getBoundingClientRect();
        const left = ev.clientX - box.left + 16;
        tip.style.left = `${Math.min(left, box.width - tip.offsetWidth - 8)}px`;
        tip.style.top = `${Math.max(0, ev.clientY - box.top - tip.offsetHeight / 2)}px`;
      })
      .on('mouseleave', () => { tip.hidden = true; hover.attr('opacity', 0); });
  }

  function tooltipHtml(spec, i, total) {
    const fmt = spec.share ? formatShare : (v) => formatNumber(v);
    const rows = [...spec.series].reverse().filter((s) => (s.values[i]?.value || 0) > 0).map((s) => {
      const v = s.values[i];
      const ci = !spec.share && spec.showCi && spec.showCi(s.id) && v.low != null && v.high != null
        ? ` <span class="ci">(${formatNumber(v.low)} – ${formatNumber(v.high)})</span>` : '';
      const inc = v.incomplete ? ' <span class="ci">incomplete</span>' : '';
      return `<tr><td><span class="swatch" style="background:${s.color}"></span>${escapeHtml(s.label)}</td><td>${fmt(v.value)}${ci}${inc}</td></tr>`;
    });
    const totalRow = spec.series.length > 1 && !spec.share ? `<tr class="total"><td>Total</td><td>${fmt(total)}</td></tr>` : '';
    const ciNote = spec.showCi && !spec.share ? '<div class="tip-note">Brackets show the 90% credible interval.</div>' : '';
    return `<div class="tip-head">${escapeHtml(spec.periods[i].long || spec.periods[i].label)}</div><table>${rows.join('')}${totalRow}</table>${ciNote}`;
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  /* Legend: groups [{header, items: [{id, label, color, tooltip?}]}], read top to bottom. */
  function drawLegend(el, groups, hasIncomplete) {
    el.innerHTML = '';
    for (const g of groups) {
      const div = document.createElement('div');
      div.className = 'legend-group';
      if (g.header) div.innerHTML = `<div class="legend-header">${escapeHtml(g.header)}</div>`;
      for (const it of g.items) {
        const row = document.createElement('div');
        row.className = 'legend-item';
        if (it.tooltip) row.title = it.tooltip;
        row.innerHTML = `<span class="swatch" style="background:${it.color}"></span>${escapeHtml(it.label)}`;
        div.appendChild(row);
      }
      el.appendChild(div);
    }
    if (hasIncomplete) {
      const row = document.createElement('div');
      row.className = 'legend-item legend-incomplete';
      row.title = 'The source data for this period does not cover all of it, or a value was carried forward.';
      row.innerHTML = '<span class="swatch hatch"></span>Incomplete data';
      el.appendChild(row);
    }
  }

  // ---------------------------------------------------------------- settings sidebar
  /* elements: [{type: 'group', title, open, elements: [...]}]; leaf types:
   *   chips | pills: {id, title, options: {value: label}, default}
   *   select: same, rendered as a dropdown
   *   toggle: {id, title, default, tooltip}
   *   range:  {id, title, min, max, step, default, format}
   * Every leaf may have enableIf(state) and showIf(state). */
  function buildSidebar(el, elements, state, onChange) {
    const leaves = [];
    el.innerHTML = '<div class="sidebar-head"><h2>Settings</h2><button class="reset" title="Reset settings">↺</button></div>';
    el.querySelector('.reset').onclick = () => {
      for (const l of leaves) state[l.id] = typeof l.default === 'function' ? l.default() : l.default;
      onChange();
    };
    for (const group of elements) {
      const sec = document.createElement('section');
      sec.className = 'sidebar-group';
      if (group.title) {
        const h = document.createElement('button');
        h.className = 'group-title';
        h.innerHTML = `<span>${group.title}</span><span class="sign">${group.open ? '−' : '+'}</span>`;
        h.onclick = () => { sec.classList.toggle('closed'); h.querySelector('.sign').textContent = sec.classList.contains('closed') ? '+' : '−'; };
        sec.appendChild(h);
        if (!group.open) sec.classList.add('closed');
      }
      for (const leaf of group.elements) {
        leaves.push(leaf);
        if (state[leaf.id] === undefined) state[leaf.id] = typeof leaf.default === 'function' ? leaf.default() : leaf.default;
        const wrap = document.createElement('div');
        wrap.className = `control control-${leaf.type}`;
        wrap.dataset.id = leaf.id;
        sec.appendChild(wrap);
      }
      el.appendChild(sec);
    }

    function render() {
      for (const leaf of leaves) {
        const wrap = el.querySelector(`.control[data-id="${leaf.id}"]`);
        const visible = leaf.showIf ? leaf.showIf(state) : true;
        const enabled = leaf.enableIf ? leaf.enableIf(state) : true;
        wrap.hidden = !visible;
        wrap.classList.toggle('disabled', !enabled);
        const tip = leaf.tooltip ? ` <span class="info" title="${escapeHtml(leaf.tooltip)}">i</span>` : '';
        if (leaf.type === 'toggle') {
          wrap.innerHTML = `<label class="toggle"><input type="checkbox" ${state[leaf.id] ? 'checked' : ''} ${enabled ? '' : 'disabled'}><span class="slider"></span><span>${leaf.title}${tip}</span></label>`;
          wrap.querySelector('input').onchange = (e) => { state[leaf.id] = e.target.checked; onChange(); };
        } else if (leaf.type === 'select') {
          const opts = typeof leaf.options === 'function' ? leaf.options(state) : leaf.options;
          wrap.innerHTML = `<div class="control-title">${leaf.title}${tip}</div><select ${enabled ? '' : 'disabled'}>${Object.entries(opts).map(([v, l]) => `<option value="${escapeHtml(v)}" ${state[leaf.id] === v ? 'selected' : ''}>${escapeHtml(l)}</option>`).join('')}</select>`;
          wrap.querySelector('select').onchange = (e) => { state[leaf.id] = e.target.value; onChange(); };
        } else if (leaf.type === 'range') {
          const lo = typeof leaf.min === 'function' ? leaf.min(state) : leaf.min;
          const hi = typeof leaf.max === 'function' ? leaf.max(state) : leaf.max;
          state[leaf.id] = Math.min(hi, Math.max(lo, state[leaf.id]));
          wrap.innerHTML = `<div class="control-title">${leaf.title}: <b>${leaf.format(state[leaf.id])}</b></div><input type="range" min="${lo}" max="${hi}" step="${leaf.step || 1}" value="${state[leaf.id]}" ${enabled ? '' : 'disabled'}>`;
          const input = wrap.querySelector('input');
          input.oninput = () => { wrap.querySelector('b').textContent = leaf.format(+input.value); };
          input.onchange = () => { state[leaf.id] = +input.value; onChange(); };
        } else {
          const opts = typeof leaf.options === 'function' ? leaf.options(state) : leaf.options;
          wrap.innerHTML = `<div class="control-title">${leaf.title}${tip}</div><div class="${leaf.type}">${Object.entries(opts).map(([v, l]) => {
            const off = leaf.disabledOptions && leaf.disabledOptions(state).includes(v);
            return `<button data-v="${escapeHtml(v)}" class="${state[leaf.id] === v ? 'active' : ''}" ${enabled && !off ? '' : 'disabled'}>${escapeHtml(l)}</button>`;
          }).join('')}</div>`;
          wrap.querySelectorAll('button').forEach((b) => { b.onclick = () => { state[leaf.id] = b.dataset.v; onChange(); }; });
        }
      }
    }
    return render;
  }

  // ---------------------------------------------------------------- table view
  const isNumCell = (v) => v !== '' && v != null && !isNaN(+v);

  /* One column's summary for the inspector: fill rate and range, or top values. */
  function columnSummary(rows, c) {
    const vals = rows.map((r) => r[c]);
    const filled = vals.filter((v) => v !== '' && v != null);
    const nums = filled.filter(isNumCell).map(Number).sort((a, b) => a - b);
    let detail;
    if (filled.length && nums.length === filled.length && !/date/i.test(c)) {
      detail = `${formatNumber(nums[0])} · ${formatNumber(nums[Math.floor(nums.length / 2)])} · ${formatNumber(nums[nums.length - 1])}`;
    } else {
      const counts = d3.rollups(filled, (l) => l.length, (v) => v).sort((a, b) => b[1] - a[1]);
      detail = `${counts.length} distinct: ${counts.slice(0, 2).map(([v, n]) => `${v.length > 18 ? `${v.slice(0, 18)}…` : v} (${n})`).join(', ')}`;
    }
    return { filled: filled.length, total: vals.length, detail };
  }

  /* Automatic sanity checks shown beside a table. */
  function tableChecks(rows) {
    const cols = rows.columns; const out = [];
    if (cols.includes('Name')) {
      const dup = rows.length - new Set(rows.map((r) => r.Name)).size;
      out.push(dup ? ['warn', `${dup} duplicate Name(s)`] : ['ok', 'Every Name is unique']);
    }
    for (const spec of Object.values(METRICS)) {
      const [lo, med, hi] = ['5th percentile', 'median', '95th percentile'].map((w) => `${spec.stem} (${w})`);
      if (!cols.includes(med)) continue;
      const bad = rows.filter((r) => [lo, med, hi].every((c) => isNumCell(r[c])) && !(+r[lo] <= +r[med] + 1e-9 && +r[med] <= +r[hi] + 1e-9));
      const blank = rows.filter((r) => !isNumCell(r[med])).length;
      out.push(bad.length ? ['warn', `${spec.stem}: ${bad.length} row(s) with percentiles out of order`] : ['ok', `${spec.stem}: 5th ≤ median ≤ 95th`]);
      if (blank) out.push([spec.stem.startsWith('Cost') ? 'info' : 'warn', `${spec.stem}: ${blank} row(s) with no median${spec.stem.startsWith('Cost') ? ' (chips with no price)' : ''}`]);
    }
    // Coverage per designer, from the End date of each row.
    if (cols.includes('End date') && cols.includes('Designer') && cols.includes('Name')) {
      const label = (s) => { const [y, m] = s.split('-').map(Number); return `Q${Math.floor((m - 1) / 3) + 1} ${y}`; };
      for (const [d, l] of d3.groups(rows, (r) => r.Designer)) {
        const ends = l.map((r) => r['End date']).sort();
        out.push(['info', `${d}: ${label(ends[0])} to ${label(ends[ends.length - 1])}`]);
      }
    }
    return out;
  }

  /* Table view. Columns can be hidden from the Inspect panel; the choice is kept per
   * table in state.hiddenColumns. */
  async function drawTable(el, side, csvName, note, state, onChange) {
    const rows = await loadCsv(csvName);
    const cols = rows.columns;
    state.hiddenColumns = state.hiddenColumns || {};
    const hidden = new Set(state.hiddenColumns[csvName] || []);
    let sortCol = null; let asc = true; let filter = '';
    const download = EMBEDDED
      ? URL.createObjectURL(new Blob([EMBEDDED[csvName]], { type: 'text/csv' }))
      : `${STAGING}${csvName}.csv`;
    el.innerHTML = `<div class="table-tools"><input type="search" placeholder="Filter rows…"><span class="count"></span><a class="download" href="${download}" download="${csvName}.csv">Download CSV</a></div>${note ? `<p class="table-note">${note}</p>` : ''}<div class="table-scroll"><table class="data-table"><thead></thead><tbody></tbody></table></div>`;
    const input = el.querySelector('input');
    input.oninput = () => { filter = input.value.toLowerCase(); render(); };

    const metricGroups = [
      ['All', () => []],
      ['Medians only', () => cols.filter((c) => /percentile\)$/.test(c))],
      ['Keys only', () => cols.filter((c) => /\((median|5th percentile|95th percentile)\)$/.test(c) || ['Notes', 'Source'].includes(c))],
    ];
    function drawSide() {
      side.innerHTML = `<div class="sidebar-head"><h2>Inspect</h2></div>
        <section class="sidebar-group"><div class="control-title">Checks</div><ul class="checks">${tableChecks(rows).map(([k, t]) => `<li class="check-${k}">${escapeHtml(t)}</li>`).join('')}</ul></section>
        <section class="sidebar-group"><div class="control-title">Columns <span class="hint">${cols.length - hidden.size} of ${cols.length} shown</span></div>
          <div class="chips col-presets">${metricGroups.map(([l], i) => `<button data-i="${i}">${l}</button>`).join('')}</div>
          <div class="col-list">${cols.map((c) => {
            const sm = columnSummary(rows, c);
            return `<label class="col-row"><input type="checkbox" data-c="${escapeHtml(c)}" ${hidden.has(c) ? '' : 'checked'}><span><b>${escapeHtml(c)}</b><span class="col-meta ${sm.filled < sm.total ? 'blank' : ''}">${sm.filled}/${sm.total} filled · ${escapeHtml(sm.detail)}</span></span></label>`;
          }).join('')}</div></section>`;
      side.querySelectorAll('.col-row input').forEach((cb) => {
        cb.onchange = () => { if (cb.checked) hidden.delete(cb.dataset.c); else hidden.add(cb.dataset.c); save(); };
      });
      side.querySelectorAll('.col-presets button').forEach((b) => {
        b.onclick = () => { hidden.clear(); metricGroups[+b.dataset.i][1]().forEach((c) => hidden.add(c)); save(); };
      });
    }
    function save() {
      state.hiddenColumns[csvName] = [...hidden];
      onChange();   // persists the choice in the URL
      drawSide(); render();
    }
    function render() {
      const shownCols = cols.filter((c) => !hidden.has(c));
      let shown = filter ? rows.filter((r) => cols.some((c) => String(r[c]).toLowerCase().includes(filter))) : rows.slice();
      if (sortCol) {
        const numeric = shown.every((r) => r[sortCol] === '' || isNumCell(r[sortCol]));
        shown.sort((a, b) => {
          const va = numeric ? (a[sortCol] === '' ? -Infinity : +a[sortCol]) : a[sortCol];
          const vb = numeric ? (b[sortCol] === '' ? -Infinity : +b[sortCol]) : b[sortCol];
          return (va < vb ? -1 : va > vb ? 1 : 0) * (asc ? 1 : -1);
        });
      }
      el.querySelector('.count').textContent = `${shown.length} of ${rows.length} rows`;
      el.querySelector('thead').innerHTML = `<tr>${shownCols.map((c) => `<th data-c="${escapeHtml(c)}">${escapeHtml(c)}${c === sortCol ? (asc ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr>`;
      el.querySelectorAll('th').forEach((th) => { th.onclick = () => { asc = sortCol === th.dataset.c ? !asc : true; sortCol = th.dataset.c; render(); }; });
      el.querySelector('tbody').innerHTML = shown.slice(0, 2000).map((r) => `<tr class="${r.Incomplete === 'true' ? 'inc' : ''}">${shownCols.map((c) => {
        const v = r[c];
        const isNum = isNumCell(v) && !/date|Name|Quarter|through/i.test(c);
        return `<td class="${isNum ? 'num' : ''}${v === '' ? ' empty' : ''}">${escapeHtml(isNum ? (+v).toLocaleString('en-US', { maximumFractionDigits: 3 }) : v)}</td>`;
      }).join('')}</tr>`).join('');
    }
    drawSide();
    render();
  }

  // ---------------------------------------------------------------- page shell
  /* Wire tabs, graph/table switch, sidebar and URL hash state around a render function. */
  function mountExplorer({ root, graphTabs, tableTabs, sidebar, state, renderGraph }) {
    const q = root.querySelector.bind(root);
    try {
      const saved = JSON.parse(decodeURIComponent(location.hash.slice(1)) || '{}');
      Object.assign(state, saved);
    } catch (e) { /* ignore a malformed hash */ }
    state.view = state.view || 'graph';
    state.tab = state.tab || graphTabs[0].id;
    state.tableTab = state.tableTab || tableTabs[0].id;

    const renderSidebar = buildSidebar(q('.sidebar .settings'), sidebar, state, update);
    q('.sidebar-toggle').onclick = () => root.classList.toggle('sidebar-hidden');

    function drawTabs() {
      const tabs = state.view === 'graph' ? graphTabs : tableTabs;
      const active = state.view === 'graph' ? state.tab : state.tableTab;
      q('.tabs').innerHTML = tabs.map((t) => `<button data-id="${t.id}" class="${t.id === active ? 'active' : ''}">${t.label}</button>`).join('');
      q('.tabs').querySelectorAll('button').forEach((b) => {
        b.onclick = () => { if (state.view === 'graph') state.tab = b.dataset.id; else state.tableTab = b.dataset.id; update(); };
      });
      q('.view-switch').querySelectorAll('button').forEach((b) => {
        b.classList.toggle('active', b.dataset.view === state.view);
        b.onclick = () => { state.view = b.dataset.view; update(); };
      });
    }

    function saveState() { history.replaceState(null, '', `#${encodeURIComponent(JSON.stringify(state))}`); }
    async function update() {
      saveState();
      drawTabs();
      renderSidebar();
      const graph = state.view === 'graph';
      q('.graph-view').hidden = !graph;
      q('.table-view').hidden = graph;
      q('.sidebar .settings').hidden = !graph;
      q('.sidebar .inspect').hidden = graph;
      root.classList.toggle('table-mode', !graph);
      if (graph) await renderGraph(state);
      else {
        const t = tableTabs.find((x) => x.id === state.tableTab) || tableTabs[0];
        await drawTable(q('.table-view'), q('.sidebar .inspect'), t.csv, t.note, state, saveState);
      }
    }
    let raf = null;
    window.addEventListener('resize', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { if (state.view === 'graph') renderGraph(state); }); });
    update();
    return update;
  }

  return {
    COLORS, DESIGNER_COLORS, CATEGORICAL, METRICS, loadTable, loadCsv, loadChipTypes,
    quarterIntervals, yearIntervals, inInterval, quarterLabel, formatNumber, lowerFirst,
    sequentialPalette, trendLine, trendSummary, drawChart, drawLegend, mountExplorer, escapeHtml,
  };
})();
