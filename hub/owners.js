/* AI Chip Owners explorer: who holds the chips, over time or at one quarter.
 * Behaviour follows epoch-website-astro/legacy/vizs/ai-chip-owners/ai-chip-owners.ts,
 * rewritten against the canonical tables in staging/. */
/* global Hub */
'use strict';

(async () => {
  const [quarterly, cumulativeByChip, cumulativeByDesigner, chipTypes] = await Promise.all([
    Hub.loadTable('owners_quarterly_by_chip'),
    Hub.loadTable('owners_cumulative_by_chip'),
    Hub.loadTable('owners_cumulative_by_designer'),
    Hub.loadChipTypes(),
  ]);

  const SMUGGLED = 'China (smuggled)';
  const DESIGNER_LABEL = { Nvidia: 'Nvidia GPUs', AMD: 'AMD GPUs', Google: 'Google TPUs', Huawei: 'Huawei Ascend', Amazon: 'Amazon Trainium', Cambricon: 'Cambricon' };
  const DESIGNER_ORDER = ['Nvidia', 'AMD', 'Google', 'Huawei', 'Amazon', 'Cambricon'];
  const INDIVIDUAL = { Nvidia: ['H100/H200', 'B200', 'B300'], AMD: ['MI300X'], Google: ['TPU v5e', 'TPU v6e', 'TPU v7'],
    Amazon: ['Trainium1', 'Trainium2'], Huawei: ['Ascend 910B', 'Ascend 910C'] };
  const CHINA_TOOLTIP = '“China” means all mainland China customers. Nvidia and AMD figures only include official imports.';
  const release = (chip) => chipTypes.release.get(chip) || new Date(0);

  // Quarterly totals per owner and designer, summed from the by-chip rows. Their
  // intervals can't be summed, so they are left out.
  const quarterlyByDesigner = [...d3.rollup(quarterly, (list) => {
    const r = list[0];
    const m = {};
    for (const k of Object.keys(r.m)) m[k] = { med: d3.sum(list, (x) => x.m[k].med || 0), p5: null, p95: null };
    return { ...r, chip: null, m, incomplete: list.some((x) => x.incomplete) };
  }, (r) => `${r.owner}|${r.designer}|${r.end.getTime()}`).values()];

  // Owner colours, fixed by overall size so they don't shift between views.
  const size = d3.rollup([...cumulativeByDesigner, ...quarterlyByDesigner], (l) => d3.sum(l, (r) => r.m.h100e.med || 0), (r) => r.owner);
  const ownerColor = new Map();
  const palette = Hub.CATEGORICAL.filter((c) => c !== Hub.COLORS.red);
  let slot = 0;
  for (const [owner] of [...size].sort((a, b) => b[1] - a[1])) {
    ownerColor.set(owner, owner === 'Other' ? Hub.COLORS.other : owner === SMUGGLED ? Hub.COLORS.red : palette[slot++ % palette.length]);
  }
  const owners = [...new Set(cumulativeByDesigner.map((r) => r.owner))].filter((o) => o !== 'Other' && o !== SMUGGLED).sort();

  // The default timeline ends at the last quarter every designer covers completely.
  function lastCompleteEnd(rows) {
    const byDesigner = d3.group(rows, (r) => r.designer);
    return d3.min([...byDesigner.values()], (l) => d3.max(l.filter((r) => !r.incomplete), (r) => r.end));
  }
  const DEFAULT_START = new Date(Date.UTC(2024, 0, 1));
  const allQuarters = [...new Set(cumulativeByDesigner.map((r) => r.end.getTime()))].sort((a, b) => a - b).map((t) => new Date(t));
  const defaultEnd = lastCompleteEnd(cumulativeByDesigner);
  const snapshotQuarters = (s) => allQuarters.filter((d) => s.showAllData || (d >= DEFAULT_START && d <= defaultEnd));

  const now = new Date();
  const nowMonth = now.getUTCFullYear() * 12 + now.getUTCMonth();
  const monthLabel = (i) => `${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i % 12]} ${Math.floor(i / 12)}`;

  const sidebar = [
    { type: 'group', open: true, elements: [
      { type: 'pills', id: 'mode', title: 'View', default: 'timeline', options: { timeline: 'Timeline', snapshot: 'Snapshot' } },
      { type: 'range', id: 'snapshotIndex', title: 'Quarter', default: () => Math.max(0, snapshotQuarters({}).length - 1), min: 0,
        max: (s) => Math.max(0, snapshotQuarters(s).length - 1), format: (i) => Hub.quarterLabel(snapshotQuarters(state)[i] || allQuarters[allQuarters.length - 1]),
        showIf: (s) => s.mode === 'snapshot' },
    ] },
    { type: 'group', open: true, elements: [
      { type: 'select', id: 'owner', title: 'Filter by chip owner', default: 'all', options: { all: 'All', ...Object.fromEntries(owners.map((o) => [o, o])) } },
      { type: 'chips', id: 'colorBy', title: 'Color by', default: 'none', options: { none: 'Chip owner', designer: 'Chip designer', chipType: 'Chip type' } },
    ] },
    { type: 'group', title: 'Display', open: true, elements: [
      { type: 'toggle', id: 'cumulative', title: 'Show cumulative', default: true, enableIf: (s) => s.mode === 'timeline',
        tooltip: 'When off, shows chips acquired in each quarter.' },
      { type: 'pills', id: 'proportion', title: 'Show data as', default: 'absolute', options: { absolute: 'Absolute', share: 'Percentage' } },
      { type: 'toggle', id: 'showSmuggled', title: 'Show speculative estimate of smuggled chips', default: false,
        enableIf: (s) => (s.mode === 'snapshot' || s.cumulative) && s.colorBy !== 'chipType' },
      { type: 'toggle', id: 'showTrend', title: 'Show trend', default: false, enableIf: (s) => s.mode === 'timeline' && s.proportion !== 'share' },
      { type: 'toggle', id: 'projectTrend', title: 'Project trend', default: false,
        tooltip: 'Extend the fitted trend, assuming growth continues at the historical rate. An extrapolation, not a forecast.',
        showIf: (s) => s.showTrend && s.mode === 'timeline' && s.proportion !== 'share' },
      { type: 'range', id: 'projectUntil', title: 'Project until end of', default: nowMonth + 12, min: nowMonth + 1, max: nowMonth + 60,
        format: monthLabel, showIf: (s) => s.showTrend && s.projectTrend && s.mode === 'timeline' },
    ] },
    { type: 'group', title: 'Additional settings', open: false, elements: [
      { type: 'toggle', id: 'showAllData', title: 'Show incomplete historic data', default: false },
    ] },
  ];
  const state = {};

  const graphTabs = ['h100e', 'power', 'units'].map((id) => ({ id, label: Hub.METRICS[id].label }));
  const tableTabs = [
    { id: 'd', label: 'Cumulative by designer', csv: 'owners_cumulative_by_designer' },
    { id: 'c', label: 'Cumulative by chip', csv: 'owners_cumulative_by_chip' },
    { id: 'q', label: 'Quarterly by chip', csv: 'owners_quarterly_by_chip' },
    { id: 't', label: 'Chip types', csv: 'chip_types' },
  ];
  const root = document.querySelector('.explorer');

  /* Chip key for the chip-type view: a chip on the short list, or its designer's pool.
   * A designer with four or fewer chip types in context shows them all. */
  function chipGrouper(rows, scopeKey) {
    const counts = d3.rollup(rows, (l) => new Set(l.map((r) => r.chip)).size, scopeKey);
    return (r) => ((INDIVIDUAL[r.designer] || []).includes(r.chip) || (counts.get(scopeKey(r)) || 0) <= 4
      ? r.chip : `Other ${DESIGNER_LABEL[r.designer] || r.designer}`);
  }

  function categorize(rows, s, groupAll) {
    const cat = new Map(); // id -> {id, label, group, color}
    let keyOf;
    if (groupAll && s.colorBy === 'none') {
      keyOf = (r) => r.owner;
      for (const r of rows) cat.set(r.owner, { id: r.owner, label: r.owner, group: '', color: ownerColor.get(r.owner), tooltip: r.owner === 'China' ? CHINA_TOOLTIP : null });
    } else if (groupAll && s.colorBy === 'designer') {
      keyOf = (r) => `${r.owner}__${r.designer}`;
      for (const r of rows) cat.set(keyOf(r), { id: keyOf(r), label: DESIGNER_LABEL[r.designer] || r.designer, group: r.owner, designer: r.designer });
    } else if (groupAll) {
      const chipKey = chipGrouper(rows, (r) => `${r.owner}|${r.designer}`);
      keyOf = (r) => `${r.owner}__${chipKey(r)}`;
      for (const r of rows) cat.set(keyOf(r), { id: keyOf(r), label: chipKey(r), group: r.owner, designer: r.designer, chip: r.chip });
    } else if (s.colorBy === 'none') {
      keyOf = (r) => r.owner;
      for (const r of rows) cat.set(r.owner, { id: r.owner, label: r.owner, group: '', color: ownerColor.get(r.owner) });
    } else if (s.colorBy === 'designer') {
      keyOf = (r) => r.designer;
      for (const r of rows) cat.set(r.designer, { id: r.designer, label: DESIGNER_LABEL[r.designer] || r.designer, group: '', color: Hub.DESIGNER_COLORS[r.designer] });
    } else {
      const chipKey = chipGrouper(rows, (r) => r.designer);
      keyOf = (r) => `${r.designer}__${chipKey(r)}`;
      for (const r of rows) cat.set(keyOf(r), { id: keyOf(r), label: chipKey(r), group: DESIGNER_LABEL[r.designer] || r.designer, designer: r.designer, chip: r.chip, designerGroup: true });
    }
    return { cat, keyOf };
  }

  /* Colour grouped categories: shades of the group's base colour, newest chip lightest,
   * pooled "Other …" chips darkest. Returns legend groups ordered top to bottom. */
  function colorGroups(cats, groupBase) {
    const byGroup = d3.group(cats, (c) => c.group);
    const out = [];
    for (const [group, list] of byGroup) {
      list.sort((a, b) => {
        const ao = a.label.startsWith('Other '); const bo = b.label.startsWith('Other ');
        if (ao !== bo) return ao ? 1 : -1;
        const di = DESIGNER_ORDER.indexOf(a.designer) - DESIGNER_ORDER.indexOf(b.designer);
        return di || (a.chip && b.chip ? release(b.chip) - release(a.chip) : 0);
      });
      const pal = Hub.sequentialPalette(groupBase(group, list[0]), list.length);
      list.forEach((c, i) => { c.color = c.color && !group ? c.color : pal[i]; });
      out.push({ header: group, items: list });
    }
    return out;
  }

  /* Order stacks bottom to top: Other first, then the categories present in the most
   * bars, keeping groups together. Smuggled chips sit directly on top of China. */
  function stackOrder(groups, valuesOf) {
    const ubiq = (items) => ({ n: d3.max(items, (c) => valuesOf(c).filter((v) => v.value > 0).length) || 0,
      t: d3.sum(items, (c) => d3.sum(valuesOf(c), (v) => v.value)) });
    const isOther = (g) => g.header === 'Other' || g.items.every((c) => c.label === 'Other');
    let sorted = [...groups].sort((a, b) => {
      if (isOther(a) !== isOther(b)) return isOther(a) ? -1 : 1;
      const ua = ubiq(a.items); const ub = ubiq(b.items);
      return ub.n - ua.n || ub.t - ua.t;
    });
    const flat = sorted.length === 1 && !sorted[0].header;
    if (flat) {
      let items = [...sorted[0].items].sort((a, b) => {
        if ((a.label === 'Other') !== (b.label === 'Other')) return a.label === 'Other' ? -1 : 1;
        const ua = ubiq([a]); const ub = ubiq([b]);
        return ub.n - ua.n || ub.t - ua.t;
      });
      const sm = items.findIndex((c) => c.label === SMUGGLED);
      if (sm >= 0) { const [x] = items.splice(sm, 1); items.splice(items.findIndex((c) => c.label === 'China') + 1, 0, x); }
      return { stack: items, legend: [{ header: '', items: [...items].reverse() }] };
    }
    const sm = sorted.findIndex((g) => g.header === SMUGGLED);
    if (sm >= 0) { const [x] = sorted.splice(sm, 1); sorted.splice(sorted.findIndex((g) => g.header === 'China') + 1, 0, x); }
    return { stack: sorted.flatMap((g) => [...g.items].reverse()), legend: [...sorted].reverse() };
  }

  function subtractSmuggled(values, catIdOther, catIdSmuggled) {
    const o = values.get(catIdOther); const s = values.get(catIdSmuggled);
    if (!o || !s) return;
    o.forEach((v, i) => { v.value = Math.max(0, v.value - (s[i]?.value || 0)); v.low = null; v.high = null; });
  }

  function renderTimeline(s) {
    const metric = s.tab;
    const cumulative = s.cumulative;
    const smuggled = s.showSmuggled && cumulative && s.colorBy !== 'chipType';
    const chipRows = cumulative ? cumulativeByChip : quarterly;
    const designerRows = cumulative ? cumulativeByDesigner : quarterlyByDesigner;
    const matches = (r) => r.owner === s.owner || (smuggled && s.owner === 'China' && r.owner === SMUGGLED);
    let rows;
    if (s.owner === 'all') rows = s.colorBy === 'chipType' ? chipRows : designerRows;
    else if (s.colorBy === 'chipType') rows = chipRows.filter(matches).length ? chipRows.filter(matches) : designerRows.filter(matches);
    else rows = designerRows.filter(matches);
    if (!smuggled) rows = rows.filter((r) => r.owner !== SMUGGLED);
    if (!rows.length) return emptyChart();

    const minStart = d3.min(rows, (r) => r.start);
    let maxEnd = d3.max(rows, (r) => r.end);
    // By default, stop at the last quarter that every designer in view covers completely.
    const complete = lastCompleteEnd(rows);
    if (!s.showAllData && s.owner === 'all' && complete && complete < maxEnd) maxEnd = complete;
    let intervals = Hub.quarterIntervals(minStart, maxEnd);
    const groupAll = s.owner === 'all' || rows.some((r) => r.owner === SMUGGLED);
    const { cat, keyOf } = categorize(rows, s, groupAll);

    // Forward-fill each owner × designer × chip series in cumulative mode, then sum into categories.
    const values = new Map([...cat.keys()].map((k) => [k, intervals.map(() => ({ value: 0, low: 0, high: 0, n: 0, incomplete: false }))]));
    for (const list of d3.group(rows, (r) => `${r.owner}|${r.designer}|${r.chip}`).values()) {
      const target = values.get(keyOf(list[0]));
      let last = null;
      intervals.forEach((iv, i) => {
        const inside = list.filter((r) => Hub.inInterval(r.end, iv));
        const t = target[i];
        if (inside.length) {
          const r = inside[inside.length - 1];
          last = r;
          t.value += r.m[metric].med || 0; t.low += r.m[metric].p5 || 0; t.high += r.m[metric].p95 || 0; t.n += 1;
          t.incomplete = t.incomplete || r.incomplete;
        } else if (cumulative && last) {
          t.value += last.m[metric].med || 0; t.n += 1; t.incomplete = true;
        }
      });
    }
    for (const arr of values.values()) for (const v of arr) if (v.n !== 1) { v.low = null; v.high = null; }
    if (smuggled && s.owner === 'all') {
      if (s.colorBy === 'none') subtractSmuggled(values, 'Other', SMUGGLED);
      else subtractSmuggled(values, 'Other__Nvidia', `${SMUGGLED}__Nvidia`);
    }

    let keep = intervals.map((_, i) => i);
    if (!s.showAllData && s.owner === 'all') keep = keep.filter((i) => intervals[i].from >= DEFAULT_START);
    intervals = keep.map((i) => intervals[i]);
    for (const [k, arr] of values) values.set(k, keep.map((i) => arr[i]));

    const cats = [...cat.values()].filter((c) => d3.sum(values.get(c.id), (v) => v.value) > 0);
    const legendGroups = colorGroups(cats, (group, first) => (groupAll ? ownerColor.get(group) : Hub.DESIGNER_COLORS[first.designer]) || '#999');
    const { stack, legend } = stackOrder(legendGroups, (c) => values.get(c.id));

    const share = s.proportion === 'share';
    if (share) intervals.forEach((_, i) => { const t = d3.sum(stack, (c) => values.get(c.id)[i].value); for (const c of stack) { const v = values.get(c.id)[i]; v.value = t > 0 ? v.value / t : 0; } });
    const periods = intervals.map((iv) => ({ label: iv.label, long: iv.long, x: iv.to.getTime() }));
    let trend = null;
    if (s.showTrend && !share) {
      const totals = periods.map((p, i) => ({ x: p.x, y: d3.sum(stack, (c) => values.get(c.id)[i].value), incomplete: stack.some((c) => values.get(c.id)[i].incomplete) }));
      const until = s.projectTrend ? new Date(Date.UTC(Math.floor(s.projectUntil / 12), (s.projectUntil % 12) + 1, 1) - 1) : null;
      trend = Hub.trendLine(totals, cumulative, until);
    }
    const m = Hub.METRICS[metric];
    setTitles(s, cumulative ? `Cumulative ${Hub.lowerFirst(m.label)}` : m.label, share, m.tooltip);
    const showCi = s.colorBy !== 'none' || smuggled ? (id) => s.colorBy !== 'none' || id === SMUGGLED : null;
    Hub.drawChart(root.querySelector('.chart'), {
      periods, series: stack.map((c) => ({ id: c.id, label: groupAll && c.group ? `${c.group}: ${c.label}` : c.label, color: c.color, values: values.get(c.id) })),
      type: 'bar', share, trend, showCi, xFormat: (d) => (d.endsWith('*') || d.endsWith('Q1') ? d : d.slice(5)),
    });
    Hub.drawLegend(root.querySelector('.legend'), legend.map((g) => ({ header: g.header, items: g.items })), stack.some((c) => values.get(c.id).some((v) => v.incomplete)));
    footnote(cumulative, trend);
  }

  function renderSnapshot(s) {
    const metric = s.tab;
    const quarters = snapshotQuarters(s);
    const idx = Math.min(s.snapshotIndex ?? quarters.length - 1, quarters.length - 1);
    const q = quarters[idx];
    const smuggled = s.showSmuggled && s.colorBy !== 'chipType';
    const inQuarter = (r) => r.end.getTime() === q.getTime();
    const matches = (r) => s.owner === 'all' || r.owner === s.owner || (smuggled && s.owner === 'China' && r.owner === SMUGGLED);
    const keepOwner = (r) => matches(r) && (smuggled || r.owner !== SMUGGLED);
    const chipRows = cumulativeByChip.filter((r) => inQuarter(r) && keepOwner(r));
    const designerRows = cumulativeByDesigner.filter((r) => inQuarter(r) && keepOwner(r));
    const ownersWithChips = new Set(chipRows.map((r) => r.owner));

    // owner -> key -> {value, low, high, n, designer, label, chip}
    const breakdown = new Map();
    const add = (owner, key, r, extra) => {
      if (!breakdown.has(owner)) breakdown.set(owner, new Map());
      const b = breakdown.get(owner);
      const e = b.get(key) || { value: 0, low: 0, high: 0, n: 0, designer: r.designer, ...extra };
      e.value += r.m[metric].med || 0; e.low += r.m[metric].p5 || 0; e.high += r.m[metric].p95 || 0; e.n += 1;
      b.set(key, e);
    };
    if (s.colorBy === 'none') designerRows.forEach((r) => add(r.owner, r.owner, r, { label: r.owner }));
    else if (s.colorBy === 'designer') designerRows.forEach((r) => add(r.owner, r.designer, r, { label: DESIGNER_LABEL[r.designer] || r.designer }));
    else {
      const chipKey = chipGrouper(chipRows, (r) => `${r.owner}|${r.designer}`);
      chipRows.forEach((r) => add(r.owner, `${r.designer}__${chipKey(r)}`, r, { label: chipKey(r), chip: r.chip }));
      designerRows.filter((r) => !ownersWithChips.has(r.owner)).forEach((r) => add(r.owner, `${r.designer}__${DESIGNER_LABEL[r.designer]}`, r, { label: DESIGNER_LABEL[r.designer] || r.designer }));
    }
    if (smuggled && s.owner === 'all' && breakdown.has('Other') && breakdown.has(SMUGGLED)) {
      const key = s.colorBy === 'none' ? 'Other' : 'Nvidia';
      const smKey = s.colorBy === 'none' ? SMUGGLED : 'Nvidia';
      const o = breakdown.get('Other').get(key); const sm = breakdown.get(SMUGGLED).get(smKey);
      if (o && sm) { o.value = Math.max(0, o.value - sm.value); o.n = 2; }
    }
    if (!breakdown.size) return emptyChart();

    const ownerList = [...breakdown.entries()].map(([owner, b]) => ({ owner, total: d3.sum([...b.values()], (e) => e.value) }))
      .sort((a, b) => (a.owner === 'Other') - (b.owner === 'Other') || b.total - a.total).map((e) => e.owner);
    const keys = new Map();
    for (const b of breakdown.values()) for (const [k, e] of b) if (!keys.has(k)) keys.set(k, e);
    const share = s.proportion === 'share';
    const ownerTotal = new Map(ownerList.map((o) => [o, d3.sum([...breakdown.get(o).values()], (e) => e.value)]));
    const series = [...keys].map(([key, e]) => ({
      id: key, label: e.label, designer: e.designer, chip: e.chip,
      values: ownerList.map((o) => {
        const x = breakdown.get(o).get(key);
        if (!x) return { value: 0 };
        return { value: share ? x.value / (ownerTotal.get(o) || 1) : x.value, low: x.n === 1 ? x.low : null, high: x.n === 1 ? x.high : null };
      }),
    }));
    let legend; let stack;
    if (s.colorBy === 'none') {
      series.forEach((sr) => { sr.color = ownerColor.get(sr.id); });
      stack = [...series].sort((a, b) => ownerList.indexOf(a.id) - ownerList.indexOf(b.id));
      legend = [{ header: '', items: stack.map((c) => ({ ...c, tooltip: c.id === 'China' ? CHINA_TOOLTIP : null })) }];
    } else {
      const byDesigner = d3.group(series, (sr) => sr.designer);
      const order = [...byDesigner.entries()].sort((a, b) => d3.sum(b[1], (x) => d3.sum(x.values, (v) => v.value)) - d3.sum(a[1], (x) => d3.sum(x.values, (v) => v.value)));
      legend = order.map(([designer, list]) => {
        list.sort((a, b) => (a.label.startsWith('Other ')) - (b.label.startsWith('Other ')) || (a.chip && b.chip ? release(b.chip) - release(a.chip) : 0));
        const pal = Hub.sequentialPalette(Hub.DESIGNER_COLORS[designer] || '#999', list.length);
        list.forEach((c, i) => { c.color = pal[i]; });
        return { header: s.colorBy === 'designer' ? '' : DESIGNER_LABEL[designer] || designer, items: list };
      });
      if (s.colorBy === 'designer') legend = [{ header: 'Chip family', items: legend.flatMap((g) => g.items) }];
      stack = [...legend].reverse().flatMap((g) => [...g.items].reverse());
    }
    const m = Hub.METRICS[metric];
    setTitles(s, `Cumulative ${Hub.lowerFirst(m.label)} at the end of ${Hub.quarterLabel(q)}`, share, m.tooltip);
    Hub.drawChart(root.querySelector('.chart'), {
      periods: ownerList.map((o) => ({ label: o, long: `${o}, ${Hub.quarterLabel(q)}` })),
      series: stack, type: 'bar', share, showCi: s.colorBy !== 'none' ? () => true : (id) => id === SMUGGLED,
    });
    Hub.drawLegend(root.querySelector('.legend'), legend, false);
    footnote(true, null);
  }

  function setTitles(s, subtitle, share, tip) {
    const who = s.owner !== 'all' ? `${s.owner} chip ownership` : 'AI chip ownership';
    root.querySelector('.chart-title').textContent = s.mode === 'snapshot' ? who : `${who} over time`;
    const el = root.querySelector('.chart-subtitle');
    el.textContent = share ? `Share of ${Hub.lowerFirst(subtitle.replace(/\s*\(.*?\)/, ''))}` : subtitle;
    el.title = tip;
  }

  function footnote(cumulative, trend) {
    const notes = [];
    if (cumulative) {
      const starts = d3.rollup(cumulativeByDesigner.filter((r) => r.owner !== SMUGGLED), (l) => Hub.quarterLabel(d3.min(l, (r) => r.start)), (r) => r.designer);
      const byStart = d3.groups([...starts], ([, q]) => q).sort((a, b) => (a[0] < b[0] ? -1 : 1));
      notes.push(`Cumulative totals count chips from each designer's first modelled quarter: ${byStart.map(([q, ds]) => `${ds.map(([d]) => d).join(', ')} from ${q}`).join('; ')}.`);
    }
    if (trend) notes.push(`${Hub.trendSummary(trend)}.`);
    root.querySelector('.footnote').textContent = notes.join(' ');
  }

  function emptyChart() {
    root.querySelector('.chart').innerHTML = '<p class="empty">No data for this selection.</p>';
    root.querySelector('.legend').innerHTML = '';
    root.querySelector('.footnote').textContent = '';
  }

  Hub.mountExplorer({ root, graphTabs, tableTabs, sidebar, state, renderGraph: (s) => (s.mode === 'snapshot' ? renderSnapshot(s) : renderTimeline(s)) });
})();
