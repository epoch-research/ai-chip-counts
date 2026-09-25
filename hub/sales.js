/* AI Chip Sales explorer: chips shipped by each designer, per quarter or cumulative.
 * Behaviour follows epoch-website-astro/legacy/vizs/ai-chip-sales/ai-chip-sales.ts,
 * rewritten against the canonical tables in staging/. */
/* global Hub */
'use strict';

(async () => {
  const [quarterly, cumulativeByChip, cumulativeByDesigner, chipTypes, orgs] = await Promise.all([
    Hub.loadTable('sales_quarterly_by_chip'),
    Hub.loadTable('sales_cumulative_by_chip'),
    Hub.loadTable('sales_cumulative_by_designer'),
    Hub.loadChipTypes(),
    Hub.loadCsv('organizations'),
  ]);
  const country = new Map(orgs.map((o) => [o.Name, o.Country]));
  const release = (chip) => chipTypes.release.get(chip) || new Date(0);
  const DESIGNERS = ['Nvidia', 'Amazon', 'Google', 'Huawei', 'AMD', 'Cambricon'];

  // Chips shown on their own in the all-designers chip view; the rest of each designer's
  // line-up is pooled so the legend stays readable.
  const INDIVIDUAL = { Nvidia: ['H100/H200', 'B200', 'B300'], AMD: ['MI300X'], Google: ['TPU v5e', 'TPU v6e', 'TPU v7'] };
  const OTHER_LABEL = { Nvidia: 'Other Nvidia', AMD: 'Other AMD', Google: 'Other TPUs' };

  const now = new Date();
  const nowMonth = now.getUTCFullYear() * 12 + now.getUTCMonth();
  const monthLabel = (i) => `${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i % 12]} ${Math.floor(i / 12)}`;

  const sidebar = [
    { type: 'group', open: true, elements: [
      { type: 'chips', id: 'colorBy', title: 'Color by', default: 'chipType',
        options: { none: 'None', chipType: 'Chip type', designer: 'Designer', country: 'Country of chip designer' } },
      { type: 'select', id: 'designer', title: 'Filter by chip designer', default: 'all',
        options: { all: 'All', ...Object.fromEntries(DESIGNERS.map((d) => [d, d])) },
        enableIf: (s) => s.colorBy === 'chipType' },
    ] },
    { type: 'group', title: 'Display', open: true, elements: [
      { type: 'toggle', id: 'cumulative', title: 'Show cumulative', default: true,
        tooltip: 'When off, shows new additions per quarter. Cumulative totals start at different dates for different designers.' },
      { type: 'pills', id: 'chartType', title: 'Chart type', default: 'bar', options: { bar: 'Bar chart', area: 'Area chart' },
        disabledOptions: (s) => (s.cumulative ? [] : ['area']) },
      { type: 'pills', id: 'proportion', title: 'Show data as', default: 'absolute', options: { absolute: 'Absolute', share: 'Percentage' },
        disabledOptions: (s) => (s.colorBy === 'none' ? ['share'] : []) },
      { type: 'toggle', id: 'showTrend', title: 'Show trend', default: false,
        enableIf: (s) => s.chartType === 'bar' && s.proportion === 'absolute' },
      { type: 'toggle', id: 'projectTrend', title: 'Project trend', default: false,
        tooltip: 'Extend the fitted trend, assuming growth continues at the historical rate. An extrapolation, not a forecast.',
        showIf: (s) => s.showTrend && s.chartType === 'bar' && s.proportion === 'absolute' },
      { type: 'range', id: 'projectUntil', title: 'Project until end of', default: nowMonth + 12, min: nowMonth + 1,
        max: nowMonth + 60, format: monthLabel, showIf: (s) => s.showTrend && s.projectTrend && s.chartType === 'bar' },
    ] },
    { type: 'group', title: 'Additional settings', open: false, elements: [
      { type: 'pills', id: 'timePeriod', title: 'Show time as', default: 'quarterly', options: { quarterly: 'Quarterly', annual: 'Annual' },
        showIf: (s) => s.chartType === 'bar' },
      { type: 'toggle', id: 'showAllData', title: 'Show incomplete historic data', default: false,
        enableIf: (s) => s.colorBy !== 'chipType' || s.designer === 'all' },
    ] },
  ];

  const graphTabs = ['h100e', 'power', 'cost', 'units'].map((id) => ({ id, label: Hub.METRICS[id].label }));
  const tableTabs = [
    { id: 'q', label: 'Quarterly by chip', csv: 'sales_quarterly_by_chip' },
    { id: 'c', label: 'Cumulative by chip', csv: 'sales_cumulative_by_chip' },
    { id: 'd', label: 'Cumulative by designer', csv: 'sales_cumulative_by_designer' },
    { id: 't', label: 'Chip types', csv: 'chip_types' },
    { id: 'o', label: 'Organizations', csv: 'organizations' },
  ];

  const root = document.querySelector('.explorer');

  function categoryFor(row, s) {
    if (s.colorBy === 'none') return 'Total';
    if (s.colorBy === 'designer') return row.designer;
    if (s.colorBy === 'country') return country.get(row.designer) || 'Unknown';
    if (s.designer !== 'all') return row.chip;
    const keep = INDIVIDUAL[row.designer];
    return !keep || keep.includes(row.chip) ? row.chip : OTHER_LABEL[row.designer];
  }

  /* Values of one series (a chip, or a designer total) in each interval. Cumulative rows
   * are stocks: take the latest in the interval and carry it forward. Quarterly rows are
   * flows: sum them. */
  function seriesValues(rows, intervals, metric, cumulative, annual) {
    const lastEnd = d3.max(rows, (r) => r.end);
    let carry = null; let carriedIncomplete = false;
    return intervals.map((iv, i) => {
      const inside = rows.filter((r) => Hub.inInterval(r.end, iv));
      if (cumulative) {
        if (inside.length) {
          const r = inside.reduce((a, b) => (a.end > b.end ? a : b));
          carry = r; carriedIncomplete = r.incomplete;
          return { value: r.m[metric].med || 0, low: r.m[metric].p5, high: r.m[metric].p95, incomplete: r.incomplete, source: true };
        }
        if (!carry) return { value: 0, source: false };
        return { value: carry.m[metric].med || 0, low: carry.m[metric].p5, high: carry.m[metric].p95,
          incomplete: carriedIncomplete || i === intervals.length - 1, source: false };
      }
      if (!inside.length) return { value: 0, source: false };
      const sum = (k) => inside.reduce((a, r) => a + (r.m[metric][k] || 0), 0);
      const partialYear = annual && lastEnd > iv.from && lastEnd < iv.to;
      const sourceIncomplete = inside.some((r) => r.incomplete);
      return { value: sum('med'), low: inside.length === 1 ? inside[0].m[metric].p5 : null,
        high: inside.length === 1 ? inside[0].m[metric].p95 : null,
        incomplete: sourceIncomplete || partialYear, sourceIncomplete, source: true };
    });
  }

  function renderGraph(s) {
    const metric = s.tab;
    const annual = s.timePeriod === 'annual' && s.chartType === 'bar';
    const cumulative = s.cumulative;
    const chartType = cumulative ? s.chartType : 'bar';
    // Designer totals come straight from their own table, except cost: a designer total is
    // blank when any of its chips lacks a price, so cost sums the priced chips instead.
    const byDesignerTable = cumulative && s.colorBy === 'designer' && metric !== 'cost';
    const filterDesigner = s.colorBy === 'chipType' && s.designer !== 'all' ? s.designer : null;
    const showAll = s.showAllData || !!filterDesigner;
    const share = s.proportion === 'share' && s.colorBy !== 'none';

    let rows = byDesignerTable ? cumulativeByDesigner : cumulative ? cumulativeByChip : quarterly;
    if (filterDesigner) rows = rows.filter((r) => r.designer === filterDesigner);

    // One sub-series per chip (or per designer total), later summed into categories.
    const subKey = (r) => (byDesignerTable ? r.designer : `${r.designer}|${r.chip}`);
    const subs = d3.group(rows, subKey);
    const minStart = d3.min(rows, (r) => r.start);
    const maxEnd = d3.max(rows, (r) => r.end);
    let intervals = annual ? Hub.yearIntervals(minStart, maxEnd) : Hub.quarterIntervals(minStart, maxEnd);

    // Categories and their members.
    const categories = new Map();
    for (const [key, list] of subs) {
      const cat = categoryFor(list[0], s);
      const vals = seriesValues(list, intervals, metric, cumulative, annual);
      if (!categories.has(cat)) categories.set(cat, { id: cat, designer: list[0].designer, members: [], chips: new Set() });
      const c = categories.get(cat);
      c.members.push(vals);
      if (list[0].chip) c.chips.add(list[0].chip);
      c.key = key;
    }
    // With one designer selected, keep its ten largest chips and pool the rest.
    if (filterDesigner) {
      const total = (c) => d3.sum(c.members, (m) => d3.sum(m, (v) => v.value));
      const ranked = [...categories.values()].sort((a, b) => total(b) - total(a));
      if (ranked.length > 10) {
        const other = { id: 'Other', designer: filterDesigner, members: [], chips: new Set() };
        for (const c of ranked.slice(10)) { other.members.push(...c.members); categories.delete(c.id); }
        categories.set('Other', other);
      }
    }
    for (const c of categories.values()) {
      c.values = intervals.map((_, i) => {
        const vs = c.members.map((m) => m[i]);
        const single = vs.length === 1 ? vs[0] : null;
        return {
          value: d3.sum(vs, (v) => v.value), low: single ? single.low : null, high: single ? single.high : null,
          incomplete: vs.some((v) => v.incomplete && v.value > 0) || (cumulative && vs.some((v) => v.incomplete)),
          sourceIncomplete: vs.some((v) => v.sourceIncomplete), source: vs.some((v) => v.source),
        };
      });
    }

    // Default window: where every designer's series is under way, up to the last period
    // with complete data. "Show incomplete historic data" lifts it.
    let keep = intervals.map((_, i) => i);
    if (!showAll) {
      const ranges = d3.rollup(rows, (l) => ({ min: d3.min(l, (r) => r.start), max: d3.max(l, (r) => r.end) }), (r) => r.designer);
      const commonStart = d3.max([...ranges.values()], (r) => r.min);
      const commonEnd = d3.min([...ranges.values()], (r) => r.max);
      const cats = [...categories.values()];
      let latest = null;
      for (let i = 0; i < intervals.length; i += 1) {
        const ok = cats.every((c) => (cumulative ? !c.values[i].incomplete : !c.values[i].sourceIncomplete));
        if (ok) latest = intervals[i].to;
      }
      const maxVisible = latest || commonEnd;
      keep = keep.filter((i) => {
        const iv = intervals[i];
        if (chartType === 'area') return commonStart <= iv.to && iv.to <= maxVisible;
        return (commonStart <= iv.from && iv.to <= maxVisible) || (annual && maxVisible < iv.to && maxVisible >= iv.from);
      });
      if (!keep.length) keep = intervals.map((_, i) => i);
    }
    intervals = keep.map((i) => intervals[i]);
    for (const c of categories.values()) c.values = keep.map((i) => c.values[i]);
    if (cumulative) {
      // A series with no row of its own in the last visible period is carried forward.
      for (const c of categories.values()) { const last = c.values[c.values.length - 1]; if (last && !last.source && last.value > 0) last.incomplete = true; }
    }

    // Legend and stack order.
    const total = (c) => d3.sum(c.values, (v) => v.value);
    let groups; let stack;
    if (s.colorBy === 'chipType' && !filterDesigner) {
      const byDesigner = d3.group([...categories.values()].filter((c) => total(c) > 0), (c) => c.designer);
      const designerOrder = [...byDesigner.entries()].sort((a, b) => d3.sum(a[1], total) - d3.sum(b[1], total));
      groups = designerOrder.map(([designer, cats]) => {
        cats.sort((a, b) => {
          const ao = a.id.startsWith('Other'); const bo = b.id.startsWith('Other');
          if (ao !== bo) return ao ? 1 : -1;
          return release([...b.chips][0]) - release([...a.chips][0]);
        });
        const palette = Hub.sequentialPalette(Hub.DESIGNER_COLORS[designer] || '#999', cats.length);
        cats.forEach((c, i) => { c.color = palette[i]; });
        return { header: designer, items: cats.map((c) => ({ id: c.id, label: c.id, color: c.color })) };
      });
      stack = [...designerOrder].reverse().flatMap(([, cats]) => [...cats].reverse());
    } else {
      let cats = [...categories.values()].filter((c) => total(c) > 0);
      if (filterDesigner) {
        cats.sort((a, b) => (a.id === 'Other') - (b.id === 'Other') || release([...b.chips][0]) - release([...a.chips][0]));
        const palette = Hub.sequentialPalette(Hub.DESIGNER_COLORS[filterDesigner], cats.length);
        cats.forEach((c, i) => { c.color = palette[i]; });
      } else {
        cats.sort((a, b) => total(a) - total(b));
        cats.forEach((c, i) => {
          c.color = s.colorBy === 'designer' ? Hub.DESIGNER_COLORS[c.id] : s.colorBy === 'none' ? Hub.COLORS.teal : Hub.CATEGORICAL[(cats.length - 1 - i) % Hub.CATEGORICAL.length];
        });
      }
      groups = s.colorBy === 'none' ? [] : [{ items: cats.map((c) => ({ id: c.id, label: c.id, color: c.color })) }];
      stack = [...cats].reverse();
    }

    if (share) {
      intervals.forEach((_, i) => {
        const t = d3.sum(stack, (c) => c.values[i].value);
        for (const c of stack) c.values[i] = { ...c.values[i], value: t > 0 ? c.values[i].value / t : 0 };
      });
    }

    const periods = intervals.map((iv) => ({ label: iv.label, long: iv.long, x: iv.to.getTime() }));
    let trend = null;
    if (s.showTrend && !share && chartType === 'bar') {
      const totals = periods.map((p, i) => ({ x: p.x, y: d3.sum(stack, (c) => c.values[i].value), incomplete: stack.some((c) => c.values[i].incomplete) }));
      const until = s.projectTrend ? new Date(Date.UTC(Math.floor(s.projectUntil / 12), (s.projectUntil % 12) + 1, 1) - 1) : null;
      trend = Hub.trendLine(totals, cumulative, until);
    }

    const m = Hub.METRICS[metric];
    const label = cumulative ? `Cumulative ${Hub.lowerFirst(m.label)}` : m.label;
    root.querySelector('.chart-subtitle').textContent = share ? `Share of ${Hub.lowerFirst(label.replace(/\s*\(.*\)$/, ''))}` : label;
    root.querySelector('.chart-subtitle').title = m.tooltip;
    const showCi = (s.colorBy === 'chipType' || s.colorBy === 'designer') ? (id) => categories.get(id)?.members.length === 1 : null;
    Hub.drawChart(root.querySelector('.chart'), {
      periods, series: stack.map((c) => ({ id: c.id, label: c.id, color: c.color, values: c.values })),
      type: chartType, share, trend, showCi,
      // "2024 Q1", then "Q2", "Q3", "Q4" until the next year starts.
      xFormat: (d) => (annual || d.endsWith('*') || d.endsWith('Q1') ? d : d.slice(5)),
    });
    Hub.drawLegend(root.querySelector('.legend'), groups, chartType === 'bar' && stack.some((c) => c.values.some((v) => v.incomplete)));

    // Footnote: where each designer's cumulative series starts, read from the data.
    const notes = [];
    if (cumulative) {
      const starts = d3.rollup(cumulativeByDesigner, (l) => Hub.quarterLabel(d3.min(l, (r) => r.start)), (r) => r.designer);
      const byStart = d3.groups([...starts], ([, q]) => q).sort((a, b) => (a[0] < b[0] ? -1 : 1));
      notes.push(`Cumulative totals count chips from each designer's first modelled quarter: ${byStart.map(([q, ds]) => `${ds.map(([d]) => d).join(', ')} from ${q}`).join('; ')}.`);
      if (!showAll) notes.push('The default view starts once every designer’s series has begun.');
    }
    if (trend) notes.push(Hub.trendSummary(trend) + '.');
    root.querySelector('.footnote').textContent = notes.join(' ');
  }

  Hub.mountExplorer({ root, graphTabs, tableTabs, sidebar, state: {}, renderGraph });
})();
