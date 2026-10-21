/**
 * UI audit for the Self-Learning Agent Lab dashboard.
 *
 * Drives the real dashboard in a real browser and reports what a user would
 * hit: console errors, failed requests, layout overflow, clipped text,
 * tap-target size, unlabeled controls, missing focus ring, and whether the
 * three core workflows (batch launch, compare, design creation) actually work.
 *
 *   node tests/ui_audit.js [baseUrl] [outDir]
 *
 * The container must already be serving the app (docker compose up -d).
 */
const { chromium } = require('playwright-core');
const fs = require('fs');
const path = require('path');

const BASE = process.argv[2] || 'http://localhost:8220';
const OUT = process.argv[3] || '/tmp/sla_ui_audit';
fs.mkdirSync(OUT, { recursive: true });

const problems = [];
const note = (kind, msg, extra = {}) => problems.push({ kind, msg, ...extra });

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await ctx.newPage();

  const consoleErrors = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));
  const badRequests = [];
  page.on('response', (r) => { if (r.status() >= 400) badRequests.push(`${r.status()} ${r.url()}`); });

  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);

  async function auditLayout(label) {
    const res = await page.evaluate(() => {
      const out = { overflowX: false, offenders: [], tiny: [], unlabeled: [], clipped: [] };
      out.overflowX = document.documentElement.scrollWidth > window.innerWidth + 2;
      const vw = window.innerWidth;
      document.querySelectorAll('*').forEach((el) => {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && r.right > vw + 2) {
          out.offenders.push({ tag: el.tagName, id: el.id, right: Math.round(r.right) });
        }
        if (el.children.length === 0 && el.scrollWidth > el.clientWidth + 4 && el.clientWidth > 0) {
          const st = getComputedStyle(el);
          // A <pre> that scrolls on purpose is not clipped text.
          const scrollable = st.overflowX === 'auto' || st.overflowX === 'scroll';
          if (st.overflow !== 'visible' && st.textOverflow !== 'ellipsis' && !scrollable) {
            out.clipped.push({ tag: el.tagName, id: el.id, text: (el.textContent || '').trim().slice(0, 30) });
          }
        }
      });
      document.querySelectorAll('button, a, input, select, textarea, [onclick]').forEach((el) => {
        // A control wrapped in a <label> inherits the label as its tap target.
        const target = el.closest('label') || el;
        const r = target.getBoundingClientRect();
        if (r.width > 0 && r.height > 0 && (r.height < 24 || r.width < 24)) {
          out.tiny.push({ tag: el.tagName, id: el.id, h: Math.round(r.height), w: Math.round(r.width) });
        }
      });
      document.querySelectorAll('input, select, textarea').forEach((el) => {
        const hasLabel = (el.id && document.querySelector(`label[for="${el.id}"]`)) ||
          el.closest('label') || el.getAttribute('aria-label') || el.title;
        if (!hasLabel) out.unlabeled.push({ tag: el.tagName, id: el.id });
      });
      return out;
    });
    if (res.overflowX) note('layout', `${label}: horizontal overflow`, { offenders: res.offenders.slice(0, 6) });
    if (res.tiny.length) note('a11y', `${label}: ${res.tiny.length} sub-24px tap targets`, { sample: res.tiny.slice(0, 6) });
    if (res.unlabeled.length) note('a11y', `${label}: ${res.unlabeled.length} unlabeled form controls`, { sample: res.unlabeled.slice(0, 8) });
    if (res.clipped.length) note('layout', `${label}: ${res.clipped.length} clipped text elements`, { sample: res.clipped.slice(0, 6) });
    return res;
  }

  let shot = 0;
  async function tab(name) {
    await page.click(`nav button[data-tab="${name}"]`);
    await page.waitForTimeout(900);
    await auditLayout(`${name} tab`);
    await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}-${name}.png`), fullPage: true });
  }

  for (const t of ['designs', 'runs', 'compare', 'reference', 'coverage', 'patterns']) await tab(t);

  // ---------------------------------------------- batch launch (core workflow)
  await page.click('nav button[data-tab="runs"]');
  await page.waitForTimeout(500);
  await page.click('#r-preset-workflow');
  await page.fill('#r-iterations', '2');
  await page.fill('#r-block', '8');
  await page.fill('#r-eval', '8');
  const checked = await page.$$eval('#r-evolvers input:checked', (els) => els.length);
  if (checked < 2) note('workflow', `evolver picker has only ${checked} selected`);

  // Clear the history first: the run list is a bounded window, so a full
  // list would hide the runs this launch creates.
  await page.click('#r-clear-junk');
  await page.waitForTimeout(800);
  const before = await page.$$eval('#run-list .card', (els) => els.length);
  await page.click('#r-launch');
  await page.waitForTimeout(3000);
  // The list is a bounded window, so a count comparison is not meaningful.
  // Check that the batch's runs are actually present.
  const batchNames = await page.$$eval('#run-list .card h4',
    (els) => els.map((e) => e.textContent.trim()).filter((t) => /workflow\//.test(t)));
  if (!batchNames.length) note('workflow', 'batch launch did not add runs to the history');
  const after = await page.$$eval('#run-list .card', (els) => els.length);
  if (after < before) note('workflow', 'run list shrank after a launch');
  const msg = await page.textContent('#r-msg');
  if (!/launched/i.test(msg || '')) note('workflow', `no launch confirmation, got: "${msg}"`);

  let finished = 0;
  for (let i = 0; i < 60; i++) {
    await page.waitForTimeout(1000);
    finished = await page.$$eval('#run-list .st-done', (els) => els.length);
    if (finished > 0) break;
  }
  if (!finished) note('workflow', 'no batch run reached a terminal state within 60s');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}-runs-after-batch.png`), fullPage: true });

  // ------------------------------------------------------- run detail view
  const card = await page.$('#run-list .card');
  if (card) {
    await card.click();
    await page.waitForTimeout(1500);
    const detail = await page.textContent('#run-detail');
    if (!detail || detail.trim().length < 40) note('workflow', 'run detail did not render');
    else {
      const cards = await page.$$eval('#run-detail .metric', (els) => els.length);
      if (cards < 6) note('ux', `run detail shows ${cards} metric cards, expected 6`);
      if (!(await page.$$eval('#run-detail .chart svg', (els) => els.length))) {
        note('ux', 'run detail has no accuracy chart');
      }
    }
    await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}-run-detail.png`), fullPage: true });
  }

  // ------------------------------------------------------ compare + summary
  await page.click('nav button[data-tab="compare"]');
  await page.waitForTimeout(1200);
  const boxes = await page.$$('#cmp-list input[type=checkbox]');
  if (!boxes.length) {
    note('workflow', 'compare tab lists no finished runs');
  } else {
    await boxes[0].check();
    await page.click('#cmp-go');
    await page.waitForTimeout(900);
    const out = await page.textContent('#cmp-out');
    if (!/Term\./.test(out || '') && !/terminal/i.test(out || '')) {
      note('ux', 'compare table missing metric headers');
    }
    if (!(await page.$$eval('#cmp-out table', (els) => els.length))) {
      note('workflow', 'compare produced no table');
    }
    if (await page.$eval('#cmp-out', (el) => el.scrollWidth > el.clientWidth + 2)) {
      note('layout', 'compare table overflows its container');
    }
  }
  if (!(await page.$$eval('#cmp-summary .card', (els) => els.length))) {
    note('ux', 'compare tab has no per-cell summary (mean +/- std)');
  }
  await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}-compare.png`), fullPage: true });

  // --------------------------------------------------------- coverage tab
  await page.click('nav button[data-tab="coverage"]');
  await page.waitForTimeout(1500);
  const covScenarios = await page.$$eval('#cov-scenarios .card', (els) => els.length);
  if (covScenarios < 10) note('ux', `coverage tab lists only ${covScenarios} scenarios`);
  const covSummary = await page.textContent('#cov-summary');
  if (!/families/.test(covSummary || '')) note('ux', 'coverage tab has no summary');
  await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}-coverage.png`), fullPage: true });

  // --------------------------------------------------------- reference tab
  await page.click('nav button[data-tab="reference"]');
  await page.waitForTimeout(900);
  if ((await page.$$eval('#ref-table tr', (els) => els.length)) < 3) {
    note('ux', 'reference tab has no measured rows');
  }
  if ((await page.$$eval('#ref-legend .card', (els) => els.length)) < 4) {
    note('ux', 'reference tab lacks the metric glossary');
  }
  if (await page.$eval('#ref-table', (el) => el.scrollWidth > el.clientWidth + 2).catch(() => false)) {
    note('layout', 'reference table overflows its container');
  }
  await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}-reference.png`), fullPage: true });

  // -------------------------------------------------------- design creation
  await page.click('nav button[data-tab="designs"]');
  await page.waitForTimeout(400);
  await page.fill('#d-name', `Audit design ${Date.now()}`);
  await page.click('#d-create');
  await page.waitForTimeout(3000);
  if ((await page.inputValue('#d-spec')).trim().length < 20) {
    note('workflow', 'new design did not populate the spec editor');
  }
  if (!/loaded|created/i.test((await page.textContent('#d-msg')) || '')) {
    note('workflow', 'design message unclear');
  }

  // ------------------------------------------------------------ flow diagrams
  const flowCards = await page.$$eval('#d-flows .flow-card', (els) => els.length);
  if (!flowCards) note('ux', 'design tab shows no flow diagrams');
  const flowSvgs = await page.$$eval('#d-flows .flow-body svg', (els) => els.length);
  const flowErrs = await page.$$eval('#d-flows .mermaid-error', (els) => els.length);
  if (flowCards && !flowSvgs) {
    note('ux', `no flow diagram rendered (${flowErrs} parse errors)`);
  }
  if (flowErrs) note('ux', `${flowErrs} flow diagram(s) failed to parse`);
  await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}-design.png`), fullPage: true });

  // ------------------------------------------------------- patterns + flows
  await page.click('nav button[data-tab="patterns"]');
  await page.waitForTimeout(2500);
  const catSvgs = await page.$$eval('#category-flows .flow-body svg', (els) => els.length);
  const catErrs = await page.$$eval('#category-flows .mermaid-error', (els) => els.length);
  if (catErrs) note('ux', `${catErrs} category diagram(s) failed to parse`);
  const patBtns = await page.$$eval('#pattern-list button[data-diagram]', (els) => els.length);
  if (patBtns) {
    await page.click('#pattern-list button[data-diagram]');
    await page.waitForTimeout(2500);
    const patSvgs = await page.$$eval('#pattern-list .flow-body svg', (els) => els.length);
    if (!patSvgs) note('ux', 'pattern flow diagram did not render on click');
  }

  // ------------------------------------------------------------- focus ring
  await page.keyboard.press('Tab');
  const focusRing = await page.evaluate(() => {
    const el = document.activeElement;
    if (!el) return null;
    const s = getComputedStyle(el);
    return { outline: s.outlineStyle, width: s.outlineWidth, shadow: (s.boxShadow || '').slice(0, 40) };
  });
  if (focusRing && focusRing.outline === 'none' && (!focusRing.shadow || focusRing.shadow === 'none')) {
    note('a11y', 'keyboard focus has no visible indicator');
  }

  // ----------------------------------------------------------------- mobile
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(700);
  await auditLayout('mobile 390px');
  const navRows = await page.evaluate(() => {
    const btns = [...document.querySelectorAll('nav button')];
    return new Set(btns.map((b) => Math.round(b.getBoundingClientRect().top))).size;
  });
  if (navRows > 1) note('responsive', `nav wraps onto ${navRows} rows on mobile`);
  await page.screenshot({ path: path.join(OUT, `${String(++shot).padStart(2, '0')}-mobile.png`), fullPage: true });

  await browser.close();
  fs.writeFileSync(path.join(OUT, 'report.json'),
                   JSON.stringify({ consoleErrors, badRequests, problems }, null, 1));

  console.log('CONSOLE ERRORS:', consoleErrors.length);
  consoleErrors.slice(0, 10).forEach((e) => console.log('  !', e.slice(0, 200)));
  console.log('BAD REQUESTS:', badRequests.length);
  badRequests.slice(0, 10).forEach((e) => console.log('  !', e));
  console.log('PROBLEMS:', problems.length);
  problems.forEach((p) => console.log(`  [${p.kind}] ${p.msg}`));
  if (consoleErrors.length || badRequests.length || problems.length) process.exitCode = 1;
})().catch((e) => { console.error('AUDIT CRASHED:', e); process.exit(1); });
