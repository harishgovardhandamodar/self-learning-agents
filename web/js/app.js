/* Self-Learning Agent Lab dashboard. Vanilla JS, no build step. */
const $ = (id) => document.getElementById(id);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const api = async (method, path, body) => {
  const r = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let detail = await r.text();
    try { detail = JSON.parse(detail).detail || detail; } catch (e) { /* raw text */ }
    throw new Error(detail);
  }
  return r.json();
};
const get = (p) => api("GET", p);
const post = (p, b) => api("POST", p, b || {});
const put = (p, b) => api("PUT", p, b || {});
const del = (p) => api("DELETE", p);

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let selectedDesign = null;
let selectedRun = null;
let pollTimer = null;
let runFilter = "";
let statusFilter = "";
const glossary = {};
const evolverHelp = {};

/* ------------------------------------------------------------- feedback */
let toastTimer = null;
function toast(msg, kind = "") {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = "toast"; }, 3800);
}
function say(el, msg, kind = "") {
  el.textContent = msg;
  el.className = `msg ${kind}`;
}

/* ----------------------------------------------------------------- tabs */
$$("nav button").forEach((b) => {
  b.onclick = () => {
    $$("nav button").forEach((x) => {
      x.classList.remove("active");
      x.setAttribute("aria-selected", "false");
    });
    $$(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    b.setAttribute("aria-selected", "true");
    $(`tab-${b.dataset.tab}`).classList.add("active");
    if (b.dataset.tab === "compare") { loadSummary(); loadRunsForCompare(); }
    if (b.dataset.tab === "reference") loadReference();
    if (b.dataset.tab === "coverage") loadCoverage();
    if (b.dataset.tab === "suite") { renderSuiteEvolvers(); loadSuiteReport(); loadSuiteAdvice(); }
    if (b.dataset.tab === "patterns") loadPatterns();
  };
});

/* ---------------------------------------------------------------- health */
async function llmHealth() {
  const el = $("llm-health");
  try {
    const h = await get("/api/llm/health");
    const ok = h.fox_services || h.ollama;
    el.textContent = ok ? `llm: ${h.fox_services ? "fox-services" : "ollama"}` : "llm: unreachable";
    el.classList.toggle("ok", !!ok);
  } catch {
    el.textContent = "llm: api down";
    el.classList.remove("ok");
  }
}

/* ------------------------------------------------------------- mermaid */
let mermaidReady = false;
let mermaidLoading = null;

/** Load mermaid once, on demand, and configure it for the dark theme. */
function loadMermaid() {
  if (mermaidReady) return Promise.resolve(true);
  if (mermaidLoading) return mermaidLoading;
  if (typeof window.mermaid === "undefined") return Promise.resolve(false);
  try {
    // mermaid v10's initialize() is synchronous and returns undefined; the
    // promise wrapper keeps the call site uniform across versions.
    const r = window.mermaid.initialize({
      startOnLoad: false,
      theme: "dark",
      securityLevel: "strict",
      themeVariables: {
        darkMode: true,
        background: "#171c24",
        primaryColor: "#1d2836",
        primaryTextColor: "#d7dee9",
        primaryBorderColor: "#5aa9ff",
        lineColor: "#8b94a5",
        secondaryColor: "#1d3a2a",
        tertiaryColor: "#2a3140",
        fontSize: "13px",
      },
      flowchart: { curve: "basis", htmlLabels: true, padding: 10, nodeSpacing: 38, rankSpacing: 46 },
    });
    mermaidReady = true;
    mermaidLoading = Promise.resolve(true);
  } catch (e) {
    mermaidLoading = Promise.resolve(false);
  }
  return mermaidLoading;
}

/**
 * Render mermaid source into a container.
 *
 * A diagram that fails to parse must not take the page down: the source is
 * shown instead, so the user can still read the flow and the author can
 * still fix it.
 */
async function renderMermaid(host, src, id) {
  if (!src) return false;
  const ready = await loadMermaid();
  if (!ready) {
    host.innerHTML = `<pre class="mermaid-src">${esc(src)}</pre>`;
    return false;
  }
  try {
    const { svg } = await window.mermaid.render(`mmd_${id || Math.random().toString(36).slice(2)}`, src);
    host.innerHTML = svg;
    const svgEl = host.querySelector("svg");
    if (svgEl) { svgEl.removeAttribute("height"); svgEl.style.maxWidth = "100%"; }
    return true;
  } catch (e) {
    host.innerHTML = `<div class="mermaid-error"><b>Diagram could not be rendered</b>
      <pre class="mermaid-src">${esc(src)}</pre></div>`;
    return false;
  }
}

/** Render a list of {id,title,mermaid,notes} into a container as cards. */
async function renderDiagrams(hostSel, diagrams) {
  const host = $(hostSel);
  if (!diagrams || !diagrams.length) {
    host.innerHTML = `<div class="empty">No diagrams available.</div>`;
    return;
  }
  host.innerHTML = diagrams.map((d, i) => `<div class="flow-card">
      <h4>${esc(d.title || "Flow")}</h4>
      <div class="flow-body" id="flow_${i}_${(d.id || "d").replace(/[^a-z0-9]/gi, "")}"></div>
      ${(d.notes || []).length ? `<div class="meta">${d.notes.map((n) => esc(n)).join("<br>")}</div>` : ""}
      <details class="raw"><summary>Source</summary>
        <pre class="detail">${esc(d.mermaid || "")}</pre></details>
    </div>`).join("");
  for (let i = 0; i < diagrams.length; i++) {
    const d = diagrams[i];
    const el = $(`flow_${i}_${(d.id || "d").replace(/[^a-z0-9]/gi, "")}`);
    if (el) await renderMermaid(el, d.mermaid, `${i}_${d.id || "d"}`);
  }
}

/* -------------------------------------------------------------- designs */
async function loadDesigns() {
  const { designs } = await get("/api/designs");
  const ev = await get("/api/evolvers");
  const st = await get("/api/streams");
  const pb = await get("/api/playbooks");
  fill($("d-evolver"), ev.evolvers.map((e) => [e.id, `${e.id} — ${e.name}`]));
  fill($("d-stream"), st.streams.map((s) => [s.id, s.name]));
  fill($("d-playbook"), pb.playbooks.map((p) => [p.id, p.title]), true);
  fill($("r-design"), designs.map((d) => [d.id, d.name]), true);
  fill($("r-stream"), st.streams.map((s) => [s.id, s.name]));
  evolverHelp.ev = ev.evolvers;

  $("design-count").textContent = designs.length ? `(${designs.length})` : "";
  const dl = $("design-list");
  if (!designs.length) {
    dl.innerHTML = `<div class="empty">No designs yet.<br>Create one on the right, or start
      from a playbook in the <b>Reference</b> tab.</div>`;
    return;
  }
  dl.innerHTML = designs.map((d) => {
    const spec = d.spec || {};
    return `<div class="card clickable${selectedDesign === d.id ? " sel" : ""}" data-id="${d.id}">
      <h4>${esc(d.name)}</h4>
      <div class="meta">${esc(d.domain)} · ${esc(spec.evolver || "")} · ${esc(spec.stream || "")}</div>
      ${d.description ? `<p>${esc(d.description)}</p>` : ""}
      <div class="card-actions">
        <button class="small" data-act="launch">Run</button>
        <button class="small ghost" data-act="delete">Delete</button>
      </div></div>`;
  }).join("");
  $$("#design-list .card").forEach((card) => {
    card.onclick = async (e) => {
      const act = e.target.dataset.act;
      if (act === "delete") {
        e.stopPropagation();
        if (!confirm("Delete this design?")) return;
        await del(`/api/designs/${card.dataset.id}`);
        toast("Design deleted", "ok");
        if (selectedDesign === card.dataset.id) selectedDesign = null;
        return loadDesigns();
      }
      if (act === "launch") {
        e.stopPropagation();
        const d = await get(`/api/designs/${card.dataset.id}`);
        await launchBatch({ design_id: d.id, stream: d.spec.stream, evolver: d.spec.evolver });
        return;
      }
      selectDesign(card.dataset.id);
    };
  });
}

function fill(sel, items, blank) {
  const cur = sel.value;
  sel.innerHTML = (blank ? `<option value="">(blank)</option>` : "") +
    items.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("");
  if (cur) sel.value = cur;
}

async function selectDesign(id) {
  selectedDesign = id;
  const d = await get(`/api/designs/${id}`);
  $("d-spec").value = JSON.stringify(d.spec, null, 2);
  say($("d-msg"), `Loaded "${d.name}" (${d.id}) — edit the spec and press Save.`, "ok");
  loadDesigns();
  await loadDesignFlow(id);
  const pat = await get(`/api/patterns?domain=${encodeURIComponent(d.domain || "general")}&limit=6`);
  $("d-patterns").innerHTML = pat.patterns.length
    ? pat.patterns.map((p) => `<div class="card"><h4>${esc(p.name)}</h4>
        <p>${esc(p.summary || "")}</p></div>`).join("")
    : `<div class="empty">No catalog mounted. Mount the AI-Agents-Patterns data
       directory (see docker-compose.yml) to see recommendations here.</div>`;
  const g = await get(`/api/guardrails?domain=${encodeURIComponent(d.domain || "general")}`);
  $("d-guardrails").innerHTML = g.guardrails.map((x) => `<div class="card">
      <h4>${esc(x.title)}</h4><p>${esc(x.rule)}</p>
      <div class="meta">enforced by: ${esc(x.enforced_by)}</div></div>`).join("");
}

/** The flow diagrams for the selected design. */
async function loadDesignFlow(id) {
  const host = $("d-flows");
  host.innerHTML = `<div class="empty">Drawing the flow…</div>`;
  try {
    const res = await get(`/api/designs/${id}/flow`);
    await renderDiagrams("d-flows", res.diagrams || []);
  } catch (e) {
    host.innerHTML = `<div class="empty">Could not load the flow diagram: ${esc(e.message || e)}</div>`;
  }
}

$("d-create").onclick = async () => {
  try {
    const pb = $("d-playbook").value;
    const d = pb
      ? await post(`/api/designs/from-playbook/${pb}?name=${encodeURIComponent($("d-name").value)}`)
      : await post("/api/designs", {
          name: $("d-name").value, domain: $("d-domain").value,
          evolver: $("d-evolver").value, stream: $("d-stream").value,
          policy: $("d-policy").value });
    toast(`Created ${d.name}`, "ok");
    await loadDesigns();
    await selectDesign(d.id);
  } catch (e) { toast(String(e.message || e), "err"); }
};

$("d-save").onclick = async () => {
  if (!selectedDesign) return toast("Select a design first", "err");
  try {
    await put(`/api/designs/${selectedDesign}`, { spec: JSON.parse($("d-spec").value) });
    toast("Spec saved", "ok");
    loadDesigns();
  } catch (e) { toast(`Could not save: ${e.message || e}`, "err"); }
};

$("d-launch").onclick = async () => {
  if (!selectedDesign) return toast("Select a design first", "err");
  const d = await get(`/api/designs/${selectedDesign}`);
  await launchBatch({
    design_id: selectedDesign, stream: d.spec.stream,
    evolver: d.spec.evolver, policy: d.spec.policy || "simulated",
    block_size: d.spec.stream_cfg?.block_size || 20,
    n_eval: d.spec.stream_cfg?.n_eval || 30,
  });
};

$("d-export").onclick = async () => {
  if (!selectedDesign) return toast("Select a design first", "err");
  try {
    const m = await post("/api/export", { design_id: selectedDesign });
    toast(`Exported to ${m.directory}`, "ok");
  } catch (e) { toast(String(e.message || e), "err"); }
};

/* ----------------------------------------------------------------- runs */
function renderEvolverPicker() {
  const evs = evolverHelp.ev || [];
  // Default to the full set: a comparison is the point of the lab.
  $("r-evolvers").innerHTML = evs.map((e) =>
    `<label title="${esc(e.description)}">
       <input type="checkbox" value="${e.id}" checked>
       ${esc(e.id)}</label>`).join("");
}
$("r-stream").onchange = () => { /* keep manual choice; presets set it explicitly */ };

$("r-launch").onclick = async () => {
  const evolvers = $$("#r-evolvers input:checked").map((i) => i.value);
  if (!evolvers.length) return toast("Pick at least one evolver", "err");
  await launchBatch({
    design_id: $("r-design").value || null,
    stream: $("r-stream").value,
    evolvers,
    policy: $("r-policy").value,
    iterations: Math.max(1, +$("r-iterations").value || 1),
    block_size: Math.max(1, +$("r-block").value || 20),
    n_eval: Math.max(1, +$("r-eval").value || 30),
  });
};

$("r-launch-one").onclick = async () => {
  const evolvers = $$("#r-evolvers input:checked").map((i) => i.value);
  await launchBatch({
    design_id: $("r-design").value || null,
    stream: $("r-stream").value,
    evolvers: evolvers.length ? evolvers : ["scoped"],
    policy: $("r-policy").value,
    iterations: 1,
    block_size: Math.max(1, +$("r-block").value || 20),
    n_eval: Math.max(1, +$("r-eval").value || 30),
  });
};

$("r-preset-topic").onclick = () => {
  $("r-stream").value = "topic";
  $("r-iterations").value = 5;
  $("r-block").value = 20;
  $("r-eval").value = 30;
  toast("Preset: topic drift, 5 iterations per evolver");
};
$("r-preset-workflow").onclick = () => {
  $("r-stream").value = "workflow";
  $("r-iterations").value = 5;
  $("r-block").value = 20;
  $("r-eval").value = 30;
  toast("Preset: workflow drift, 5 iterations per evolver");
};

async function launchBatch(opts) {
  try {
    const body = {
      design_id: opts.design_id || null,
      stream: opts.stream, evolvers: opts.evolvers || [opts.evolver || "scoped"],
      policy: opts.policy || "simulated",
      iterations: opts.iterations || 1,
      block_size: opts.block_size || 20,
      n_eval: opts.n_eval || 30,
    };
    const res = await post("/api/runs/batch", body);
    say($("r-msg"), `Launched ${res.count} run(s) on ${res.stream}.`, "ok");
    toast(`Launched ${res.count} run(s)`, "ok");
    $("r-progress").hidden = false;
    $("r-bar").style.width = "4%";
    $("r-phase").textContent = "running…";
    loadRuns();
    if (res.runs?.length) watchRun(res.runs[0].run_id);
  } catch (e) {
    say($("r-msg"), `Launch failed: ${e.message || e}`, "err");
    toast(`Launch failed: ${e.message || e}`, "err");
  }
}

$("r-filter").oninput = (e) => { runFilter = e.target.value.toLowerCase(); loadRuns(); };
$("r-status-filter").onchange = (e) => { statusFilter = e.target.value; loadRuns(); };

$("r-clear-junk").onclick = async () => {
  let n = 0;
  for (const st of ["cancelled", "error"]) {
    const r = await del(`/api/runs?only_status=${st}`);
    n += r.deleted || 0;
  }
  toast(n ? `Cleared ${n} failed/cancelled run(s)` : "Nothing to clear", "ok");
  loadRuns();
  loadRunsForCompare();
};

async function loadRuns() {
  let runs = [];
  try { ({ runs } = await get("/api/runs?limit=200")); } catch (e) { return; }
  const filtered = runs.filter((r) => {
    if (statusFilter && r.status !== statusFilter) return false;
    if (!runFilter) return true;
    const hay = `${r.id} ${r.name} ${r.evolver} ${r.stream_id} ${r.policy}`.toLowerCase();
    return hay.includes(runFilter);
  });
  // Render a bounded window: the DOM should not hold hundreds of cards
  // that are re-rendered every few seconds.
  const MAX_ROWS = 60;
  const shown = filtered.slice(0, MAX_ROWS);
  $("run-count").textContent = runs.length
    ? `(${filtered.length} shown${filtered.length < runs.length ? ` of ${runs.length}` : ""})` : "";
  const rl = $("run-list");
  rl.className = "cards scroll";
  if (!shown.length) {
    rl.innerHTML = `<div class="empty">${runs.length ? "No runs match the filter." :
      "No runs yet. Launch a comparison on the left — the preset buttons set up the paper's scenarios."}</div>`;
    return;
  }
  rl.innerHTML = shown.map((r) => {
    const env = r.result && r.result.metrics ? r.result : null;
    const m = env ? env.metrics : null;
    const term = m ? m.terminal_advantage : null;
    const cls = term == null ? "" : (term > 0 ? "ok" : (term < 0 ? "bad" : ""));
    const seeds = r.config ? `seeds ${r.config.train_seed ?? 1}/${r.config.eval_seed ?? 2}` : "";
    return `<div class="card clickable${selectedRun === r.id ? " sel" : ""}" data-id="${r.id}">
      <h4>${esc(r.name || r.id)}
        <span class="st-${r.status}">● ${esc(r.status)}</span>
        ${term != null ? `<span class="pill ${cls}">Term ${term > 0 ? "+" : ""}${term}</span>` : ""}
      </h4>
      <div class="meta">${esc(r.evolver)} · ${esc(r.stream_id)} · ${esc(r.policy)} ·
        ${esc(seeds)} · ${(r.progress * 100).toFixed(0)}% ${r.status === "running" ? "· " + esc(r.message || "") : ""}</div>
      <div class="card-actions">
        ${r.status === "running" || r.status === "pending" ? `<button class="small ghost" data-act="cancel">Cancel</button>` : ""}
        <button class="small ghost" data-act="dup">Repeat</button>
        <button class="small ghost" data-act="del">Delete</button>
      </div></div>`;
  }).join("");
  $$("#run-list .card").forEach((card) => {
    card.onclick = async (e) => {
      const act = e.target.dataset.act;
      const id = card.dataset.id;
      if (act === "cancel") {
        e.stopPropagation();
        await post(`/api/runs/${id}/cancel`);
        toast("Cancelling…");
        return loadRuns();
      }
      if (act === "del") {
        e.stopPropagation();
        await del(`/api/runs/${id}`);
        toast("Run deleted", "ok");
        if (selectedRun === id) selectedRun = null;
        return loadRuns();
      }
      if (act === "dup") {
        e.stopPropagation();
        const r = await post(`/api/runs/${id}/duplicate`);
        toast("Re-launched with fresh task draws", "ok");
        return loadRuns();
      }
      watchRun(id);
    };
  });
}

async function watchRun(id) {
  selectedRun = id;
  loadRuns();
  clearInterval(pollTimer);
  const show = async () => {
    let r;
    try { r = await get(`/api/runs/${id}`); } catch (e) { return; }
    renderRun(r);
    if (r.status === "running" || r.status === "pending") {
      $("r-progress").hidden = false;
      $("r-bar").style.width = `${Math.max(4, (r.progress || 0) * 100)}%`;
      $("r-phase").textContent = r.message || "running…";
      return;
    }
    clearInterval(pollTimer);
    $("r-bar").style.width = "100%";
    $("r-phase").textContent = r.status;
    loadRuns();
  };
  await show();
  pollTimer = setInterval(show, 1500);
}

/* --------------------------------------------------------- run detail */
function renderRun(r) {
  const host = $("run-detail");
  if (r.status === "error" && r.error) {
    host.className = "detail";
    host.innerHTML = `<div class="card" style="border-color:var(--bad)">
      <h4>Run failed</h4><p>${esc(r.message || "")}</p>
      <details class="raw"><summary>Traceback</summary>
      <pre class="detail">${esc(r.error)}</pre></details></div>`;
    return;
  }
  const env = r.result && r.result.metrics ? r.result : null;
  if (!env) {
    host.className = "detail-empty";
    host.textContent = `Run ${r.id} — ${r.status}${r.message ? ": " + r.message : ""}. No results yet.`;
    return;
  }
  const m = env.metrics || {};
  host.className = "";
  const keys = ["terminal_advantage", "average_advantage", "plasticity", "stability",
                "interference", "forgetting"];
  const cards = keys.map((k) => {
    const g = glossary[k] || { short: k, label: k, meaning: "" };
    const v = m[k];
    const good = (k === "stability") ? v > 0 : ((k === "interference" || k === "forgetting") ? v < 1 : v > 0);
    const cls = v == null ? "" : (good ? "good" : "bad");
    const shown = v == null ? "–" : (v > 0 ? "+" : "") + v;
    return `<div class="metric ${cls}" title="${esc(g.meaning)}">
      <div class="k">${esc(g.short)}</div>
      <div class="v">${shown}</div>
      <div class="s">${esc(g.label)}</div>
      <div class="s meaning">${esc(g.meaning)}</div>
    </div>`;
  }).join("");

  // Transfer and shape: the measures the paper's protocol implies but does
  // not tabulate. Without them a run cannot say whether learning generalised.
  const extra = [["forward_transfer", m.forward_transfer],
                 ["backward_transfer", m.backward_transfer],
                 ["area_under_curve", m.area_under_curve]]
    .map(([k, v]) => {
      const g = glossary[k] || { short: k, label: k, meaning: "" };
      const shown = v == null ? "–" : (v > 0 ? "+" : "") + v;
      return `<div class="metric" title="${esc(g.meaning)}">
        <div class="k">${esc(g.short)}</div><div class="v">${shown}</div>
        <div class="s">${esc(g.label)}</div>
        <div class="s meaning">${esc(g.meaning)}</div></div>`;
    }).join("");

  const guard = env.guardrails;
  const guardHtml = guard ? `<div class="metric-grid" style="margin-top:12px">
      <div class="metric ${guard.total ? "bad" : "good"}">
        <div class="k">GUARDRAILS</div>
        <div class="v">${guard.total}</div>
        <div class="s">proposed guardrail violations</div>
        <div class="s meaning">${guard.total
          ? Object.entries(guard.by_violation).map(([k, v]) => `${esc(k)} × ${v}`).join(", ")
          : "The agent proposed no action that breaks a declared guardrail."}</div>
      </div></div>` : "";

  const modes = (env.failure_modes || []);
  const modeHtml = modes.length ? `<ul class="modes">${modes.map((f) =>
    `<li><b>${esc(f.title || f.id)}</b> <span class="pill ${f.severity === "high" ? "bad" : ""}">${esc(f.severity || "")}</span>
      <div class="meta">${esc(f.detail || "")}</div>
      ${f.cause ? `<div class="meta">cause: ${esc(f.cause)}</div>` : ""}</li>`).join("")}</ul>` : "";

  // Judged against the measured bands: is this number normal for this
  // evolver on this stream, or did something change?
  const a = r.assessment;
  const bandHtml = a ? `<div class="row-btns" style="margin-top:12px">
      <span class="verdict ${esc(a.verdict)}">${a.verdict === "expected" ? "as measured" :
        a.verdict === "better" ? "above the measured band" :
        a.verdict === "worse" ? "below the measured band" : "no reference band"}</span>
      ${(a.notes || []).map((n) => `<span class="meta">${esc(n)}</span>`).join("")}
    </div>` : "";

  const ck = r.checkpoints || [];
  const chart = lineChart(ck, m.groups || []);

  host.innerHTML = `
    <div class="row-btns">
      <span class="pill">${esc(r.id)}</span>
      <span class="pill">${esc(r.evolver)}</span>
      <span class="pill">${esc(r.stream_id)}</span>
      <span class="pill">${esc(r.policy)}</span>
      <span class="st-${r.status}">${esc(r.status)}</span>
      <span class="meta">T=${m.T} · ${esc(r.message || "")}</span>
    </div>
    <div class="metric-grid">${cards}</div>
    ${bandHtml}
    <h3 style="margin-top:18px">Transfer and shape</h3>
    <div class="metric-grid">${extra}</div>
    ${guardHtml}
    ${chart}
    ${modeHtml}
    <details class="raw"><summary>Raw metrics &amp; retrieval diagnostics</summary>
      <pre class="detail">${esc(JSON.stringify({ metrics: m, retrieval: env.retrieval, router: env.router }, null, 1))}</pre>
    </details>`;
}

/* ------------------------------------------------------------- charts */
const CHART_COLORS = ["#5aa9ff", "#4cc38a", "#ffb020", "#ff6b6b", "#b98cff", "#3ad9d9"];

function lineChart(checkpoints, groups) {
  if (!checkpoints || checkpoints.length < 2) return "";
  const series = groups.length ? groups : Object.keys(checkpoints[0].metrics || {});
  if (!series.length) return "";
  const W = 720, H = 230, PL = 42, PR = 12, PT = 12, PB = 28;
  const xs = checkpoints.map((c) => c.t);
  const all = series.flatMap((g) => checkpoints.map((c) => (c.metrics || {})[g]))
    .filter((v) => typeof v === "number");
  if (!all.length) return "";
  let lo = Math.min(...all), hi = Math.max(...all);
  const span = Math.max(hi - lo, 5);
  lo = Math.max(0, lo - span * 0.12); hi = Math.min(100, hi + span * 0.12);
  const x = (t) => PL + ((t - Math.min(...xs)) / Math.max(1, Math.max(...xs) - Math.min(...xs))) * (W - PL - PR);
  const y = (v) => PT + (1 - (v - lo) / Math.max(1e-6, hi - lo)) * (H - PT - PB);
  const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const v = lo + f * (hi - lo), yy = y(v);
    return `<line x1="${PL}" y1="${yy}" x2="${W - PR}" y2="${yy}" stroke="#2a3140" stroke-width="1"/>
            <text x="${PL - 6}" y="${yy + 3}" fill="#98a2b3" font-size="10" text-anchor="end">${Math.round(v)}</text>`;
  }).join("");
  const lines = series.map((g, i) => {
    const pts = checkpoints.map((c) => {
      const v = (c.metrics || {})[g];
      return typeof v === "number" ? `${x(c.t).toFixed(1)},${y(v).toFixed(1)}` : null;
    }).filter(Boolean).join(" ");
    const col = CHART_COLORS[i % CHART_COLORS.length];
    return `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round"/>`;
  }).join("");
  const dots = series.map((g, i) => checkpoints.map((cp) => {
    const v = (cp.metrics || {})[g];
    if (typeof v !== "number") return "";
    const col = CHART_COLORS[i % CHART_COLORS.length];
    return `<circle cx="${x(cp.t).toFixed(1)}" cy="${y(v).toFixed(1)}" r="3" fill="${col}"/>`;
  }).join("")).join("");
  const xlabels = checkpoints.map((c) =>
    `<text x="${x(c.t).toFixed(1)}" y="${H - 8}" fill="#98a2b3" font-size="10" text-anchor="middle">t=${c.t}</text>`).join("");
  return `<div class="chart"><h4>Accuracy per family across checkpoints (A[t,k])</h4>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy per family across checkpoints">
      ${grid}${lines}${dots}${xlabels}
    </svg>
    <div class="legend">${series.map((g, i) =>
      `<span><i style="background:${CHART_COLORS[i % CHART_COLORS.length]}"></i>${esc(g)}</span>`).join("")}</div></div>`;
}

/* -------------------------------------------------------------- compare */
$("cmp-go").onclick = async () => {
  const ids = $$("#cmp-list input:checked").map((i) => i.value);
  if (!ids.length) return toast("Select at least one run", "err");
  const { rows } = await post("/api/compare", { run_ids: ids });
  if (!rows.length) { $("cmp-out").innerHTML = `<p class="hint">No completed runs with metrics.</p>`; return; }
  // The glossary arrives with the reference bands; make sure it is loaded so
  // the headers speak the user's language (Term., P, S ...) not raw keys.
  if (!Object.keys(glossary).length) await loadReference();
  const cols = ["terminal_advantage", "average_advantage", "plasticity", "stability", "interference", "forgetting"];
  let h = `<table class="cmp"><tr><th>run</th>${cols.map((c) =>
    `<th title="${esc(glossary[c]?.meaning || "")}">${esc(glossary[c]?.short || c)}</th>`).join("")}<th>failure modes</th></tr>`;
  rows.forEach((r) => {
    h += `<tr><td>${esc(r.name || r.run_id)} <span class="meta">${esc(r.evolver)}/${esc(r.stream)}</span></td>` +
      cols.map((c) => {
        const v = r[c];
        const good = (c === "stability") ? v > 0 : ((c === "interference" || c === "forgetting") ? v < 1 : v > 0);
        return `<td style="color:${v == null ? "inherit" : (good ? "var(--ok)" : "var(--bad)")}">${v == null ? "–" : v}</td>`;
      }).join("") + `<td class="mode">${esc((r.failure_modes || []).join(", ") || "—")}</td></tr>`;
  });
  $("cmp-out").innerHTML = h + "</table>";
};
$("cmp-all").onclick = () => {
  $$("#cmp-list input[type=checkbox]").forEach((i) => { i.checked = true; });
  toast("Selected all finished runs");
};

async function loadRunsForCompare() {
  const { runs } = await get("/api/runs?limit=200");
  const done = runs.filter((r) => r.status === "done").slice(0, 80);
  $("cmp-count").textContent = done.length ? `(${done.length} finished)` : "";
  const cl = $("cmp-list");
  cl.className = "cards scroll";
  if (!done.length) {
    cl.innerHTML = `<div class="empty">No finished runs yet. Launch a comparison first.</div>`;
    return;
  }
  cl.innerHTML = done.map((r) => {
    const env = r.result && r.result.metrics ? r.result : null;
    const term = env ? env.metrics.terminal_advantage : null;
    return `<label class="checkline">
      <input type="checkbox" value="${r.id}">
      <span class="lbl">${esc(r.name || r.id)}
        <span class="sub">${esc(r.evolver)}/${esc(r.stream_id)}${term != null ? ` · Term ${term > 0 ? "+" : ""}${term}` : ""}</span>
      </span></label>`;
  }).join("");
}

async function loadSummary() {
  const { cells } = await get("/api/runs/summary");
  const host = $("cmp-summary");
  if (!cells.length) {
    host.innerHTML = `<div class="empty">No finished runs to aggregate yet.</div>`;
    return;
  }
  host.innerHTML = cells.map((c) => {
    const t = c.metrics.terminal_advantage;
    const p = c.metrics.plasticity;
    return `<div class="card">
      <h4>${esc(c.evolver)} <span class="pill">${esc(c.stream)}</span> <span class="pill">n=${c.n}</span></h4>
      <div class="meta">terminal ${t ? `${t.mean > 0 ? "+" : ""}${t.mean} ± ${t.std}` : "–"} ·
        plasticity ${p ? `${p.mean > 0 ? "+" : ""}${p.mean} ± ${p.std}` : "–"}</div>
      <div class="meta">interference ${c.metrics.interference ? c.metrics.interference.mean.toFixed(2) : "–"} ·
        forgetting ${c.metrics.forgetting ? c.metrics.forgetting.mean.toFixed(2) : "–"}</div>
    </div>`;
  }).join("");
}

/* ------------------------------------------------------------ reference */
async function loadReference() {
  const data = await get("/api/benchmarks");
  Object.assign(glossary, data.glossary || {});
  Object.assign(evolverHelp, data.evolvers || {});
  $("ref-source").textContent = `Measured reference bands — ${data.source || ""}. Bands are
    mean ± std over repeated runs; a run outside its band is flagged.`;
  $("ref-legend").innerHTML = Object.entries(data.glossary || {}).map(([k, g]) =>
    `<div class="card"><h4>${esc(g.short)} · ${esc(g.label)}</h4><p>${esc(g.meaning)}</p></div>`).join("");
  const sc = $("ref-scenario");
  const cur = sc.value;
  sc.innerHTML = Object.keys(data.scenarios || {}).map((s) =>
    `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  if (cur && data.scenarios[cur]) sc.value = cur;
  sc.onchange = () => renderReference(data.scenarios[sc.value], sc.value);
  renderReference(data.scenarios[sc.value], sc.value);
}

function renderReference(scn, id) {
  if (!scn) { $("ref-table").innerHTML = ""; return; }
  $("ref-note").textContent = scn.note || "";
  const cols = ["terminal_advantage", "plasticity", "stability", "interference", "forgetting"];
  let h = `<table class="cmp ref"><tr><th>evolver</th>${cols.map((c) =>
    `<th title="${esc(glossary[c]?.meaning || "")}">${esc(glossary[c]?.short || c)}</th>`).join("")}</tr>`;
  scn.rows.forEach((r) => {
    h += `<tr><td><b>${esc(r.evolver)}</b><div class="meta">${esc(r.description)}</div></td>` +
      cols.map((c) => {
        const m = r.metrics[c];
        if (!m) return "<td>–</td>";
        const s = m.std > 0 ? ` <span class="meta">±${m.std}</span>` : "";
        return `<td>${m.mean > 0 ? "+" : ""}${m.mean}${s}</td>`;
      }).join("") + "</tr>";
  });
  $("ref-table").innerHTML = h + "</table>";
}

/* ------------------------------------------------------------- coverage */
let coverageData = null;

async function loadCoverage() {
  try {
    coverageData = await get("/api/coverage");
  } catch (e) {
    $("cov-scenarios").innerHTML = `<div class="empty">Could not load coverage: ${esc(e.message || e)}</div>`;
    return;
  }
  const t = coverageData.totals;
  $("cov-summary").textContent =
    `${t.scenarios} scenarios · ${t.families} task families · ${t.environments} environments · ` +
    `${t.drift_types} drift types. The grid below is what the suite exercises; the gaps list ` +
    `is what it does not.`;

  $("cov-scenarios").innerHTML = coverageData.scenarios.map((s) => `<div class="card">
      <h4>${esc(s.name)} <span class="pill">K=${s.K}</span></h4>
      <div class="meta">${esc(s.actual_drifts.join(" · "))} — ${esc(s.environments.join(", "))}</div>
      <p>${esc(s.description || "")}</p>
      <div class="meta">${s.families.map((f) => esc(f.name)).join(" | ")}</div>
    </div>`).join("");

  const missing = coverageData.combinations_missing || [];
  $("cov-gaps").innerHTML = missing.length
    ? missing.map((m) => `<div class="card"><h4>${esc(m)}</h4>
        <p>No scenario combines this drift type with this environment.</p></div>`).join("")
    : `<div class="empty">No drift-type/environment combination is missing.</div>`;

  const streams = [...new Set(coverageData.scenarios.map((s) => s.id))];
  fill($("rec-stream"), streams.map((s) => [s, s]));
  $("rec-stream").onchange = loadRecommendation;
  $("rec-metric").onchange = loadRecommendation;
  loadRecommendation();
}

async function loadRecommendation() {
  const stream = $("rec-stream").value;
  const metric = $("rec-metric").value;
  const host = $("rec-out");
  if (!stream) { host.innerHTML = `<div class="empty">No runs yet — launch a comparison first.</div>`; return; }
  host.innerHTML = `<div class="empty">Asking…</div>`;
  try {
    const r = await get(`/api/stats/recommend?stream=${encodeURIComponent(stream)}&metric=${encodeURIComponent(metric)}`);
    if (!r.recommendation) {
      const avail = (r.available_streams || []).length
        ? ` Streams with data: ${r.available_streams.join(", ")}.` : "";
      host.innerHTML = `<div class="empty">${esc(r.reason || "No finished runs yet.")}${esc(avail)}</div>`;
      return;
    }
    const sig = r.significant
      ? `<span class="pill ok">significant</span>`
      : `<span class="pill">not significant</span>`;
    host.innerHTML = `<div class="card">
        <h4>${esc(r.recommendation)} <span class="pill">${esc(r.metric)}</span></h4>
        <div class="meta">mean ${r.mean > 0 ? "+" : ""}${r.mean} ·
          margin over ${esc(r.ranking[1] ? r.ranking[1].evolver : "runner-up")}:
          ${r.margin_over_runner_up != null ? r.margin_over_runner_up.toFixed(2) : "–"} ${sig}</div>
        ${r.caveats && r.caveats.length ? `<p><b>Caveats:</b> ${r.caveats.map(esc).join("<br>")}</p>` : ""}
        <div class="meta">guards: ${Object.entries(r.guards || {}).map(([k, v]) =>
          `${k} ${v == null ? "–" : v}`).join(" · ")}</div>
      </div>
      <div class="table-wrap"><table class="cmp">
        <tr><th>rank</th><th>evolver</th><th>n</th><th>mean</th><th>std</th><th>margin</th><th>significant</th></tr>
        ${r.ranking.map((x) => `<tr><td>${x.rank}</td><td>${esc(x.evolver)}</td>
          <td>${x.n}</td><td>${x.mean > 0 ? "+" : ""}${x.mean}</td><td>±${x.std}</td>
          <td>${x.margin == null ? "–" : x.margin.toFixed(2)}</td>
          <td>${x.significant == null ? "–" : (x.significant ? "yes" : "no")}</td></tr>`).join("")}
      </table></div>`;
  } catch (e) {
    host.innerHTML = `<div class="empty">${esc(e.message || e)}</div>`;
  }
}

/* --------------------------------------------------------------- suite */
let suiteRunIds = [];
let suitePoll = null;

function renderSuiteEvolvers() {
  const evs = evolverHelp.ev || [];
  $("s-evolvers").innerHTML = evs.map((e) =>
    `<label title="${esc(e.description)}">
       <input type="checkbox" value="${e.id}" ${e.id === "scoped" ? "checked" : ""}>
       ${esc(e.id)}</label>`).join("");
}

function suitePayload() {
  return {
    scope: $("s-scope").value,
    evolvers: $$("#s-evolvers input:checked").map((i) => i.value),
    iterations: Math.max(1, +$("s-iterations").value || 5),
    block_size: Math.max(1, +$("s-block").value || 20),
    n_eval: Math.max(1, +$("s-eval").value || 30),
  };
}

$("s-estimate").onclick = async () => {
  try {
    const est = await post("/api/suite/estimate", suitePayload());
    const el = $("s-estimate-out");
    el.style.display = "block";
    el.textContent =
      `${est.runs} runs (${est.scenarios} scenarios × ${est.evolvers} evolvers × ` +
      `${est.iterations} iterations)\n` +
      `${est.training_episodes.toLocaleString()} training episodes, ` +
      `${est.evaluation_episodes.toLocaleString()} evaluation episodes\n` +
      est.note;
  } catch (e) { toast(String(e.message || e), "err"); }
};

$("s-run").onclick = async () => {
  const body = suitePayload();
  if (!body.evolvers.length) return toast("Pick at least one evolver", "err");
  try {
    const res = await post("/api/suite/run", body);
    suiteRunIds = res.runs.map((r) => r.run_id);
    say($("s-msg"), `Launched ${res.count} runs across ${res.scenarios.length} scenarios.`, "ok");
    toast(`Suite launched: ${res.count} runs`, "ok");
    $("s-progress").hidden = false;
    $("s-bar").style.width = "3%";
    $("s-phase").textContent = "running…";
    watchSuite();
  } catch (e) {
    say($("s-msg"), `Suite run failed: ${e.message || e}`, "err");
    toast(`Suite run failed: ${e.message || e}`, "err");
  }
};

async function watchSuite() {
  clearInterval(suitePoll);
  const tick = async () => {
    if (!suiteRunIds.length) return clearInterval(suitePoll);
    let done = 0;
    for (const id of suiteRunIds) {
      try {
        const r = await get(`/api/runs/${id}`);
        if (r.status === "done" || r.status === "error" || r.status === "cancelled") done++;
      } catch (e) { done++; }
    }
    const pct = Math.round((done / suiteRunIds.length) * 100);
    $("s-bar").style.width = `${Math.max(3, pct)}%`;
    $("s-phase").textContent = `${done}/${suiteRunIds.length} runs finished`;
    if (done < suiteRunIds.length) return;
    clearInterval(suitePoll);
    $("s-phase").textContent = "suite complete — reading the report";
    await loadSuiteReport();
    await loadSuiteAdvice();
    toast("Suite complete", "ok");
  };
  await tick();
  suitePoll = setInterval(tick, 3000);
}

async function loadSuiteReport() {
  const metric = $("s-report-metric").value;
  const host = $("s-report");
  host.innerHTML = `<p class="hint">Reading…</p>`;
  try {
    const rep = await get(`/api/suite/report?metric=${encodeURIComponent(metric)}`);
    if (!rep.rows.length) {
      host.innerHTML = `<p class="hint">${esc(rep.note || "No report yet.")}</p>`;
      $("s-power").innerHTML = "";
      return;
    }
    const cols = ["terminal_advantage", "plasticity", "stability", "interference", "forgetting"];
    let h = `<table class="cmp"><tr><th>stream</th><th>evolver</th><th>geometry</th><th>n</th>
      ${cols.map((c) => `<th title="${esc(glossary[c]?.meaning || "")}">${esc(glossary[c]?.short || c)}</th>`).join("")}
      <th>margin</th><th>effect</th><th>significant</th><th>guardrails</th><th>units</th></tr>`;
    rep.rows.forEach((r) => {
      const sig = r.is_best ? `<span class="pill ok">best</span>`
        : (r.significant ? `<span class="pill ok">yes</span>` : `<span class="pill">no</span>`);
      h += `<tr><td>${esc(r.stream)}</td><td>${esc(r.evolver)}${r.is_reference ? ' <span class="pill">ref</span>' : ""}</td>` +
        `<td class="meta">b${r.block_size}/e${r.n_eval}</td><td>${r.n}</td>` +
        cols.map((c) => `<td>${r.metrics[c] == null ? "–" : r.metrics[c]}</td>`).join("") +
        `<td>${r.margin_vs_best == null ? "–" : r.margin_vs_best}</td>` +
        `<td>${r.effect_size == null ? "–" : `${r.effect_size} (${esc(r.effect_label || "")})`}</td>` +
        `<td>${sig}</td>` +
        `<td>${(r.guardrails && r.guardrails.total) || 0}</td>` +
        `<td>${(r.cost && r.cost.total_units) || "–"}</td></tr>`;
    });
    h += `</table>`;
    host.innerHTML = h + `<p class="hint">${esc(rep.note || "")}</p>`;

    // The ceiling: how much of the achievable gain the best real method got.
    if (rep.ceiling && rep.ceiling.length) {
      host.innerHTML += `<h4>How much of the achievable gain was captured</h4>
        <div class="table-wrap"><table class="cmp">
        <tr><th>stream</th><th>best real method</th><th>achieved</th><th>ceiling (oracle)</th>
        <th>captured</th><th>headroom</th></tr>` +
        rep.ceiling.map((c) => `<tr><td>${esc(c.stream)}</td><td>${esc(c.best_real)}</td>
          <td>${c.best_mean > 0 ? "+" : ""}${c.best_mean}</td><td>${c.ceiling > 0 ? "+" : ""}${c.ceiling}</td>
          <td>${c.captured_pct == null ? "–" : c.captured_pct + "%"}</td>
          <td>${c.headroom_pp == null ? "–" : c.headroom_pp.toFixed(2)}</td></tr>`).join("") +
        `</table></div>
        <p class="hint">The ceiling is what perfect scope isolation would deliver -
          an agent that always knew which family a request belonged to. When a
          method captures 100% of it, the remaining gap is in the base model's
          competence, not in the routing.</p>`;
    }

    if (rep.power && rep.power.length) {
      $("s-power").innerHTML = `<table class="cmp"><tr><th>stream</th><th>evolver</th>
        <th>n</th><th>std</th><th>power now</th><th>iterations for 5pp</th></tr>` +
        rep.power.map((p) => `<tr><td>${esc(p.stream)}</td><td>${esc(p.evolver)}</td>
          <td>${p.n}</td><td>${p.std}</td><td>${p.power_now == null ? "–" : p.power_now}</td>
          <td>${p.iterations_needed_for_5pp == null ? "–" : p.iterations_needed_for_5pp}</td></tr>`).join("") +
        `</table>`;
    }
  } catch (e) {
    host.innerHTML = `<p class="hint">${esc(e.message || e)}</p>`;
  }
}

async function loadSuiteAdvice() {
  const domain = $("s-domain").value;
  const host = $("s-advice");
  try {
    const a = await get(`/api/suite/advice?domain=${encodeURIComponent(domain)}`);
    const cal = a.calibration ? `<p><b>Calibration</b><br>${esc(a.calibration.summary)}<br>
      <span class="meta">${esc(a.calibration.reading)}</span></p>` : "";
    host.innerHTML = `<div class="card">
        <h4>${a.recommendation ? `Use <b>${esc(a.recommendation)}</b>` : "No recommendation yet"}</h4>
        ${a.wins && Object.keys(a.wins).length ? `<div class="meta">wins: ${Object.entries(a.wins).map(([k, v]) => `${esc(k)} × ${v}`).join(", ")}</div>` : ""}
        ${a.domain_scenarios && a.domain_scenarios.length ? `<div class="meta">scenarios for ${esc(domain)}: ${a.domain_scenarios.map(esc).join(", ")}</div>` : ""}
        ${a.caveats && a.caveats.length ? `<p><b>Caveats</b><br>${a.caveats.map(esc).join("<br>")}</p>` : ""}
        ${cal}
        ${a.next_steps && a.next_steps.length ? `<p><b>Next</b><br>${a.next_steps.map((s) => "• " + esc(s)).join("<br>")}</p>` : ""}
        ${a.unmeasured && a.unmeasured.length ? `<p><b>Not measured yet</b><br>${a.unmeasured.map(esc).join("<br>")}</p>` : ""}
      </div>`;
  } catch (e) {
    host.innerHTML = `<div class="empty">${esc(e.message || e)}</div>`;
  }
}

$("s-refresh-report").onclick = loadSuiteReport;
$("s-report-metric").onchange = loadSuiteReport;
$("s-domain").onchange = loadSuiteAdvice;
$("s-export-report").onclick = async () => {
  try {
    const metric = $("s-report-metric").value;
    const rep = await get(`/api/suite/report?metric=${encodeURIComponent(metric)}`);
    const blob = new Blob([JSON.stringify(rep, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `suite-report-${metric}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast("Report exported", "ok");
  } catch (e) { toast(String(e.message || e), "err"); }
};

/* ------------------------------------------------------------- patterns */
$("p-domain").onchange = loadPatterns;
async function loadPatterns() {
  const dom = $("p-domain").value;
  const { patterns, available, source } = await get(`/api/patterns?domain=${encodeURIComponent(dom)}&limit=40`);
  const { scenarios, category_diagrams } = await get(`/api/scenarios/catalog?domain=${encodeURIComponent(dom)}`);
  const head = available
    ? `<p class="hint">Catalog: ${esc(source)}</p>`
    : `<div class="empty">Pattern catalog not mounted. Set AGENT_PATTERNS_DIR or mount
       AI-Agents-Patterns/data into the container to browse ${patterns.length} patterns.</div>`;
  $("pattern-list").innerHTML = head + patterns.map((p, i) => {
    const hasDiagram = !!p.has_diagram;
    return `<div class="card pattern-card">
      <h4>${esc(p.name)} <span class="pill">${esc(p.cat || "")}</span></h4>
      <div class="meta">${(p.domains || []).map(esc).join(", ")}</div>
      <p>${esc(p.summary || "")}</p>
      ${(p.failures || []).length ? `<p><b>fails:</b> ${esc(p.failures.join("; "))}</p>` : ""}
      ${(p.protections || []).length ? `<p><b>guards:</b> ${esc(p.protections.join("; "))}</p>` : ""}
      ${hasDiagram ? `<button class="small ghost" data-diagram="${i}">Show flow</button>
        <div class="flow-body" id="pat_${i}" hidden></div>` : ""}
    </div>`;
  }).join("") +
    (scenarios.length ? `<h3>Failure scenarios (${scenarios.length})</h3>` + scenarios.map((s) =>
      `<div class="card"><h4>${esc(s.title || s.id)} <span class="pill">${esc(s.severity || "")}</span></h4>
        <div class="meta">${esc(s.domain || "")}</div><p>${esc(s.description || "")}</p></div>`).join("") : "");

  // Pattern flow buttons: render the diagram on demand so the page stays fast.
  $$("#pattern-list button[data-diagram]").forEach((btn) => {
    btn.onclick = async () => {
      const i = +btn.dataset.diagram;
      const p = patterns[i];
      const host = $(`pat_${i}`);
      if (!host) return;
      if (!host.hidden) { host.hidden = true; btn.textContent = "Show flow"; return; }
      host.hidden = false;
      btn.textContent = "Hide flow";
      if (!host.dataset.rendered) {
        host.innerHTML = `<div class="meta">Drawing…</div>`;
        const res = await get(`/api/patterns/${encodeURIComponent(p.id)}`);
        await renderMermaid(host, res.flow?.mermaid, `pat${i}`);
        host.dataset.rendered = "1";
      }
    };
  });

  await renderDiagrams("category-flows", category_diagrams || []);
}

/* ---------------------------------------------------------------- boot */
(async function boot() {
  await loadDesigns();
  renderEvolverPicker();
  await Promise.all([loadRuns(), loadRunsForCompare(), loadReference(), loadPatterns()]);
  llmHealth();
  setInterval(loadRuns, 8000);
  setInterval(loadRunsForCompare, 15000);
})();
