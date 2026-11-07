(function () {
  "use strict";

  const RAW = window.SEG_DATA;
  const SEGS = ["A", "B", "C", "D"];
  const COL = ["#4263eb", "#0ca678", "#f08c00", "#d6336c"];
  const COL_SOFT = ["rgba(66,99,235,.14)", "rgba(12,166,120,.14)", "rgba(240,140,0,.14)", "rgba(214,51,108,.14)"];
  const PROF = ["Artist", "Healthcare", "Entertainment", "Engineer", "Doctor", "Lawyer", "Executive", "Marketing", "Homemaker"];
  const PROF_FR = ["Artiste", "Santé", "Divertissement", "Ingénieur", "Médecin", "Avocat", "Cadre dirigeant", "Marketing", "Au foyer"];
  const SP = ["Low", "Average", "High"];
  const SP_FR = ["Faible", "Moyen", "Élevé"];
  const VAR = ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6", "Cat_7"];
  const AGE_T = [25, 30, 35, 40, 45, 50, 55, 60, 70];
  const AGE_LBL = ["<25", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54", "55–59", "60–69", "70+"];
  const FEAT_NAMES = ["Genre", "Statut marital", "Âge", "Diplômé", "Profession", "Expérience", "Score de dépenses", "Taille du foyer", "Catégorie Var_1"];
  const FEAT_SIZES = [2, 3, 10, 3, 10, 5, 3, 7, 8];
  const fmt = new Intl.NumberFormat("fr-FR");
  const pct = (x, d = 1) => (x * 100).toFixed(d).replace(".", ",") + " %";
  const $ = (id) => document.getElementById(id);

  const rows = [];
  for (let i = 0; i < RAW.length; i += 10) {
    rows.push({ g: RAW[i], m: RAW[i + 1], age: RAW[i + 2], gr: RAW[i + 3], p: RAW[i + 4], we: RAW[i + 5], sp: RAW[i + 6], f: RAW[i + 7], v: RAW[i + 8], s: RAW[i + 9], ref: "H-" + String(i / 10 + 1).padStart(4, "0") });
  }

  const ageBin = (a) => { for (let i = 0; i < AGE_T.length; i++) if (a < AGE_T[i]) return i; return 9; };
  const weBin = (w) => (w < 0 ? 4 : w === 0 ? 0 : w === 1 ? 1 : w <= 5 ? 2 : 3);
  const famBin = (f) => (f < 1 ? 6 : Math.min(f, 6) - 1);
  const features = (r) => [r.g < 0 ? 0 : r.g, r.m < 0 ? 2 : r.m, ageBin(r.age), r.gr < 0 ? 2 : r.gr, r.p < 0 ? 9 : r.p, weBin(r.we), r.sp < 0 ? 0 : r.sp, famBin(r.f), r.v < 0 ? 7 : r.v];

  /* ---------- Naive Bayes ---------- */
  function train(set) {
    const prior = [0, 0, 0, 0];
    const counts = SEGS.map(() => FEAT_SIZES.map((n) => new Array(n).fill(0)));
    set.forEach((r) => { prior[r.s]++; features(r).forEach((v, k) => counts[r.s][k][v]++); });
    const logLik = counts.map((segC, s) => segC.map((c, k) => c.map((x) => Math.log((x + 1) / (prior[s] + FEAT_SIZES[k])))));
    const total = prior.reduce((a, b) => a + b, 0);
    return { logPrior: prior.map((p) => Math.log(p / total)), logLik };
  }
  function predict(model, fv) {
    const sc = SEGS.map((_, s) => fv.reduce((acc, v, k) => (v === null ? acc : acc + model.logLik[s][k][v]), model.logPrior[s]));
    const mx = Math.max(...sc);
    const ex = sc.map((x) => Math.exp(x - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    const probs = ex.map((x) => x / sum);
    let best = 0; probs.forEach((p, i) => { if (p > probs[best]) best = i; });
    return { probs, best };
  }
  function drivers(model, fv, best) {
    return fv.map((v, k) => {
      if (v === null) return null;
      const others = SEGS.map((_, s) => s).filter((s) => s !== best);
      const meanOther = Math.log(others.reduce((a, s) => a + Math.exp(model.logLik[s][k][v]), 0) / others.length);
      return { k, w: model.logLik[best][k][v] - meanOther };
    }).filter(Boolean).sort((a, b) => b.w - a.w);
  }

  const trainSet = rows.filter((_, i) => i % 5 !== 0);
  const valSet = rows.filter((_, i) => i % 5 === 0);
  const valModel = train(trainSet);
  const confusion = SEGS.map(() => [0, 0, 0, 0]);
  let correct = 0;
  valSet.forEach((r) => { const p = predict(valModel, features(r)); r.pred = p; confusion[r.s][p.best]++; if (p.best === r.s) correct++; });
  const accuracy = correct / valSet.length;
  const baseline = Math.max(...[0, 1, 2, 3].map((s) => trainSet.filter((r) => r.s === s).length)) / trainSet.length;
  const model = train(rows);

  /* ---------- Stats ---------- */
  function stats(set) {
    return SEGS.map((_, s) => {
      const g = set.filter((r) => r.s === s), n = g.length || 1;
      const ages = g.reduce((a, r) => a + r.age, 0) / n;
      const fam = g.filter((r) => r.f > 0), we = g.filter((r) => r.we >= 0);
      const prof = PROF.map((_, p) => g.filter((r) => r.p === p).length);
      const topP = prof.indexOf(Math.max(...prof));
      return {
        n: g.length, share: g.length / (set.length || 1), age: ages,
        married: g.filter((r) => r.m === 0).length / n, grad: g.filter((r) => r.gr === 0).length / n,
        low: g.filter((r) => r.sp === 0).length / n, avg: g.filter((r) => r.sp === 1).length / n, high: g.filter((r) => r.sp === 2).length / n,
        fam: fam.reduce((a, r) => a + r.f, 0) / (fam.length || 1), we: we.reduce((a, r) => a + r.we, 0) / (we.length || 1),
        male: g.filter((r) => r.g === 0).length / n, topProf: topP, topProfShare: prof[topP] / n, prof,
      };
    });
  }

  /* ---------- Chart defaults ---------- */
  Chart.defaults.font.family = "Geist, system-ui, sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = "#64748b";
  Chart.defaults.borderColor = "#eef0f3";
  Chart.defaults.plugins.tooltip.backgroundColor = "#0f172a";
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.titleFont = { weight: "600" };
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.pointStyle = "circle";
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.maintainAspectRatio = false;

  /* ---------- Explore filters ---------- */
  const filterDefs = [
    { id: "f-gender", label: "Genre", opts: [["0", "Homme"], ["1", "Femme"]], key: "g" },
    { id: "f-married", label: "Marié(e)", opts: [["0", "Oui"], ["1", "Non"]], key: "m" },
    { id: "f-grad", label: "Diplômé(e)", opts: [["0", "Oui"], ["1", "Non"]], key: "gr" },
    { id: "f-spend", label: "Dépenses", opts: SP_FR.map((l, i) => [String(i), l]), key: "sp" },
    { id: "f-prof", label: "Profession", opts: PROF_FR.map((l, i) => [String(i), l]), key: "p" },
    { id: "f-age", label: "Âge", opts: [["0-29", "18–29 ans"], ["30-44", "30–44 ans"], ["45-59", "45–59 ans"], ["60-120", "60 ans et +"]], key: "age" },
  ];
  const fWrap = $("filters");
  filterDefs.forEach((d) => {
    const w = document.createElement("label");
    w.className = "field";
    w.innerHTML = `<span>${d.label}</span><select id="${d.id}"><option value="">Tous</option>${d.opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select>`;
    fWrap.appendChild(w);
  });
  const filtered = () => rows.filter((r) => filterDefs.every((d) => {
    const v = $(d.id).value;
    if (!v) return true;
    if (d.key === "age") { const [a, b] = v.split("-").map(Number); return r.age >= a && r.age <= b; }
    return r[d.key] === Number(v);
  }));

  /* ---------- KPIs ---------- */
  function renderKpis(set) {
    const st = stats(set);
    $("kpi-total").textContent = fmt.format(set.length);
    $("kpi-total-sub").textContent = set.length === rows.length ? "Clients historiques étiquetés" : `sur ${fmt.format(rows.length)} · filtre actif`;
    $("kpi-segs").innerHTML = st.map((x, s) => `
      <div class="seg-kpi">
        <div class="seg-kpi-top"><span class="dot" style="background:${COL[s]}"></span><span>Segment ${SEGS[s]}</span></div>
        <div class="seg-kpi-val">${fmt.format(x.n)}</div>
        <div class="bar"><span style="width:${(x.share * 100).toFixed(1)}%;background:${COL[s]}"></span></div>
        <div class="muted small">${pct(x.share)} de la sélection</div>
      </div>`).join("");
  }

  /* ---------- Charts ---------- */
  const donut = new Chart($("c-donut"), {
    type: "doughnut",
    data: { labels: SEGS.map((s) => "Segment " + s), datasets: [{ data: [0, 0, 0, 0], backgroundColor: COL, borderColor: "#fff", borderWidth: 3, hoverOffset: 6 }] },
    options: { cutout: "68%", plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${fmt.format(c.raw)} clients · ${pct(c.raw / c.dataset.data.reduce((a, b) => a + b, 0))}` } } } },
  });
  const ageChart = new Chart($("c-age"), {
    type: "bar",
    data: { labels: AGE_LBL, datasets: SEGS.map((s, i) => ({ label: "Segment " + s, data: [], backgroundColor: COL[i], borderRadius: 3, maxBarThickness: 14 })) },
    options: { interaction: { mode: "index", intersect: false }, scales: { x: { grid: { display: false } }, y: { ticks: { callback: (v) => v + " %" } } }, plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label} : ${c.raw.toFixed(1).replace(".", ",")} %` } } } },
  });
  const spendChart = new Chart($("c-spend"), {
    type: "bar",
    data: { labels: SEGS.map((s) => "Segment " + s), datasets: SP_FR.map((l, i) => ({ label: l, data: [], backgroundColor: ["#cbd5e1", "#64748b", "#0f172a"][i], borderRadius: 2, maxBarThickness: 30 })) },
    options: { indexAxis: "y", scales: { x: { stacked: true, max: 100, ticks: { callback: (v) => v + " %" } }, y: { stacked: true, grid: { display: false } } }, plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label} : ${c.raw.toFixed(1).replace(".", ",")} %` } } } },
  });
  const RADAR_AX = ["Marié(e)", "Diplômé(e)", "Dépenses moy./élevées", "Âge moyen", "Taille foyer", "Expérience pro."];
  const radar = new Chart($("c-radar"), {
    type: "radar",
    data: { labels: RADAR_AX, datasets: SEGS.map((s, i) => ({ label: "Segment " + s, data: [], borderColor: COL[i], backgroundColor: COL_SOFT[i], pointBackgroundColor: COL[i], borderWidth: 2, pointRadius: 2 })) },
    options: {
      scales: { r: { min: 0, max: 100, ticks: { display: false, stepSize: 25 }, pointLabels: { font: { size: 11 }, color: "#334155" }, grid: { color: "#e9ecef" }, angleLines: { color: "#e9ecef" } } },
      plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label} : ${c.dataset.real[c.dataIndex]}` } } },
    },
  });
  const varChart = new Chart($("c-var"), {
    type: "bar",
    data: { labels: VAR.map((v) => v.replace("_", " ")), datasets: SEGS.map((s, i) => ({ label: "Segment " + s, data: [], backgroundColor: COL[i], borderRadius: 3, maxBarThickness: 16 })) },
    options: { interaction: { mode: "index", intersect: false }, scales: { x: { grid: { display: false } }, y: { ticks: { callback: (v) => v + " %" } } }, plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => ` ${c.dataset.label} : ${c.raw.toFixed(1).replace(".", ",")} % de la catégorie` } } } },
  });

  function renderCharts(set) {
    const st = stats(set);
    donut.data.datasets[0].data = st.map((x) => x.n); donut.update();
    $("donut-center").textContent = fmt.format(set.length);

    ageChart.data.datasets.forEach((ds, s) => {
      const g = set.filter((r) => r.s === s);
      ds.data = AGE_LBL.map((_, b) => (g.length ? (g.filter((r) => ageBin(r.age) === b).length / g.length) * 100 : 0));
    });
    ageChart.update();

    spendChart.data.datasets.forEach((ds, i) => { ds.data = st.map((x) => [x.low, x.avg, x.high][i] * 100); });
    spendChart.update();

    radar.data.datasets.forEach((ds, s) => {
      const x = st[s];
      ds.data = [x.married * 100, x.grad * 100, (x.avg + x.high) * 100, (x.age / 70) * 100, (x.fam / 5) * 100, (x.we / 5) * 100].map((v) => Math.min(100, v));
      ds.real = [pct(x.married), pct(x.grad), pct(x.avg + x.high), x.age.toFixed(1).replace(".", ",") + " ans", x.fam.toFixed(2).replace(".", ",") + " pers.", x.we.toFixed(1).replace(".", ",") + " ans"];
    });
    radar.update();

    varChart.data.datasets.forEach((ds, s) => {
      ds.data = VAR.map((_, v) => { const c = set.filter((r) => r.v === v); return c.length ? (c.filter((r) => r.s === s).length / c.length) * 100 : 0; });
    });
    varChart.update();

    renderHeatmap(set);
  }

  function renderHeatmap(set) {
    const body = PROF.map((_, p) => {
      const g = set.filter((r) => r.p === p);
      if (!g.length) return "";
      const shares = SEGS.map((_, s) => g.filter((r) => r.s === s).length / g.length);
      const top = shares.indexOf(Math.max(...shares));
      return `<tr><th scope="row">${PROF_FR[p]}<span class="muted small"> · ${fmt.format(g.length)}</span></th>${shares.map((x, s) => `
        <td><div class="heat ${s === top ? "is-top" : ""}" style="background:${hexA(COL[s], 0.06 + x * 1.1)};color:${x > 0.42 ? "#fff" : "#0f172a"}">${(x * 100).toFixed(0)} %</div></td>`).join("")}</tr>`;
    }).join("");
    $("heatmap").innerHTML = `<thead><tr><th scope="col">Profession</th>${SEGS.map((s, i) => `<th scope="col"><span class="dot" style="background:${COL[i]}"></span>${s}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
  }
  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${Math.min(1, a).toFixed(3)})`;
  }

  function refreshExplore() {
    const set = filtered();
    renderKpis(set);
    renderCharts(set);
    const active = filterDefs.filter((d) => $(d.id).value).length;
    $("f-reset").hidden = !active;
    $("f-count").textContent = active ? `${active} filtre${active > 1 ? "s" : ""} actif${active > 1 ? "s" : ""} · ${fmt.format(set.length)} clients` : "Aucun filtre · base complète";
  }
  filterDefs.forEach((d) => $(d.id).addEventListener("change", refreshExplore));
  $("f-reset").addEventListener("click", () => { filterDefs.forEach((d) => ($(d.id).value = "")); refreshExplore(); });

  /* ---------- Model KPIs & confusion ---------- */
  $("kpi-acc").textContent = pct(accuracy);
  $("kpi-acc-sub").textContent = `vs ${pct(baseline)} au hasard pondéré · ×${(accuracy / baseline).toFixed(1).replace(".", ",")}`;
  $("confusion").innerHTML = `<thead><tr><th scope="col">Réel \\ Prédit</th>${SEGS.map((s) => `<th scope="col">${s}</th>`).join("")}<th scope="col">Rappel</th></tr></thead><tbody>${confusion.map((row, i) => {
    const t = row.reduce((a, b) => a + b, 0);
    return `<tr><th scope="row"><span class="dot" style="background:${COL[i]}"></span>${SEGS[i]}</th>${row.map((x, j) => `<td><div class="heat sm" style="background:${i === j ? hexA(COL[i], 0.12 + (x / t) * 1.2) : "rgba(15,23,42," + ((x / t) * 0.35).toFixed(3) + ")"};color:${i === j && x / t > 0.4 ? "#fff" : "#0f172a"}">${x}</div></td>`).join("")}<td class="num">${pct(row[i] / t, 0)}</td></tr>`;
  }).join("")}</tbody>`;

  /* ---------- Simulator ---------- */
  const simFields = [
    { id: "s-g", label: "Genre", opts: ["Homme", "Femme"] },
    { id: "s-m", label: "Déjà marié(e)", opts: ["Oui", "Non"] },
    { id: "s-age", label: "Âge", type: "number" },
    { id: "s-gr", label: "Diplômé(e)", opts: ["Oui", "Non"] },
    { id: "s-p", label: "Profession", opts: PROF_FR },
    { id: "s-we", label: "Expérience (années)", type: "number2" },
    { id: "s-sp", label: "Score de dépenses", opts: SP_FR },
    { id: "s-f", label: "Taille du foyer", opts: ["1", "2", "3", "4", "5", "6+"] },
    { id: "s-v", label: "Catégorie Var_1", opts: VAR.map((v) => v.replace("_", " ")) },
  ];
  const simDefaults = { "s-g": 1, "s-m": 0, "s-age": 46, "s-gr": 0, "s-p": 0, "s-we": 1, "s-sp": 1, "s-f": 2, "s-v": 5 };
  $("sim-form").innerHTML = simFields.map((f) => {
    if (f.type === "number") return `<label class="field"><span>${f.label} <output id="s-age-out">${simDefaults[f.id]}</output> ans</span><input type="range" id="${f.id}" min="18" max="89" value="${simDefaults[f.id]}" /></label>`;
    if (f.type === "number2") return `<label class="field"><span>${f.label} <output id="s-we-out">${simDefaults[f.id]}</output></span><input type="range" id="${f.id}" min="0" max="14" value="${simDefaults[f.id]}" /></label>`;
    return `<label class="field"><span>${f.label}</span><select id="${f.id}">${f.opts.map((o, i) => `<option value="${i}" ${i === simDefaults[f.id] ? "selected" : ""}>${o}</option>`).join("")}</select></label>`;
  }).join("");
  function runSim() {
    const v = (id) => Number($(id).value);
    $("s-age-out").textContent = v("s-age");
    $("s-we-out").textContent = v("s-we");
    const r = { g: v("s-g"), m: v("s-m"), age: v("s-age"), gr: v("s-gr"), p: v("s-p"), we: v("s-we"), sp: v("s-sp"), f: v("s-f") + 1, v: v("s-v") };
    const fv = features(r);
    const { probs, best } = predict(model, fv);
    const sorted = [...probs].sort((a, b) => b - a);
    const margin = sorted[0] - sorted[1];
    const conf = sorted[0] >= 0.6 ? ["Élevée", "ok"] : sorted[0] >= 0.42 ? ["Modérée", "mid"] : ["Faible", "low"];
    $("sim-result").innerHTML = `
      <div class="sim-head">
        <div class="seg-badge" style="background:${COL[best]}">${SEGS[best]}</div>
        <div>
          <div class="muted small">Segment recommandé</div>
          <div class="sim-title">Rattacher au segment ${SEGS[best]}</div>
          <div class="chip ${conf[1]}">Confiance ${conf[0]} · marge ${pct(margin, 0)}</div>
        </div>
      </div>
      <div class="prob-list" role="list">${probs.map((p, s) => `
        <div class="prob" role="listitem">
          <span class="prob-l"><span class="dot" style="background:${COL[s]}"></span>${SEGS[s]}</span>
          <div class="prob-track"><span style="width:${(p * 100).toFixed(1)}%;background:${COL[s]}"></span></div>
          <span class="prob-v num">${pct(p)}</span>
        </div>`).join("")}</div>
      <div class="muted small label-up">Facteurs déterminants</div>
      <div class="drivers">${drivers(model, fv, best).slice(0, 4).map((d) => `<span class="driver ${d.w >= 0 ? "pos" : "neg"}">${FEAT_NAMES[d.k]} <b>${d.w >= 0 ? "+" : "−"}${Math.abs(d.w).toFixed(2).replace(".", ",")}</b></span>`).join("")}</div>`;
  }
  simFields.forEach((f) => $(f.id).addEventListener("input", runSim));

  /* ---------- Decision grid ---------- */
  function renderGrid() {
    const band = $("g-age").value, mar = $("g-mar").value;
    const set = rows.filter((r) => {
      if (mar && r.m !== Number(mar)) return false;
      if (!band) return true;
      const [a, b] = band.split("-").map(Number); return r.age >= a && r.age <= b;
    });
    $("grid").innerHTML = `<thead><tr><th scope="col">Profession \\ Dépenses</th>${SP_FR.map((l) => `<th scope="col">${l}</th>`).join("")}</tr></thead><tbody>${PROF.map((_, p) => `<tr><th scope="row">${PROF_FR[p]}</th>${SP.map((_, sp) => {
      const g = set.filter((r) => r.p === p && r.sp === sp);
      if (g.length < 8) return `<td><div class="cell empty" title="Effectif insuffisant">n &lt; 8</div></td>`;
      const sh = SEGS.map((_, s) => g.filter((r) => r.s === s).length / g.length);
      const top = sh.indexOf(Math.max(...sh));
      return `<td><div class="cell" style="border-color:${COL[top]};background:${hexA(COL[top], 0.06 + sh[top] * 0.25)}" title="${sh.map((x, i) => SEGS[i] + " " + pct(x, 0)).join(" · ")}">
        <span class="cell-seg" style="background:${COL[top]}">${SEGS[top]}</span><span class="cell-p num">${pct(sh[top], 0)}</span><span class="muted xsmall num">n=${g.length}</span></div></td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
  }
  $("g-age").addEventListener("change", renderGrid);
  $("g-mar").addEventListener("change", renderGrid);

  /* ---------- Batch scoring ---------- */
  let scored = valSet.map((r) => ({ id: r.ref, r, pred: r.pred, truth: r.s }));
  let source = "val";
  let page = 0;
  const PAGE = 12;
  function parseCsv(text) {
    const lines = text.replace(/\r/g, "").split("\n").filter((l) => l.trim());
    const head = lines[0].split(",").map((h) => h.trim());
    const idx = (n) => head.indexOf(n);
    const need = ["Gender", "Ever_Married", "Age", "Graduated", "Profession", "Work_Experience", "Spending_Score", "Family_Size", "Var_1"];
    const miss = need.filter((n) => idx(n) < 0);
    if (miss.length) throw new Error("Colonnes manquantes : " + miss.join(", "));
    return lines.slice(1).map((l) => {
      const c = l.split(",").map((x) => x.trim());
      const get = (n) => c[idx(n)] ?? "";
      const num = (n) => (get(n) === "" ? -1 : Math.round(parseFloat(get(n))));
      const r = {
        g: get("Gender") === "Female" ? 1 : 0, m: get("Ever_Married") === "Yes" ? 0 : get("Ever_Married") === "No" ? 1 : -1,
        age: num("Age") < 0 ? 40 : num("Age"), gr: get("Graduated") === "Yes" ? 0 : get("Graduated") === "No" ? 1 : -1,
        p: PROF.indexOf(get("Profession")), we: num("Work_Experience"), sp: Math.max(0, SP.indexOf(get("Spending_Score"))), f: num("Family_Size"), v: VAR.indexOf(get("Var_1")),
      };
      const truth = idx("Segmentation") >= 0 ? SEGS.indexOf(get("Segmentation")) : -1;
      return { id: idx("ID") >= 0 ? get("ID") : "—", r, pred: predict(model, features(r)), truth };
    });
  }
  $("upload").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        scored = parseCsv(String(reader.result));
        source = "new";
        $("upload-msg").textContent = `${fmt.format(scored.length)} clients importés depuis « ${file.name} » et classifiés.`;
        $("upload-msg").className = "upload-msg ok";
        page = 0; renderBatch();
      } catch (err) {
        $("upload-msg").textContent = err.message;
        $("upload-msg").className = "upload-msg err";
      }
    };
    reader.readAsText(file);
  });
  ["b-seg", "b-prof", "b-spend", "b-conf"].forEach((id) => $(id).addEventListener("change", () => { page = 0; renderBatch(); }));
  $("b-prev").addEventListener("click", () => { page = Math.max(0, page - 1); renderBatch(); });
  $("b-next").addEventListener("click", () => { page++; renderBatch(); });

  function batchFiltered() {
    const seg = $("b-seg").value, prof = $("b-prof").value, sp = $("b-spend").value, conf = Number($("b-conf").value);
    return scored.filter((x) => (seg === "" || x.pred.best === Number(seg)) && (prof === "" || x.r.p === Number(prof)) && (sp === "" || x.r.sp === Number(sp)) && x.pred.probs[x.pred.best] >= conf);
  }
  function renderBatch() {
    const isVal = source === "val";
    $("b-source").textContent = isVal ? "Échantillon de validation (clients historiques non vus à l'entraînement)" : "Nouveaux clients importés";
    $("b-truth-h").hidden = !isVal;
    const counts = [0, 0, 0, 0]; scored.forEach((x) => counts[x.pred.best]++);
    $("b-summary").innerHTML = counts.map((c, s) => `
      <button type="button" class="sum ${$("b-seg").value === String(s) ? "is-active" : ""}" data-seg="${s}" aria-pressed="${$("b-seg").value === String(s)}">
        <span class="dot" style="background:${COL[s]}"></span><span class="sum-l">${SEGS[s]}</span>
        <span class="sum-v num">${fmt.format(c)}</span><span class="muted xsmall num">${pct(c / (scored.length || 1), 0)}</span>
      </button>`).join("");
    $("b-summary").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      $("b-seg").value = $("b-seg").value === b.dataset.seg ? "" : b.dataset.seg; page = 0; renderBatch();
    }));
    const list = batchFiltered();
    const pages = Math.max(1, Math.ceil(list.length / PAGE));
    page = Math.min(page, pages - 1);
    const slice = list.slice(page * PAGE, page * PAGE + PAGE);
    $("b-body").innerHTML = slice.length ? slice.map((x) => {
      const p = x.pred.probs[x.pred.best];
      return `<tr>
        <td class="mono">${x.id}</td>
        <td>${x.r.g === 1 ? "F" : "H"} · ${x.r.age} ans</td>
        <td>${x.r.p >= 0 ? PROF_FR[x.r.p] : "—"}</td>
        <td>${SP_FR[x.r.sp]}</td>
        <td>${x.r.m === 0 ? "Oui" : x.r.m === 1 ? "Non" : "—"}</td>
        <td><span class="pill" style="background:${hexA(COL[x.pred.best], 0.12)};color:${COL[x.pred.best]}">Segment ${SEGS[x.pred.best]}</span></td>
        <td><div class="conf"><div class="prob-track sm"><span style="width:${(p * 100).toFixed(0)}%;background:${COL[x.pred.best]}"></span></div><span class="num small">${pct(p, 0)}</span></div></td>
        ${isVal ? `<td>${x.truth === x.pred.best ? '<span class="ok-t">✓ ' + SEGS[x.truth] + "</span>" : '<span class="ko-t">' + SEGS[x.truth] + "</span>"}</td>` : ""}
      </tr>`;
    }).join("") : `<tr><td colspan="8" class="empty-row">Aucun client ne correspond à ces critères.</td></tr>`;
    $("b-page").textContent = `${fmt.format(list.length)} résultats · page ${page + 1} / ${pages}`;
    $("b-prev").disabled = page === 0;
    $("b-next").disabled = page >= pages - 1;
  }
  $("b-export").addEventListener("click", () => {
    const list = batchFiltered();
    const csv = ["ID,Gender,Age,Profession,Spending_Score,Segment_Predit,Probabilite,P_A,P_B,P_C,P_D"].concat(list.map((x) =>
      [x.id, x.r.g === 1 ? "Female" : "Male", x.r.age, x.r.p >= 0 ? PROF[x.r.p] : "", SP[x.r.sp], SEGS[x.pred.best], x.pred.probs[x.pred.best].toFixed(3), ...x.pred.probs.map((p) => p.toFixed(3))].join(","))).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = source === "val" ? "validation_segmentation.csv" : "nouveaux_clients_segmentes.csv";
    a.click();
  });
  $("b-prof").innerHTML = `<option value="">Toutes</option>` + PROF_FR.map((l, i) => `<option value="${i}">${l}</option>`).join("");

  /* ---------- Recommendations ---------- */
  function renderReco() {
    const st = stats(rows);
    const value = st.map((x) => x.low * 1 + x.avg * 2 + x.high * 3);
    const potential = st.map((x, s) => value[s] * x.n);
    const totPot = potential.reduce((a, b) => a + b, 0);
    const order = [0, 1, 2, 3].sort((a, b) => value[b] - value[a]);
    const youngest = st.map((x) => x.age).indexOf(Math.min(...st.map((x) => x.age)));

    function persona(x, s) {
      if (s === youngest && x.age < 38) return {
        name: "Jeunes actifs à budget maîtrisé", tone: "Acquisition",
        offer: "Citadines et compactes d'entrée de gamme, véhicules d'occasion certifiés, offres LOA/LLD à mensualité basse.",
        channels: ["Réseaux sociaux", "Influence", "Campagnes mobiles", "Partenariats écoles & employeurs"],
        message: "« Votre première voiture, sans compromis sur le budget. »",
        kpi: "Coût par lead, taux de souscription au financement",
      };
      if (x.avg >= 0.4) return {
        name: "Familles établies, cœur de marché", tone: "Montée en gamme",
        offer: "SUV familiaux, monospaces et finitions intermédiaires ; packs sécurité et confort ; reprise de l'ancien véhicule.",
        channels: ["CRM e-mail personnalisé", "Essais en concession", "Événements week-end", "Programme de parrainage"],
        message: "« L'espace et la sérénité pour toute la famille. »",
        kpi: "Panier moyen, taux d'upsell sur les options",
      };
      if (x.married >= 0.7 && x.avg + x.high >= 0.45) return {
        name: "Couples matures à fort potentiel", tone: "Fidélisation premium",
        offer: "Berlines et SUV premium, motorisations hybrides, extensions de garantie et contrats d'entretien.",
        channels: ["Conseiller dédié", "Invitations VIP", "Presse et digital premium", "Programme de fidélité"],
        message: "« Le plaisir de conduire, accompagné à chaque étape. »",
        kpi: "Valeur vie client, taux de renouvellement",
      };
      return {
        name: "Profils pragmatiques sensibles au prix", tone: "Conversion",
        offer: "Modèles à fort rapport qualité-prix, promotions ciblées, primes à la conversion et offres de fin de série.",
        channels: ["Retargeting display", "SMS promotionnels", "Comparateurs en ligne", "Portes ouvertes"],
        message: "« La bonne voiture au juste prix, maintenant. »",
        kpi: "Taux de conversion, coût d'acquisition",
      };
    }

    $("reco-priority").innerHTML = order.map((s, rank) => `
      <div class="prio">
        <span class="prio-rank">${rank + 1}</span>
        <span class="seg-badge sm" style="background:${COL[s]}">${SEGS[s]}</span>
        <div class="prio-main">
          <div class="prio-top"><span>${persona(st[s], s).name}</span><span class="num">${pct(potential[s] / totPot, 0)} du budget</span></div>
          <div class="bar"><span style="width:${((potential[s] / totPot) * 100).toFixed(1)}%;background:${COL[s]}"></span></div>
        </div>
      </div>`).join("");

    $("reco-cards").innerHTML = SEGS.map((S, s) => {
      const x = st[s], p = persona(x, s);
      const ev = [
        `Âge moyen <b>${x.age.toFixed(1).replace(".", ",")} ans</b>`,
        `<b>${pct(x.married, 0)}</b> marié(e)s · <b>${pct(x.grad, 0)}</b> diplômé(e)s`,
        `Dépenses : <b>${pct(x.low, 0)}</b> faibles · <b>${pct(x.avg, 0)}</b> moyennes · <b>${pct(x.high, 0)}</b> élevées`,
        `Profession dominante : <b>${PROF_FR[x.topProf]}</b> (${pct(x.topProfShare, 0)})`,
        `Foyer moyen <b>${x.fam.toFixed(1).replace(".", ",")}</b> pers.`,
      ];
      return `<article class="card reco" style="--seg:${COL[s]}">
        <header class="reco-h">
          <span class="seg-badge" style="background:${COL[s]}">${S}</span>
          <div><div class="muted xsmall label-up">${p.tone} · ${fmt.format(x.n)} clients</div><h3>${p.name}</h3></div>
        </header>
        <ul class="evidence">${ev.map((e) => `<li>${e}</li>`).join("")}</ul>
        <dl class="reco-dl">
          <div><dt>Offre produit</dt><dd>${p.offer}</dd></div>
          <div><dt>Message clé</dt><dd class="quote">${p.message}</dd></div>
          <div><dt>Canaux prioritaires</dt><dd class="tags">${p.channels.map((c) => `<span>${c}</span>`).join("")}</dd></div>
          <div><dt>KPI de pilotage</dt><dd>${p.kpi}</dd></div>
        </dl>
      </article>`;
    }).join("");

    const signals = [];
    PROF.forEach((_, pi) => {
      const g = rows.filter((r) => r.p === pi);
      SEGS.forEach((S, s) => { const sh = g.filter((r) => r.s === s).length / g.length; signals.push({ t: `${PROF_FR[pi]} → ${S}`, sh, n: g.length, s }); });
    });
    [0, 1, 2].forEach((sp) => {
      const g = rows.filter((r) => r.sp === sp);
      SEGS.forEach((S, s) => { const sh = g.filter((r) => r.s === s).length / g.length; signals.push({ t: `${["Dépenses faibles", "Dépenses moyennes", "Dépenses élevées"][sp]} → ${S}`, sh, n: g.length, s }); });
    });
    $("signals").innerHTML = signals.filter((x) => x.n >= 200).sort((a, b) => b.sh - a.sh).slice(0, 5).map((x) => `
      <li><span class="dot" style="background:${COL[x.s]}"></span><span class="sig-t">${x.t}</span><span class="num sig-v">${pct(x.sh, 0)}</span><span class="muted xsmall num">n=${fmt.format(x.n)}</span></li>`).join("");
  }

  /* ---------- Init ---------- */
  refreshExplore();
  runSim();
  renderGrid();
  renderBatch();
  renderReco();
})();
