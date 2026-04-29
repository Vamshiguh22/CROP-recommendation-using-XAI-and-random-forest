/* ═══════════════════════════════════════════════════════════════
   main.js — CropAI Frontend Logic
   Handles: form sync, API calls, Chart.js rendering, XAI display
   ═══════════════════════════════════════════════════════════════ */

"use strict";

// ── State ──────────────────────────────────────────────────────────────────
let shapChartInstance = null;
let top5ChartInstance = null;
let modelMeta         = null;

// ── Slider ↔ Input Sync ────────────────────────────────────────────────────
function syncInput(field, val) {
  const num = parseFloat(val);
  document.getElementById(`input-${field}`).value = val;
  document.getElementById(`display-${field}`).textContent =
    Number.isInteger(num) ? num : num.toFixed(1);
  updateSliderGradient(field, val);
}

function syncSlider(field, val) {
  document.getElementById(`slider-${field}`).value = val;
  document.getElementById(`display-${field}`).textContent =
    val !== '' ? (Number.isInteger(parseFloat(val)) ? parseFloat(val) : parseFloat(val).toFixed(1)) : '—';
  updateSliderGradient(field, val);
}

function updateSliderGradient(field, val) {
  const slider = document.getElementById(`slider-${field}`);
  const pct    = ((val - slider.min) / (slider.max - slider.min)) * 100;
  slider.style.setProperty('--pct', pct.toFixed(1) + '%');
  slider.style.background =
    `linear-gradient(90deg, var(--accent) ${pct.toFixed(1)}%, rgba(255,255,255,0.1) 0%)`;
}

// ── Reset Form ─────────────────────────────────────────────────────────────
function resetForm() {
  const defaults = { N: 90, P: 42, K: 43, temperature: 21, humidity: 82, ph: 6.5, rainfall: 203 };
  Object.entries(defaults).forEach(([f, v]) => {
    document.getElementById(`input-${f}`).value  = v;
    document.getElementById(`slider-${f}`).value = v;
    syncInput(f, v);
  });
  document.getElementById('results-section').style.display = 'none';
  hideToasts();
}

// ── Collect Inputs ─────────────────────────────────────────────────────────
function getInputValues() {
  const fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'];
  const data   = {};
  for (const f of fields) {
    const raw = document.getElementById(`input-${f}`).value;
    if (raw === '' || raw === null) throw new Error(`Please fill in all fields (missing: ${f})`);
    data[f] = parseFloat(raw);
    if (isNaN(data[f]))           throw new Error(`Invalid value for ${f}`);
  }
  return data;
}

// ── Toast ──────────────────────────────────────────────────────────────────
function showToast(type, msg) {
  const el = document.getElementById(type === 'error' ? 'errorToast' : 'successToast');
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.style.display = 'none'; }, 4000);
}
function hideToasts() {
  ['errorToast', 'successToast'].forEach(id => {
    document.getElementById(id).style.display = 'none';
  });
}

// ── Loader ─────────────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('loader').style.display   = on ? 'flex' : 'none';
  document.getElementById('btnPredict').disabled    = on;
}

// ── Main Prediction Flow ───────────────────────────────────────────────────
async function runPrediction() {
  hideToasts();
  let data;
  try { data = getInputValues(); }
  catch (e) { showToast('error', e.message); return; }

  setLoading(true);

  try {
    // Call /predict and /explain in parallel
    const [predRes, explainRes] = await Promise.all([
      fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }),
      fetch('/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }),
    ]);

    const predJson    = await predRes.json();
    const explainJson = await explainRes.json();

    if (predJson.status !== 'ok')    throw new Error(predJson.message || 'Prediction failed');
    if (explainJson.status !== 'ok') throw new Error(explainJson.message || 'Explanation failed');

    setLoading(false);
    renderResults(predJson, explainJson);
    showToast('success', `✅ Recommended: ${predJson.crop.toUpperCase()}`);

  } catch (err) {
    setLoading(false);
    showToast('error', '❌ ' + err.message);
  }
}

// ── Render All Results ─────────────────────────────────────────────────────
function renderResults(pred, explain) {
  const section = document.getElementById('results-section');
  section.style.display = 'block';
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Crop card
  document.getElementById('resultEmoji').textContent = pred.emoji || '🌱';
  document.getElementById('resultCrop').textContent  = pred.crop;
  const confPct = pred.confidence;
  document.getElementById('confBar').style.width     = confPct + '%';
  document.getElementById('confText').textContent    = `${confPct}% confidence (Random Forest)`;

  // Model agreement banner
  const preds = pred.model_predictions;
  const crops = [preds.random_forest.crop, preds.logistic_regression.crop, preds.naive_bayes.crop];
  const allAgree = crops.every(c => c === crops[0]);
  const banner   = document.getElementById('agreementBanner');
  if (allAgree) {
    banner.textContent = '✅ All 3 models agree — high confidence in this recommendation!';
    banner.className   = 'agreement-banner agreement-full';
  } else {
    banner.textContent = `⚠️ Models partially agree: RF→${preds.random_forest.crop}, LR→${preds.logistic_regression.crop}, NB→${preds.naive_bayes.crop}`;
    banner.className   = 'agreement-banner agreement-part';
  }

  // SHAP Chart
  renderShapChart(explain);

  // Top-5 Chart
  renderTop5Chart(pred.top5_crops);

  // XAI Text
  renderXaiText(explain);

  // Model predictions
  renderModelPreds(pred.model_predictions);
}

// ── SHAP Chart ─────────────────────────────────────────────────────────────
function renderShapChart(explain) {
  const contribs = explain.feature_contributions;
  const labels   = contribs.map(c => c.label);
  const values   = contribs.map(c => c.shap_value);
  const colors   = values.map(v =>
    v > 0 ? 'rgba(34,197,94,0.75)' : 'rgba(239,68,68,0.7)'
  );
  const borders = values.map(v =>
    v > 0 ? '#22c55e' : '#ef4444'
  );

  if (shapChartInstance) shapChartInstance.destroy();
  const ctx = document.getElementById('shapChart').getContext('2d');
  shapChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'SHAP Value',
        data: values,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1.5,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `SHAP: ${ctx.raw.toFixed(4)} (${ctx.raw > 0 ? 'supports' : 'opposes'})`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#7a9a7e' },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#e8f5e9', font: { weight: '600' } },
        },
      },
    },
  });
}

// ── Top-5 Crops Chart ──────────────────────────────────────────────────────
function renderTop5Chart(top5) {
  const labels = Object.keys(top5);
  const data   = Object.values(top5);
  const colors = data.map((_, i) =>
    i === 0 ? 'rgba(34,197,94,0.8)' : 'rgba(255,255,255,0.1)'
  );

  if (top5ChartInstance) top5ChartInstance.destroy();
  const ctx = document.getElementById('top5Chart').getContext('2d');
  top5ChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [{
        data,
        backgroundColor: [
          'rgba(34,197,94,0.85)', 'rgba(59,130,246,0.7)',
          'rgba(245,158,11,0.7)', 'rgba(168,85,247,0.7)',
          'rgba(239,68,68,0.6)',
        ],
        borderColor: 'rgba(255,255,255,0.05)',
        borderWidth: 2,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#7a9a7e', padding: 14, font: { size: 11 } },
        },
        tooltip: {
          callbacks: { label: (c) => ` ${c.label}: ${c.raw}%` },
        },
      },
    },
  });
}

// ── XAI Text Explanation ───────────────────────────────────────────────────
function renderXaiText(explain) {
  const container  = document.getElementById('xaiBody');
  const contribs   = explain.feature_contributions;
  const sentences  = explain.text_explanation;
  container.innerHTML = '';

  sentences.forEach((html, idx) => {
    const c   = contribs[idx];
    const div = document.createElement('div');
    div.className = `xai-row ${c.level === 'Medium' ? 'medium' : c.level === 'Low' ? 'low' : ''}`;
    div.style.animationDelay = `${idx * 80}ms`;
    div.innerHTML = html;
    container.appendChild(div);
  });
}

// ── Model Predictions ──────────────────────────────────────────────────────
function renderModelPreds(preds) {
  const grid  = document.getElementById('modelPredGrid');
  const names = {
    random_forest: 'Random Forest',
    logistic_regression: 'Logistic Regression',
    naive_bayes: 'Naive Bayes',
  };
  grid.innerHTML = '';
  Object.entries(preds).forEach(([key, val], i) => {
    const card = document.createElement('div');
    card.className = `model-pred-card${key === 'random_forest' ? ' best' : ''}`;
    card.style.animationDelay = `${i * 100}ms`;
    card.innerHTML = `
      <div class="pred-model-name">${names[key]}${key === 'random_forest' ? '<span class="pred-best-badge">PRIMARY</span>' : ''}</div>
      <div class="pred-crop-name">${val.crop}</div>
      <div class="pred-conf">${val.confidence}% confidence</div>
    `;
    grid.appendChild(card);
  });
}

// ── Load Model Info on Page Load ───────────────────────────────────────────
async function loadModelInfo() {
  try {
    const res  = await fetch('/model-info');
    const json = await res.json();
    if (json.status !== 'ok') return;
    modelMeta = json.metadata;
    renderAccuracyCards(modelMeta.accuracies);
    renderHeroStats(modelMeta);
    renderConfusionMatrix(modelMeta);
  } catch (_) {
    /* server not ready yet — silently ignore */
  }
}

function renderHeroStats(meta) {
  const el = document.getElementById('heroStats');
  if (!meta) return;
  el.innerHTML = `
    <div class="stat-pill">📊 ${meta.class_names.length} Crop Types</div>
    <div class="stat-pill">🧪 ${meta.total_records.toLocaleString()} Records</div>
    <div class="stat-pill">✅ ${meta.accuracies.random_forest}% RF Accuracy</div>
  `;
}

function renderAccuracyCards(accs) {
  const grid = document.getElementById('accGrid');
  const cards = [
    { name: 'Random Forest',       key: 'random_forest',       val: accs.random_forest,       best: true  },
    { name: 'Logistic Regression', key: 'logistic_regression', val: accs.logistic_regression, best: false },
    { name: 'Naive Bayes',         key: 'naive_bayes',         val: accs.naive_bayes,         best: false },
  ];
  grid.innerHTML = '';
  cards.forEach(c => {
    const isAmber = c.val < 95;
    const div = document.createElement('div');
    div.className = `acc-card${c.best ? ' best' : ''}`;
    div.innerHTML = `
      <div class="acc-model">${c.name}${c.best ? ' ⭐' : ''}</div>
      <div class="acc-value${isAmber ? ' amber' : ''}">${c.val}%</div>
      <div class="acc-bar-wrap"><div class="acc-bar${isAmber ? ' amber' : ''}" style="width:0%" data-w="${c.val}"></div></div>
      <div class="acc-note">Test set accuracy</div>
    `;
    grid.appendChild(div);
    // Animate bar
    setTimeout(() => {
      div.querySelector('.acc-bar').style.width = c.val + '%';
    }, 200);
  });
}

function renderConfusionMatrix(meta) {
  const container = document.getElementById('cmContainer');
  const cm        = meta.confusion_matrices.random_forest;
  const labels    = meta.class_names;
  if (!cm || !labels) { container.innerHTML = '<div class="cm-loading">Not available.</div>'; return; }

  // Find max for colour scaling
  const flat = cm.flat();
  const max  = Math.max(...flat);

  let html = '<table class="cm-table"><thead><tr><th></th>';
  labels.forEach(l => { html += `<th style="writing-mode:vertical-rl;transform:rotate(180deg);max-height:80px">${l}</th>`; });
  html += '</tr></thead><tbody>';
  cm.forEach((row, ri) => {
    html += `<tr><th style="text-align:right;padding-right:6px;color:#7a9a7e">${labels[ri]}</th>`;
    row.forEach((v, ci) => {
      const alpha = max > 0 ? (v / max) * 0.85 : 0;
      const bg    = ri === ci ? `rgba(34,197,94,${alpha})` : v > 0 ? `rgba(239,68,68,${alpha * 0.6})` : 'transparent';
      html += `<td style="background:${bg};color:${v>0?'#e8f5e9':'#3a4a3e'}">${v}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

// ── Background Particles ───────────────────────────────────────────────────
function spawnParticles() {
  const container = document.getElementById('bgParticles');
  for (let i = 0; i < 18; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size  = Math.random() * 6 + 2;
    const left  = Math.random() * 100;
    const delay = Math.random() * 12;
    const dur   = Math.random() * 14 + 8;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${left}%; bottom:-10px;
      animation-delay:${delay}s;
      animation-duration:${dur}s;
      opacity:${Math.random() * 0.3 + 0.1};
    `;
    container.appendChild(p);
  }
}

// ── Init sliders on load ───────────────────────────────────────────────────
function initSliders() {
  const defaults = { N: 90, P: 42, K: 43, temperature: 21, humidity: 82, ph: 6.5, rainfall: 203 };
  Object.entries(defaults).forEach(([f, v]) => updateSliderGradient(f, v));
}

// ── Bootstrap ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  spawnParticles();
  initSliders();
  loadModelInfo();
});
