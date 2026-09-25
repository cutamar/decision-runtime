const token = document.querySelector('meta[name="decision-token"]').content;
const $ = id => document.getElementById(id);
const modelNames = {frozen: 'MiniLM encoder', adapted: 'Adapted MiniLM', sparse: 'TF IDF baseline'};
const phaseNames = {preparing: 'Prepare data', source: 'Get MiniLM', adapting: 'Adapt encoder', fitting: 'Fit and calibrate', evaluating: 'Check holdout'};
let activeRun = null;
let latestTiming = null;
let presetCatalog = {};
let labels = [];
let clockTimer = null;

async function readUtf8(file) {
  if (file.size > 100 * 1024 * 1024) throw new Error('File exceeds 100 MiB');
  return new TextDecoder('utf-8', {fatal: true}).decode(await file.arrayBuffer());
}
function setStatus(id, message, error = false) {
  const node = $(id);
  node.textContent = message;
  node.classList.toggle('error', error);
}
async function api(path, payload) {
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Decision-Web': token}, body: JSON.stringify(payload)});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}
async function poll(jobId, onUpdate) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`, {cache: 'no-store'});
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Could not read job');
    onUpdate(job);
    if (job.status === 'complete') return job.result;
    if (job.status === 'failed') throw new Error(job.message);
    await new Promise(resolve => setTimeout(resolve, 1200));
  }
}
function fmt(value, percent = false) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return 'Unavailable';
  return percent ? `${(Number(value) * 100).toFixed(1)}%` : String(value);
}
function metric(label, value, help) {
  const box = document.createElement('div');
  box.className = 'metric';
  const number = document.createElement('strong'); number.textContent = value;
  const name = document.createElement('span'); name.textContent = label;
  box.append(number, name);
  if (help) { const description = document.createElement('small'); description.textContent = help; box.append(description); }
  return box;
}
function detail(target, pairs) {
  target.replaceChildren();
  for (const [key, value] of pairs) {
    const line = document.createElement('div');
    const name = document.createElement('strong'); name.textContent = `${key}: `;
    const content = document.createElement('span'); content.textContent = String(value); line.append(name, content);
    target.append(line);
  }
}
function renderLabels(items) {
  labels = items;
  const chips = items.map(label => { const chip = document.createElement('span'); chip.className = 'chip'; chip.textContent = label; return chip; });
  $('label-list').replaceChildren(...chips);
  if (!chips.length) { const empty = document.createElement('span'); empty.className = 'empty-chip'; empty.textContent = 'Choose a labeled file to discover labels'; $('label-list').append(empty); }
}
function selectSource() {
  const source = $('data-source').value;
  const custom = source === 'upload';
  $('upload-fields').classList.toggle('hidden', !custom);
  $('data-notes').classList.toggle('hidden', !custom);
  $('preset-note').classList.toggle('hidden', custom);
  if (custom) {
    renderLabels([]);
    setStatus('setup-status', 'Choose a CSV or JSONL file to begin.');
  } else {
    const preset = presetCatalog[source];
    renderLabels(preset?.labels || []);
    $('preset-note').textContent = preset ? `${preset.name.replaceAll('Open-Jev', 'Open Jev')} uses pinned public synthetic data. Scores here describe a fixed label task and are separate from published model results.` : 'Loading public dataset details…';
    setStatus('setup-status', preset ? `Ready to prepare ${preset.name.replaceAll('Open-Jev', 'Open Jev')}.` : 'Loading public dataset details…');
  }
}
function showProgress(model) {
  const phases = model === 'sparse' ? ['preparing', 'fitting', 'evaluating'] : model === 'adapted' ? ['preparing', 'source', 'adapting', 'fitting', 'evaluating'] : ['preparing', 'source', 'fitting', 'evaluating'];
  $('progress-steps').replaceChildren(...phases.map(phase => { const item = document.createElement('li'); item.dataset.phase = phase; item.textContent = phaseNames[phase]; return item; }));
  $('training-progress').classList.remove('hidden');
  const started = Date.now();
  const updateClock = () => { const seconds = Math.floor((Date.now() - started) / 1000); $('progress-clock').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`; };
  updateClock(); clockTimer = setInterval(updateClock, 1000);
}
function updateProgress(job) {
  $('progress-message').textContent = job.message;
  const steps = [...$('progress-steps').children];
  const index = steps.findIndex(item => item.dataset.phase === job.phase);
  steps.forEach((item, position) => { item.classList.toggle('done', index > position || job.status === 'complete'); item.classList.toggle('active', index === position && job.status !== 'complete'); });
  if (job.status === 'failed') $('training-progress').classList.add('failed');
}
function stopProgress() { clearInterval(clockTimer); clockTimer = null; $('training-progress').classList.add('hidden'); $('training-progress').classList.remove('failed'); }
function renderEvaluation(target, report) {
  target.replaceChildren(
    metric('Accuracy', fmt(report.supported_accuracy, true), 'Share of labeled rows given the correct top label.'),
    metric('Macro F1', fmt(report.macro_f1, true), 'Average F1 across labels; each label gets equal weight.'),
    metric('Coverage', fmt(report.coverage, true), `${fmt(report.accepted)} of ${fmt(report.rows)} rows would receive a suggestion; the rest defer for review.`),
    metric('Error after acceptance', fmt(report.accepted_error_rate, true), 'Wrong suggestions divided by all accepted suggestions. Unavailable means the model deferred every row.')
  );
}
function renderRun(result) {
  activeRun = result.run_id;
  $('workflow-status').textContent = 'Model ready';
  const report = result.holdout;
  if (!report) throw new Error('This saved run predates the current app. Fit a new model to see its holdout results.');
  $('model-name').textContent = modelNames[result.model] || result.model;
  $('model-source').textContent = result.benchmark?.name.replaceAll('Open-Jev', 'Open Jev') || 'Your labeled data';
  $('result-caveat').textContent = result.preset ? 'These synthetic, publicly labeled examples are useful for checking the workflow; they do not predict performance on your company’s messages.' : 'These numbers describe one split of your labeled file. Check label quality, representative sampling and group overlap before relying on them.';
  $('row-count').textContent = `${fmt(report.rows)} holdout rows`;
  $('results-context').textContent = `This exact model was checked on ${fmt(report.rows)} labeled rows kept apart from fitting and threshold selection.`;
  renderEvaluation($('holdout-metrics'), report);
  $('secondary-metrics').replaceChildren(
    metric('Calibration error', fmt(report.ece_10_bins, true), 'Mean gap between predicted confidence and observed accuracy across 10 confidence bins; lower is better.'),
    metric('Log loss', report.negative_log_likelihood?.toFixed(3) ?? 'Unavailable', 'Penalizes confident wrong predictions; lower is better.'),
    metric('Brier score', report.multiclass_brier_mean_sum?.toFixed(3) ?? 'Unavailable', 'Mean squared probability error across labels; lower is better.')
  );
  const split = result.partition_rows || {};
  const details = [
    ['Accepted source rows', fmt(result.import?.accepted_rows)],
    ['Rejected source rows', fmt(result.import?.rejected_rows)],
    ['Training / development / calibration / policy / holdout', ['train', 'development', 'calibration', 'policy', 'test'].map(key => fmt(split[key])).join(' / ')],
    ['Labels', (report.labels || []).join(', ')],
    ['Accepted / deferred on holdout', `${fmt(report.accepted)} / ${fmt(report.rows - report.accepted)}`],
    ['Model bundle', result.bundle],
    ['Timing workload', result.benchmark_texts]
  ];
  if (result.benchmark) details.push(['Public source revision', result.benchmark.source_revision]);
  detail($('run-details'), details);
  $('holdout-report').textContent = JSON.stringify(report, null, 2);
  $('development-report').textContent = JSON.stringify({development: result.development, policy_selection: result.policy_selection, qualification_status: result.qualification_status}, null, 2);
  $('shift-panel').classList.toggle('hidden', !result.preset);
  $('shift-metrics').replaceChildren(); $('shift-report-wrap').classList.add('hidden'); setStatus('shift-status', '');
  $('prediction').classList.add('hidden'); setStatus('try-status', '');
  $('results').classList.remove('hidden'); $('try').classList.remove('hidden'); $('benchmark-current').disabled = false;
}

$('data-source').addEventListener('change', selectSource);
$('dataset-file').addEventListener('change', async () => {
  const file = $('dataset-file').files[0];
  $('file-name').textContent = file?.name || 'CSV or JSONL · up to 100 MiB';
  renderLabels([]);
  if (!file) return;
  try {
    setStatus('setup-status', 'Reading labels from your file…');
    const result = await api('/api/preview', {filename: file.name, content: await readUtf8(file)});
    renderLabels(result.labels);
    setStatus('setup-status', `Found ${result.labels.length} labels. Ready to fit.`);
  } catch (error) { setStatus('setup-status', error.message, true); }
});
$('start').addEventListener('click', async () => {
  const source = $('data-source').value;
  const file = $('dataset-file').files[0];
  if (source === 'upload' && (!file || !labels.length)) { setStatus('setup-status', 'Choose a labeled file and wait for its labels to appear.', true); return; }
  const model = document.querySelector('input[name="model"]:checked').value;
  $('start').disabled = true; $('results').classList.add('hidden'); $('try').classList.add('hidden'); $('benchmark-current').disabled = true; activeRun = null;
  showProgress(model);
  $('workflow-status').textContent = 'Fitting model';
  try {
    const request = {model, min_coverage: $('coverage').value, max_accepted_error: $('error').value, confidence: $('confidence').value};
    if (source === 'upload') Object.assign(request, {filename: file.name, content: await readUtf8(file), labels, provenance: $('provenance').value, license_notice: $('license-notice').value});
    else request.preset = source;
    setStatus('setup-status', 'Starting your run…');
    const started = await api('/api/runs', request);
    const result = await poll(started.job_id, job => { updateProgress(job); setStatus('setup-status', job.message, job.status === 'failed'); });
    renderRun(result);
    setStatus('setup-status', 'Model ready. Holdout results are below.');
    $('results').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) { $('workflow-status').textContent = 'Ready to begin'; setStatus('setup-status', error.message, true); }
  finally { stopProgress(); $('start').disabled = false; }
});
$('evaluate-ood').addEventListener('click', async () => {
  if (!activeRun) return;
  $('evaluate-ood').disabled = true; $('evaluate-ood').classList.add('busy');
  try {
    setStatus('shift-status', 'Checking the current model on shifted examples…');
    const started = await api(`/api/runs/${activeRun}/evaluate`, {split: 'ood'});
    const report = await poll(started.job_id, job => setStatus('shift-status', job.message));
    renderEvaluation($('shift-metrics'), report);
    $('shift-report').textContent = JSON.stringify(report, null, 2);
    $('shift-report-wrap').classList.remove('hidden');
    setStatus('shift-status', `Checked ${fmt(report.rows)} shifted rows with this model.`);
  } catch (error) { setStatus('shift-status', error.message, true); }
  finally { $('evaluate-ood').disabled = false; $('evaluate-ood').classList.remove('busy'); }
});
const abstentionReasons = {below_threshold: 'Its confidence did not meet the chosen threshold.', ambiguous_top_choice: 'The top choices were too close to call.', label_review_only: 'This label is configured for review.', unsupported_input: 'The input could not be scored reliably.', input_too_long: 'The message is longer than this model supports.'};
$('try-button').addEventListener('click', async () => {
  const text = $('trial-text').value.trim();
  if (!activeRun) { setStatus('try-status', 'Fit a model first.', true); return; }
  if (!text) { setStatus('try-status', 'Write or paste a message first.', true); return; }
  $('try-button').disabled = true; $('try-button').classList.add('busy'); $('prediction').classList.add('hidden');
  try {
    setStatus('try-status', 'Scoring this message locally…');
    const result = await api(`/api/runs/${activeRun}/predict`, {text});
    $('prediction-label').textContent = result.suggested_choice || result.choice || 'No label';
    const accepted = result.status === 'would_accept';
    $('prediction-policy').textContent = accepted ? 'Would suggest' : 'Needs review';
    $('prediction-policy').classList.toggle('defer', !accepted);
    $('prediction-explanation').textContent = accepted ? 'The top label passes this model’s review threshold.' : (abstentionReasons[result.abstention_reason] || 'The model would defer this message for review. The top label is shown for inspection.');
    const rows = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]).map(([label, probability]) => {
      const row = document.createElement('div'); row.className = 'prob-row';
      const name = document.createElement('span'); name.className = 'prob-label'; name.textContent = label; const value = document.createElement('strong'); value.className = 'prob-value'; value.textContent = fmt(probability, true);
      const track = document.createElement('div'); track.className = 'prob-track'; const bar = document.createElement('span'); bar.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`; track.append(bar); row.append(name, track, value); return row;
    });
    $('probability-list').replaceChildren(...rows);
    $('prediction').classList.remove('hidden'); setStatus('try-status', 'Prediction ready.');
  } catch (error) { setStatus('try-status', error.message, true); }
  finally { $('try-button').disabled = false; $('try-button').classList.remove('busy'); }
});

function renderTiming(report) {
  latestTiming = report;
  $('timing-metrics').replaceChildren(
    metric('Cold load', `${report.cold_load_ms.toFixed(1)} ms`, 'Time to load the bundle before the first call.'),
    metric('Warm p50', `${report.warm_p50_ms.toFixed(1)} ms`, 'Half of timed calls were this fast or faster.'),
    metric('Warm p95', `${report.warm_p95_ms.toFixed(1)} ms`, '95% of timed calls were this fast or faster.'),
    metric('Sequential calls / s', report.sequential_calls_per_second.toFixed(1), 'Average calls per second, one at a time.')
  );
  detail($('timing-device'), [
    ['CPU', report.cpu_model || report.processor || 'Unknown'],
    ['OS / architecture', `${report.machine} / ${report.architecture}`],
    ['Peak RSS', `${report.peak_rss_mib.toFixed(1)} MiB`],
    ['Bundle / workload ID', `${report.manifest_sha256.slice(0, 12)} / ${report.workload_sha256.slice(0, 12)}`],
    ['Method', `${report.warmup_calls} warmups, ${report.timed_calls} timed calls, concurrency ${report.concurrency}`]
  ]);
  $('download-timing').classList.remove('hidden');
}
async function timeBundle(path, payload) {
  $('benchmark-current').disabled = true; $('benchmark-copied').disabled = true; $('benchmark-current').classList.add('busy');
  try {
    setStatus('timing-status', 'Loading bundle and measuring CPU calls…');
    const started = await api(path, payload);
    const report = await poll(started.job_id, job => setStatus('timing-status', job.message));
    renderTiming(report);
    setStatus('timing-status', 'Timing complete. Download the report to compare this device with another.');
  } catch (error) { setStatus('timing-status', error.message, true); }
  finally { $('benchmark-current').disabled = !activeRun; $('benchmark-copied').disabled = false; $('benchmark-current').classList.remove('busy'); }
}
$('benchmark-current').addEventListener('click', () => { if (activeRun) timeBundle(`/api/runs/${activeRun}/benchmark`, {}); });
$('workload-file').addEventListener('change', () => { $('workload-name').textContent = $('workload-file').files[0]?.name || 'Choose file'; });
$('benchmark-copied').addEventListener('click', async () => {
  const file = $('workload-file').files[0];
  if (!file) { setStatus('timing-status', 'Choose a representative texts JSON file.', true); return; }
  try { await timeBundle('/api/benchmark', {bundle_path: $('bundle-path').value, trusted_public_key_path: $('public-key-path').value, texts: JSON.parse(await readUtf8(file))}); }
  catch (error) { setStatus('timing-status', error.message, true); }
});
$('download-timing').addEventListener('click', () => {
  if (!latestTiming) return;
  const blob = new Blob([JSON.stringify(latestTiming, null, 2) + '\n'], {type: 'application/json'});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `decision-timing-${latestTiming.manifest_sha256.slice(0, 12)}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
$('reports-file').addEventListener('change', async () => {
  const files = [...$('reports-file').files]; $('reports-name').textContent = `${files.length} file(s) selected`;
  const target = $('device-comparison'); target.replaceChildren();
  try {
    if (files.length < 2) throw new Error('Choose at least two timing reports.');
    const reports = await Promise.all(files.map(async file => JSON.parse(await readUtf8(file))));
    if (reports.some(report => report.manifest_sha256 !== reports[0].manifest_sha256 || report.workload_sha256 !== reports[0].workload_sha256 || report.timed_calls !== reports[0].timed_calls)) throw new Error('Reports must use the same bundle, text workload and timed-call count.');
    const table = document.createElement('table'); const head = document.createElement('tr');
    for (const name of ['Device', 'Cold load', 'P50', 'P95', 'Calls/s', 'Peak MiB']) { const cell = document.createElement('th'); cell.textContent = name; head.append(cell); } table.append(head);
    for (const report of reports) {
      const row = document.createElement('tr');
      for (const value of [report.cpu_model || report.machine, `${report.cold_load_ms.toFixed(1)} ms`, `${report.warm_p50_ms.toFixed(1)} ms`, `${report.warm_p95_ms.toFixed(1)} ms`, report.sequential_calls_per_second.toFixed(1), report.peak_rss_mib.toFixed(1)]) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }
      table.append(row);
    }
    target.append(table);
  } catch (error) { target.textContent = error.message; }
});

Promise.all([fetch('/api/presets', {cache: 'no-store'}).then(response => response.json()), fetch('/api/runs/current', {cache: 'no-store'}).then(response => response.json())])
  .then(([catalog, current]) => { presetCatalog = catalog.presets || {}; selectSource(); if (current.run) { renderRun(current.run); setStatus('setup-status', 'Your latest model is ready below. Fit again to replace it.'); } })
  .catch(error => setStatus('setup-status', `Could not load app state: ${error.message}`, true));

const navLinks = [...document.querySelectorAll('.masthead nav a')];
const sections = navLinks.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
const sectionObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      navLinks.forEach(link => link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`));
    }
  }
}, {rootMargin: '-12% 0px -65% 0px'});
sections.forEach(section => sectionObserver.observe(section));
