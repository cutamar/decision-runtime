const token = document.querySelector('meta[name="decision-token"]').content;
const $ = id => document.getElementById(id);
let activeRun = null;
let latestTiming = null;
let presetCatalog = {};
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
  const response = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Decision-Web': token},
    body: JSON.stringify(payload)
  });
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
  if (value === null || value === undefined) return '—';
  return percent ? `${(value * 100).toFixed(1)}%` : String(value);
}
function metric(label, value) {
  const box = document.createElement('div');
  box.className = 'metric';
  const strong = document.createElement('strong');
  strong.textContent = value;
  const name = document.createElement('span');
  name.textContent = label;
  box.append(strong, name);
  return box;
}
function renderMetrics(target, report) {
  const items = [
    metric('Supported accuracy', fmt(report.classification?.accuracy ?? report.supported_accuracy, true)),
    metric('Macro F1', fmt(report.classification?.per_label?.['macro avg']?.['f1-score'] ?? report.macro_f1, true)),
    metric('Would accept / coverage', `${fmt(report.accepted)} / ${fmt(report.end_to_end_coverage ?? report.coverage, true)}`),
    metric('Accepted error', fmt(report.accepted_error_rate, true))
  ];
  if (report.ece_10_bins !== undefined) items.push(
    metric('Calibration error (10 bins)', fmt(report.ece_10_bins, true)),
    metric('Log loss', report.negative_log_likelihood?.toFixed(3) ?? '—'),
    metric('Brier score', report.multiclass_brier_mean_sum?.toFixed(3) ?? '—')
  );
  target.replaceChildren(...items);
}
function detail(target, pairs) {
  target.replaceChildren();
  for (const [key, value] of pairs) {
    const line = document.createElement('div');
    const name = document.createElement('strong');
    name.textContent = `${key}: `;
    line.append(name, document.createTextNode(value));
    target.append(line);
  }
}

fetch('/api/presets', {cache: 'no-store'}).then(response => response.json()).then(data => {
  presetCatalog = data.presets;
  if ($('data-source').value !== 'upload') selectSource();
}).catch(() => setStatus('status', 'Could not load public preset metadata.', true));
function selectSource() {
  const source = $('data-source').value;
  const preset = presetCatalog[source];
  $('upload-fields').classList.toggle('hidden', source !== 'upload');
  $('preset-note').classList.toggle('hidden', source === 'upload');
  $('labels').readOnly = source !== 'upload';
  if (preset) {
    $('labels').value = preset.labels.join(', ');
    $('provenance').value = `${preset.name}; pinned public synthetic dataset`;
    $('license-notice').value = preset.license;
    setStatus('status', `Ready to prepare ${preset.name}.`);
  } else if (source === 'upload') {
    $('labels').value = '';
    $('provenance').value = 'Operator-supplied exploratory routing examples';
    $('license-notice').value = 'Operator-supplied data; permission must be verified before customer use';
    setStatus('status', 'Choose a labeled file to begin.');
  }
}
$('data-source').addEventListener('change', selectSource);

$('dataset-file').addEventListener('change', async () => {
  const file = $('dataset-file').files[0];
  $('file-name').textContent = file?.name || 'No file selected';
  if (!file) return;
  try {
    setStatus('status', 'Reading labels…');
    const result = await api('/api/preview', {filename: file.name, content: await readUtf8(file)});
    $('labels').value = result.labels.join(', ');
    setStatus('status', `Found ${result.labels.length} labels. Ready to fit.`);
  } catch (error) { setStatus('status', error.message, true); }
});

$('start').addEventListener('click', async () => {
  const source = $('data-source').value;
  const file = $('dataset-file').files[0];
  if (source === 'upload' && !file) { setStatus('status', 'Choose a dataset first.', true); return; }
  $('start').disabled = true;
  $('results').classList.add('hidden');
  $('external').classList.add('hidden');
  $('published-evaluation').classList.add('hidden');
  $('benchmark-current').disabled = true;
  activeRun = null;
  try {
    setStatus('status', source === 'upload' ? 'Uploading and validating data…' : 'Preparing pinned public data…');
    const request = {
      model: $('model').value,
      min_coverage: $('coverage').value,
      max_accepted_error: $('error').value,
      confidence: $('confidence').value,
      provenance: $('provenance').value,
      license_notice: $('license-notice').value
    };
    if (source === 'upload') {
      request.filename = file.name;
      request.content = await readUtf8(file);
      request.labels = $('labels').value.split(',').map(item => item.trim()).filter(Boolean);
    } else {
      request.preset = source;
    }
    const started = await api('/api/runs', request);
    const result = await poll(started.job_id, job => setStatus('status', job.message));
    activeRun = result.run_id;
    const report = result.development;
    renderMetrics($('metrics'), report);
    const datasetDetails = [
      ['Accepted rows', fmt(result.import.accepted_rows)],
      ['Rejected rows', fmt(result.import.rejected_rows)],
      ['Split', Object.entries(result.partition_rows).map(([name, count]) => `${name}: ${count}`).join(' · ')],
      ['Model', result.model],
      ['Bundle', result.bundle],
      ['Timing workload', result.benchmark_texts]
    ];
    if (result.benchmark) datasetDetails.push(
      ['Source revision', result.benchmark.source_revision],
      ['Excluded non-hard targets', fmt(Object.values(result.benchmark.excluded_soft_or_unlabeled).reduce((sum, count) => sum + count, 0))]
    );
    detail($('dataset-summary'), datasetDetails);
    detail($('decision-summary'), [
      ['Would accept', fmt(report.accepted)],
      ['Would defer', fmt(report.deferred)],
      ['Accepted errors', fmt(report.accepted_errors)],
      ['Qualification', result.qualification_status]
    ]);
    $('report').textContent = JSON.stringify(report, null, 2);
    $('results').classList.remove('hidden');
    $('external').classList.remove('hidden');
    $('benchmark-current').disabled = false;
    if (result.preset) $('published-evaluation').classList.remove('hidden');
    setStatus('status', 'Candidate ready. The final test remains untouched.');
  } catch (error) { setStatus('status', error.message, true); }
  finally { $('start').disabled = false; }
});

async function evaluatePublished(split) {
  if (!activeRun) return;
  $('evaluate-test').disabled = true;
  $('evaluate-ood').disabled = true;
  try {
    setStatus('published-status', `Evaluating published ${split.toUpperCase()} subset…`);
    const started = await api(`/api/runs/${activeRun}/evaluate`, {published_split: split});
    const report = await poll(started.job_id, job => setStatus('published-status', job.message));
    renderMetrics($('published-metrics'), report);
    $('published-report').textContent = JSON.stringify(report, null, 2);
    $('published-detail').classList.remove('hidden');
    setStatus('published-status', `${split.toUpperCase()}: ${report.rows} rows. Derived-task diagnostic only.`);
  } catch (error) { setStatus('published-status', error.message, true); }
  finally { $('evaluate-test').disabled = false; $('evaluate-ood').disabled = false; }
}
$('evaluate-test').addEventListener('click', () => evaluatePublished('test'));
$('evaluate-ood').addEventListener('click', () => evaluatePublished('ood'));

$('evaluation-file').addEventListener('change', () => {
  $('evaluation-name').textContent = $('evaluation-file').files[0]?.name || 'No file selected';
});
$('evaluate').addEventListener('click', async () => {
  const file = $('evaluation-file').files[0];
  if (!activeRun || !file) { setStatus('evaluation-status', 'Choose a separate labeled file.', true); return; }
  $('evaluate').disabled = true;
  try {
    setStatus('evaluation-status', 'Evaluating labeled data…');
    const started = await api(`/api/runs/${activeRun}/evaluate`, {filename: file.name, content: await readUtf8(file)});
    const report = await poll(started.job_id, job => setStatus('evaluation-status', job.message));
    renderMetrics($('external-metrics'), report);
    $('external-report').textContent = JSON.stringify(report, null, 2);
    $('external-detail').classList.remove('hidden');
    setStatus('evaluation-status', `Evaluated ${report.rows} rows. Descriptive results only.`);
  } catch (error) { setStatus('evaluation-status', error.message, true); }
  finally { $('evaluate').disabled = false; }
});

function renderTiming(report) {
  latestTiming = report;
  $('timing-metrics').replaceChildren(
    metric('Cold load', `${report.cold_load_ms.toFixed(1)} ms`),
    metric('Warm p50', `${report.warm_p50_ms.toFixed(1)} ms`),
    metric('Warm p95', `${report.warm_p95_ms.toFixed(1)} ms`),
    metric('Sequential calls / s', report.sequential_calls_per_second.toFixed(1))
  );
  detail($('timing-device'), [
    ['CPU', report.cpu_model || report.processor || 'Unknown'],
    ['OS / architecture', `${report.machine} / ${report.architecture}`],
    ['Peak RSS', `${report.peak_rss_mib.toFixed(1)} MiB`],
    ['Bundle and workload', `${report.manifest_sha256.slice(0, 12)} / ${report.workload_sha256.slice(0, 12)}`],
    ['Method', `${report.warmup_calls} warmups, ${report.timed_calls} timed calls, concurrency ${report.concurrency}`]
  ]);
  $('download-timing').classList.remove('hidden');
}
async function timeBundle(path, payload) {
  $('benchmark-current').disabled = true;
  $('benchmark-copied').disabled = true;
  try {
    setStatus('timing-status', 'Loading bundle and running repeated CPU calls…');
    const started = await api(path, payload);
    const report = await poll(started.job_id, job => setStatus('timing-status', job.message));
    renderTiming(report);
    setStatus('timing-status', 'Timing complete on this device. Download the report to compare with another device.');
  } catch (error) { setStatus('timing-status', error.message, true); }
  finally { $('benchmark-current').disabled = !activeRun; $('benchmark-copied').disabled = false; }
}
$('benchmark-current').addEventListener('click', () => {
  if (activeRun) timeBundle(`/api/runs/${activeRun}/benchmark`, {});
});
$('workload-file').addEventListener('change', () => {
  $('workload-name').textContent = $('workload-file').files[0]?.name || 'No file selected';
});
$('benchmark-copied').addEventListener('click', async () => {
  const file = $('workload-file').files[0];
  if (!file) { setStatus('timing-status', 'Choose a representative texts JSON file.', true); return; }
  try {
    const texts = JSON.parse(await readUtf8(file));
    await timeBundle('/api/benchmark', {bundle_path: $('bundle-path').value,
      trusted_public_key_path: $('public-key-path').value, texts});
  } catch (error) { setStatus('timing-status', error.message, true); }
});
$('download-timing').addEventListener('click', () => {
  if (!latestTiming) return;
  const blob = new Blob([JSON.stringify(latestTiming, null, 2) + '\n'], {type: 'application/json'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `decision-timing-${latestTiming.manifest_sha256.slice(0, 12)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
$('reports-file').addEventListener('change', async () => {
  const files = [...$('reports-file').files];
  $('reports-name').textContent = `${files.length} file(s) selected`;
  const target = $('device-comparison');
  target.replaceChildren();
  try {
    if (files.length < 2) throw new Error('Choose at least two timing reports.');
    const reports = await Promise.all(files.map(async file => JSON.parse(await readUtf8(file))));
    if (reports.some(report => report.manifest_sha256 !== reports[0].manifest_sha256 ||
        report.workload_sha256 !== reports[0].workload_sha256 || report.timed_calls !== reports[0].timed_calls)) {
      throw new Error('Reports must use the same bundle, text workload and timed-call count.');
    }
    const table = document.createElement('table');
    const head = document.createElement('tr');
    for (const name of ['Device', 'Cold load', 'P50', 'P95', 'Calls/s', 'Peak MiB']) {
      const cell = document.createElement('th'); cell.textContent = name; head.append(cell);
    }
    table.append(head);
    for (const report of reports) {
      const row = document.createElement('tr');
      for (const value of [report.cpu_model || report.machine, `${report.cold_load_ms.toFixed(1)} ms`,
          `${report.warm_p50_ms.toFixed(1)} ms`, `${report.warm_p95_ms.toFixed(1)} ms`,
          report.sequential_calls_per_second.toFixed(1), report.peak_rss_mib.toFixed(1)]) {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }
      table.append(row);
    }
    target.append(table);
  } catch (error) { target.textContent = error.message; }
});
