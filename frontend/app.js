const banner = document.getElementById('banner');
const summary = document.getElementById('summary');
const successPanel = document.getElementById('success-panel');
const errorPanel = document.getElementById('error-panel');
const metaList = document.getElementById('meta-list');
const form = document.getElementById('process-form');
const submitButton = document.getElementById('submit-btn');
const confidenceRange = document.getElementById('confidence_threshold');
const confidenceInput = document.getElementById('confidence_threshold_input');
const confidenceDisplay = document.getElementById('confidence_display');
const progressPanel = document.getElementById('progress-panel');
const progressBar = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const progressMessage = document.getElementById('progress-message');
const progressCount = document.getElementById('progress-count');

const apiBase = "/api/v1";
let activeJobId = null;
let activePollTimer = null;
let serverMaxLabels = null;

function showBanner(message, kind = 'error') {
  banner.textContent = message;
  banner.className = `banner ${kind}`;
}

function hideBanner() {
  banner.textContent = '';
  banner.className = 'banner hidden';
}

function setConfidenceValue(value) {
  const normalized = Math.min(1, Math.max(0, Number(value) || 0));
  const displayValue = normalized.toFixed(2);
  confidenceRange.value = displayValue;
  confidenceInput.value = displayValue;
  confidenceDisplay.value = displayValue;
  confidenceDisplay.textContent = displayValue;
}

function updateProgress(job) {
  const progress = Math.min(100, Math.max(0, Number(job.progress_percentage) || 0));
  progressPanel.className = 'panel progress-panel';
  progressBar.style.width = `${progress}%`;
  progressLabel.textContent = `${progress}%`;
  progressMessage.textContent = job.message || '处理中';
  progressCount.textContent = `${job.processed || 0} / ${job.total || 0}`;
}

function resetProgress() {
  progressPanel.className = 'panel progress-panel hidden';
  progressBar.style.width = '0%';
  progressLabel.textContent = '0%';
  progressMessage.textContent = '等待开始';
  progressCount.textContent = '0 / 0';
}

function stopPolling() {
  if (activePollTimer) {
    clearTimeout(activePollTimer);
    activePollTimer = null;
  }
  activeJobId = null;
}

function renderMeta(info) {
  metaList.innerHTML = `
    <div>
      <p class="field-label">元素提取模型</p>
      <p>${info.model}</p>
    </div>
    <div>
      <p class="field-label">候选校验模型</p>
      <p>${info.validation_model}</p>
    </div>
    <div>
      <p class="field-label">支持格式</p>
      <p>${info.allowed_extensions.join(', ')}</p>
    </div>
    <div>
      <p class="field-label">命名规则</p>
      <p>优先按 aa_bb_cc.ext 输出；未识别到有效元素时保留原文件名。</p>
    </div>
    <div>
      <p class="field-label">运行设备</p>
      <p>${info.device || (info.cuda_available ? 'cuda' : 'cpu')}</p>
    </div>
  `;

  setConfidenceValue(info.confidence_threshold);
  const maxLabelsInput = document.getElementById('max_labels');
  if (maxLabelsInput) {
    maxLabelsInput.max = info.max_labels;
    maxLabelsInput.value = info.max_labels;
  }
  serverMaxLabels = info.max_labels;
}

function renderSummary(result) {
  summary.className = 'summary-grid';
  summary.innerHTML = `
    <div class="summary-card">
      <span>扫描图片</span>
      <strong>${result.total}</strong>
    </div>
    <div class="summary-card success">
      <span>已输出文件</span>
      <strong>${result.successful}</strong>
    </div>
    <div class="summary-card danger">
      <span>处理失败</span>
      <strong>${result.failed}</strong>
    </div>
  `;
}

function renderSuccess(result) {
  if (!result.results.length) {
    successPanel.className = 'panel hidden';
    successPanel.innerHTML = '';
    return;
  }

  successPanel.className = 'panel result-list';
  successPanel.innerHTML = `
    <div class="panel-head">
      <h3>输出结果</h3>
      <span>${result.results.length} 张</span>
    </div>
    ${result.results.map((item) => `
      <article class="result-row">
        <div class="result-main">
          <div>
            <p class="field-label">原文件名</p>
            <p class="file-name">${item.original_filename}</p>
          </div>
          <div>
            <p class="field-label">${item.status === 'kept_original_name' ? '输出文件名' : '重命名结果'}</p>
            <p class="file-name highlight">${item.renamed_filename}</p>
          </div>
        </div>
        <div class="result-badges">
          <span class="status-badge ${item.status === 'kept_original_name' ? 'status-muted' : 'status-renamed'}">
            ${item.status === 'kept_original_name' ? '保留原名' : '已重命名'}
          </span>
        </div>
        <div class="tag-list">
          ${item.labels.length
            ? item.labels.map((label) => `
              <div class="tag">
                <span>${label.label}</span>
                <strong>${label.confidence_percentage}%</strong>
              </div>
            `).join('')
            : '<div class="tag tag-empty"><span>未识别到满足阈值的元素</span></div>'}
        </div>
        <div class="path-block">
          <p class="field-label">输出路径</p>
          <p>${item.output_path}</p>
        </div>
        ${item.message ? `<p class="inline-note">${item.message}</p>` : ''}
      </article>
    `).join('')}
  `;
}

function renderErrors(result) {
  if (!result.errors.length) {
    errorPanel.className = 'panel hidden';
    errorPanel.innerHTML = '';
    return;
  }

  errorPanel.className = 'panel error-list';
  errorPanel.innerHTML = `
    <div class="panel-head">
      <h3>失败结果</h3>
      <span>${result.errors.length} 张</span>
    </div>
    ${result.errors.map((item) => `
      <article class="error-row">
        <div>
          <p class="field-label">文件</p>
          <p class="file-name">${item.filename}</p>
        </div>
        <div>
          <p class="field-label">原因</p>
          <p class="error-text">${item.error}</p>
        </div>
      </article>
    `).join('')}
  `;
}

async function fetchInfo() {
  try {
    const response = await fetch(`${apiBase}/info`);
    if (!response.ok) {
      throw new Error('无法读取后端配置');
    }
    const data = await response.json();
    renderMeta(data);
    hideBanner();
  } catch (error) {
    showBanner('无法连接本地后端，请先启动 FastAPI 服务。');
  }
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`${apiBase}/process-directory/${jobId}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || '无法获取任务状态');
    }

    updateProgress(data);

    if (data.status === 'completed') {
      stopPolling();
      submitButton.disabled = false;
      submitButton.textContent = '开始批量处理';
      renderSummary(data);
      renderSuccess(data);
      renderErrors(data);
      return;
    }

    if (data.status === 'failed') {
      stopPolling();
      submitButton.disabled = false;
      submitButton.textContent = '开始批量处理';
      showBanner(data.message || '处理失败');
      return;
    }

    activePollTimer = setTimeout(() => pollJob(jobId), 800);
  } catch (error) {
    stopPolling();
    submitButton.disabled = false;
    submitButton.textContent = '开始批量处理';
    showBanner(error.message || '无法获取处理进度');
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  hideBanner();
  summary.className = 'summary-grid hidden';
  successPanel.className = 'panel hidden';
  errorPanel.className = 'panel hidden';
  stopPolling();
  resetProgress();
  submitButton.disabled = true;
  submitButton.textContent = '创建任务中...';

  const maxLabelsRaw = Number(document.getElementById('max_labels').value) || 1;
  const maxLabelsCap = serverMaxLabels || Number(document.getElementById('max_labels').max) || 3;
  const maxLabels = Math.min(maxLabelsRaw, maxLabelsCap);

  const payload = {
    source_path: document.getElementById('source_path').value.trim(),
    output_path: document.getElementById('output_path').value.trim(),
    confidence_threshold: Number(confidenceRange.value),
    max_labels: maxLabels,
    recursive: document.getElementById('recursive').checked,
    include_camera: document.getElementById('include_camera').checked,
    include_type: document.getElementById('include_type').checked,
    include_elements: document.getElementById('include_elements').checked,
    label_language: document.querySelector('input[name="label_language"]:checked').value
  };

  try {
    const response = await fetch(`${apiBase}/process-directory`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || '处理失败');
    }

    activeJobId = data.job_id;
    submitButton.textContent = '处理中...';
    updateProgress(data);
    await pollJob(activeJobId);
  } catch (error) {
    stopPolling();
    submitButton.disabled = false;
    submitButton.textContent = '开始批量处理';
    showBanner(error.message || '处理失败');
  }
});

confidenceRange.addEventListener('input', (event) => {
  setConfidenceValue(event.target.value);
});

confidenceInput.addEventListener('input', (event) => {
  setConfidenceValue(event.target.value);
});

setConfidenceValue(0.01);
resetProgress();
fetchInfo();
