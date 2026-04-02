document.addEventListener('alpine:init', () => {
  Alpine.data('app', () => ({
    mode: 'default',
    useDefaultData: true,
    availableModels: { teacher: [], student: [] },
    savedTasks: [],

    task: {
      name: '', description: '', guidelines: '',
      fields: [], metrics: [], examples: [],
      train_ratio: 0.6, val_ratio: 0.2, test_ratio: 0.2,
    },

    teacherModel: 'gemini/gemini-3.1-pro-preview',
    studentModels: ['gemini/gemini-2.5-flash-lite'],
    evalTrials: 10,
    threads: 50,

    jobId: null,
    jobStatus: null,
    polling: null,
    results: null,
    error: null,

    async init() {
      const [modelsRes, tasksRes] = await Promise.all([
        fetch('/api/models').then(r => r.json()),
        fetch('/api/tasks').then(r => r.json()),
      ]);
      this.availableModels = modelsRes;
      this.savedTasks = tasksRes;
      await this.loadDefaultTask();
    },

    async loadDefaultTask() {
      this.mode = 'default';
      this.useDefaultData = true;
      const defaultName = this.savedTasks.find(t => t.includes('liquid_death'));
      if (defaultName) {
        const data = await fetch(`/api/tasks/${defaultName}`).then(r => r.json());
        this.task = data;
      }
    },

    freshTask() {
      this.mode = 'fresh';
      this.useDefaultData = false;
      this.task = {
        name: '', description: '', guidelines: '',
        fields: [
          { name: 'input_text', description: 'The text to analyze', field_type: 'input', value_type: 'str' },
          { name: 'output_label', description: 'The classification result', field_type: 'output', value_type: 'str' },
        ],
        metrics: [
          { name: 'accuracy', metric_type: 'exact_match', weight: 1.0, target_field: 'output_label', rule_config: {}, custom_code: '' },
        ],
        examples: [],
        train_ratio: 0.6, val_ratio: 0.2, test_ratio: 0.2,
      };
      this.results = null;
      this.jobStatus = null;
    },

    addField(type) {
      this.task.fields.push({ name: '', description: '', field_type: type, value_type: 'str' });
    },
    removeField(idx) {
      this.task.fields.splice(idx, 1);
    },

    addMetric() {
      this.task.metrics.push({
        name: '', metric_type: 'exact_match', weight: 0.5,
        target_field: this.task.fields.find(f => f.field_type === 'output')?.name || '',
        rule_config: {}, custom_code: '',
      });
    },
    removeMetric(idx) {
      this.task.metrics.splice(idx, 1);
    },

    addExample() {
      const row = {};
      this.task.fields.forEach(f => { row[f.name] = ''; });
      this.task.examples.push(row);
    },
    removeExample(idx) {
      this.task.examples.splice(idx, 1);
    },

    importCSV() {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.csv,.tsv';
      input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const text = await file.text();
        const sep = file.name.endsWith('.tsv') ? '\t' : ',';
        const lines = text.trim().split('\n');
        if (lines.length < 2) return;
        const headers = lines[0].split(sep).map(h => h.trim().replace(/^"|"$/g, ''));
        for (let i = 1; i < lines.length; i++) {
          const vals = lines[i].split(sep).map(v => v.trim().replace(/^"|"$/g, ''));
          const row = {};
          headers.forEach((h, j) => { row[h] = vals[j] || ''; });
          this.task.examples.push(row);
        }
      };
      input.click();
    },

    addStudentModel() {
      this.studentModels.push(this.availableModels.student[0]?.id || '');
    },
    removeStudentModel(idx) {
      this.studentModels.splice(idx, 1);
    },

    get exampleSplitInfo() {
      const n = this.task.examples.length;
      const tr = Math.round(n * this.task.train_ratio);
      const va = Math.round(n * this.task.val_ratio);
      const te = n - tr - va;
      return `${n} examples (${tr} train / ${va} val / ${te} test)`;
    },

    get outputFields() {
      return this.task.fields.filter(f => f.field_type === 'output');
    },

    // Model helpers
    getModelLabel(modelId, list) {
      const m = list.find(x => x.id === modelId);
      if (!m) return modelId;
      return `${m.name} - $${m.input_cost}/$${m.output_cost} per 1M tokens`;
    },

    async startRun() {
      this.error = null;
      this.results = null;
      this.jobStatus = { status: 'pending', current_step: 'Starting...', progress_pct: 0 };

      const payload = {
        task: this.task,
        teacher_model: this.teacherModel,
        student_models: this.studentModels.filter(m => m.trim()),
        num_eval_trials: this.evalTrials,
        threads: this.threads,
      };

      try {
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json();
          this.error = err.detail || 'Failed to start';
          this.jobStatus = null;
          return;
        }
        const { job_id } = await res.json();
        this.jobId = job_id;
        this.polling = setInterval(() => this.pollStatus(), 2000);
      } catch (e) {
        this.error = e.message;
        this.jobStatus = null;
      }
    },

    async pollStatus() {
      if (!this.jobId) return;
      try {
        const res = await fetch(`/api/jobs/${this.jobId}`);
        if (res.status === 404) {
          clearInterval(this.polling);
          this.jobId = null;
          this.jobStatus = null;
          this.error = 'Job lost -- the server restarted. Please run again.';
          return;
        }
        const status = await res.json();
        this.jobStatus = status;

        if (status.status === 'completed') {
          clearInterval(this.polling);
          this.results = status.results;
        } else if (status.status === 'failed') {
          clearInterval(this.polling);
          this.error = status.error;
        }
      } catch (e) {
        // keep polling
      }
    },

    async copyPrompt(model) {
      const prompt = this.results?.students?.[model]?.prompt;
      if (!prompt) return;
      try {
        await navigator.clipboard.writeText(prompt);
        // Brief visual feedback
        const btn = event.target;
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1500);
      } catch (e) {
        // Fallback for non-HTTPS contexts
        const ta = document.createElement('textarea');
        ta.value = prompt;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
    },

    async downloadPrompt(model) {
      const prompt = this.results?.students?.[model]?.prompt;
      if (!prompt) return;
      const blob = new Blob([prompt], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `production_prompt_${model.split('/').pop()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    },

    getResultColumns() {
      if (!this.results) return [];
      const cols = [];
      const mono = this.results.monolith;
      cols.push({
        label: 'Monolith', model: mono.model.split('/').pop(),
        scores: mono.scores, latency: mono.latency, cost: mono.cost,
      });
      for (const [model, data] of Object.entries(this.results.students)) {
        const short = model.split('/').pop();
        const cost = data.cost;
        cols.push({ label: 'Naive', model: short, scores: data.naive.scores, latency: data.naive.latency, cost });
        cols.push({ label: 'DSPy', model: short, scores: data.optimized.scores, latency: data.optimized.latency, cost });
      }
      return cols;
    },

    getMetricNames() {
      if (!this.results) return [];
      const cols = this.getResultColumns();
      if (cols.length === 0) return [];
      return Object.keys(cols[0].scores);
    },

    formatScore(score) {
      if (!score) return '-';
      const pct = (score.mean * 100).toFixed(1) + '%';
      if (score.std < 0.001) return pct;
      return pct + ' +/-' + (score.std * 100).toFixed(1) + '%';
    },

    formatLatency(lat) {
      if (!lat) return '-';
      return Math.round(lat.mean).toLocaleString() + 'ms';
    },

    formatCost(cost) {
      if (!cost || cost.input_cost === null) return '-';
      return `$${cost.input_cost} / $${cost.output_cost}`;
    },

    isBest(metricName, colIdx) {
      const cols = this.getResultColumns();
      const values = cols.map(c => c.scores[metricName]?.mean || 0);
      const max = Math.max(...values);
      return cols[colIdx].scores[metricName]?.mean === max && max > 0;
    },

    isFastestLatency(colIdx) {
      const cols = this.getResultColumns();
      const values = cols.map(c => c.latency?.mean || Infinity);
      const min = Math.min(...values);
      return cols[colIdx].latency?.mean === min;
    },

    isCheapest(colIdx) {
      const cols = this.getResultColumns();
      const values = cols.map(c => c.cost?.input_cost ?? Infinity);
      const min = Math.min(...values);
      return (cols[colIdx].cost?.input_cost ?? Infinity) === min && min < Infinity;
    },
  }));
});
