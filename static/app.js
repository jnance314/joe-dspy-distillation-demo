document.addEventListener('alpine:init', () => {
  Alpine.data('app', () => ({
    availableModels: { teacher: [], student: [] },
    savedTasks: [],
    selectedTask: null,

    task: {
      name: '', description: '', guidelines: '', module_key: '',
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

    // Edited prompt re-evaluation state
    editedPrompts: {},      // {columnKey: "edited prompt text"}
    originalPrompts: {},    // {columnKey: "original prompt text"} (for reset)
    editedResults: null,    // Table 2 columns array
    editedLoading: false,
    activeEditTab: null,    // which column's editor is open

    async init() {
      const [modelsRes, tasksRes] = await Promise.all([
        fetch('/api/models').then(r => r.json()),
        fetch('/api/tasks').then(r => r.json()),
      ]);
      this.availableModels = modelsRes;
      this.savedTasks = tasksRes;
      if (tasksRes.length > 0) {
        await this.loadTask(tasksRes[0]);
      }
    },

    async loadTask(taskName) {
      this.selectedTask = taskName;
      this.results = null;
      this.jobStatus = null;
      this.error = null;
      this.editedPrompts = {};
      this.originalPrompts = {};
      this.editedResults = null;
      this.activeEditTab = null;
      const data = await fetch(`/api/tasks/${taskName}`).then(r => r.json());
      this.task = data;
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

    // -- Metric descriptions --------------------------------------------------

    getMetricDescription(m) {
      const field = m.target_field;
      switch (m.metric_type) {
        case 'exact_match':
          return `Compares the model's "${field}" output to the expected value. Case-insensitive. Score: 1.0 if they match, 0.0 if not. Example: expected "true", model says "true" → 1.0; model says "false" → 0.0.`;
        case 'f1_phrases':
          return `Parses comma-separated phrases from "${field}" and computes precision + recall (F1) with fuzzy matching. Catches partial matches. Example: expected "premium quality, leverage", model outputs "premium quality, best-in-class" → precision 50%, recall 50%, F1 = 50%.`;
        case 'rule_quality': {
          const rules = [];
          if (m.rule_config?.banned_words?.length) rules.push(`no banned words (${m.rule_config.banned_words.length} phrases)`);
          if (m.rule_config?.no_passive_voice) rules.push('no passive voice');
          if (m.rule_config?.max_sentence_length) rules.push(`sentences under ${m.rule_config.max_sentence_length} words`);
          const ruleStr = rules.length ? rules.join(', ') : 'configured rules';
          return `Checks "${field}" against structural rules: ${ruleStr}. Each rule that passes adds to the score. Example: 3 rules configured, 2 pass → score 0.67.`;
        }
        case 'custom':
          return `Custom Python scoring function for "${field}". Returns a float between 0.0 (worst) and 1.0 (best). The logic is specific to this task.`;
        default:
          return `Scores the "${field}" output.`;
      }
    },

    getFieldDescription(f) {
      if (f.field_type === 'input') {
        return `This is what the model receives. ${f.description}`;
      }
      return `This is what the model produces. ${f.description}. The optimizer scores this against the expected value in the training data.`;
    },

    // -- Run experiment -------------------------------------------------------

    async startRun() {
      this.error = null;
      this.results = null;
      this.editedPrompts = {};
      this.originalPrompts = {};
      this.editedResults = null;
      this.activeEditTab = null;
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
          this._initEditedPrompts();
        } else if (status.status === 'failed') {
          clearInterval(this.polling);
          this.error = status.error;
        }
      } catch (e) {
        // keep polling
      }
    },

    // -- Edited prompt system -------------------------------------------------

    _initEditedPrompts() {
      if (!this.results) return;

      // Monolith
      const monoPrompt = this.results.monolith.prompt || '';
      this.editedPrompts['monolith'] = monoPrompt;
      this.originalPrompts['monolith'] = monoPrompt;

      // Per student: naive + dspy
      for (const [model, data] of Object.entries(this.results.students)) {
        const naivePrompt = data.naive?.prompt || '';
        const dspyPrompt = data.prompt || '';
        this.editedPrompts[`naive:${model}`] = naivePrompt;
        this.originalPrompts[`naive:${model}`] = naivePrompt;
        this.editedPrompts[`dspy:${model}`] = dspyPrompt;
        this.originalPrompts[`dspy:${model}`] = dspyPrompt;
      }

      // Default to first tab
      const keys = this.getEditTabKeys();
      if (keys.length > 0) this.activeEditTab = keys[0];
    },

    getEditTabKeys() {
      if (!this.results) return [];
      const keys = ['monolith'];
      for (const model of Object.keys(this.results.students)) {
        keys.push(`naive:${model}`);
        keys.push(`dspy:${model}`);
      }
      return keys;
    },

    getEditTabLabel(key) {
      if (key === 'monolith') return 'Monolith';
      const [type, model] = [key.split(':')[0], key.split(':').slice(1).join(':')];
      const short = model.split('/').pop();
      if (type === 'naive') return `Naive (${short})`;
      return `DSPy (${short})`;
    },

    getEditTabModel(key) {
      if (key === 'monolith') return this.results.monolith.model;
      return key.split(':').slice(1).join(':');
    },

    resetPrompt(key) {
      this.editedPrompts[key] = this.originalPrompts[key] || '';
    },

    async evalEdited() {
      this.editedLoading = true;
      this.editedResults = null;
      this.error = null;

      // Use simple labels (Monolith, Naive, DSPy) to match Table 1 columns
      const columns = this.getEditTabKeys().map(key => {
        let label = 'Monolith';
        if (key.startsWith('naive:')) label = 'Naive';
        else if (key.startsWith('dspy:')) label = 'DSPy';
        return {
          label,
          model: this.getEditTabModel(key),
          edited_prompt: this.editedPrompts[key] || '',
        };
      });

      try {
        const res = await fetch('/api/eval-edited', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            task: this.task,
            columns: columns,
            num_trials: this.evalTrials,
            threads: this.threads,
          }),
        });
        if (!res.ok) {
          const err = await res.json();
          this.error = err.detail || 'Re-evaluation failed';
          this.editedLoading = false;
          return;
        }
        const data = await res.json();
        this.editedResults = data.columns;
      } catch (e) {
        this.error = e.message;
      }
      this.editedLoading = false;
    },

    // -- Prompt copy/download -------------------------------------------------

    async copyPrompt(model) {
      const prompt = this.results?.students?.[model]?.prompt;
      if (!prompt) return;
      try {
        await navigator.clipboard.writeText(prompt);
        const btn = event.target;
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1500);
      } catch (e) {
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

    // -- Table rendering helpers ----------------------------------------------

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

    isBest(metricName, colIdx, cols) {
      const values = cols.map(c => c.scores[metricName]?.mean || 0);
      const max = Math.max(...values);
      return cols[colIdx].scores[metricName]?.mean === max && max > 0;
    },

    isFastestLatency(colIdx, cols) {
      const values = cols.map(c => c.latency?.mean || Infinity);
      const min = Math.min(...values);
      return cols[colIdx].latency?.mean === min;
    },

    isCheapest(colIdx, cols) {
      const values = cols.map(c => c.cost?.input_cost ?? Infinity);
      const min = Math.min(...values);
      return (cols[colIdx].cost?.input_cost ?? Infinity) === min && min < Infinity;
    },

    // Delta helper for Table 2 vs Table 1
    getDelta(editedCol, metricName) {
      const origCols = this.getResultColumns();
      const orig = origCols.find(c => c.label === editedCol.label && c.model === editedCol.model);
      if (!orig || !orig.scores[metricName] || !editedCol.scores[metricName]) return null;
      return editedCol.scores[metricName].mean - orig.scores[metricName].mean;
    },

    formatDelta(delta) {
      if (delta === null || delta === undefined) return '';
      const sign = delta >= 0 ? '+' : '';
      return `${sign}${(delta * 100).toFixed(1)}%`;
    },

    deltaClass(delta) {
      if (delta === null || delta === undefined) return '';
      if (delta > 0.001) return 'color: var(--success);';
      if (delta < -0.001) return 'color: var(--danger);';
      return 'color: var(--text-dim);';
    },
  }));
});
