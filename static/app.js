/* ===================== app.js — ShieldAI Dashboard ===================== */

const API = 'http://localhost:5000/api';
let modelStats = null;
let demoProfiles = [];

const FEATURE_GROUPS_META = {
  Identity:   { color: '#6366f1', icon: '🪪' },
  Activity:   { color: '#10b981', icon: '📊' },
  Network:    { color: '#f59e0b', icon: '🌐' },
  Behavioral: { color: '#ef4444', icon: '🔄' },
  Content:    { color: '#8b5cf6', icon: '📝' },
  Engagement: { color: '#06b6d4', icon: '💬' }
};

const FEATURE_LABELS = {
  has_profile_picture: 'Has Profile Picture (0/1)',
  username_digit_ratio: 'Username Digit Ratio (0–1)',
  username_length: 'Username Length',
  has_bio: 'Has Bio (0/1)',
  bio_length: 'Bio Length (chars)',
  bio_url_count: 'URLs in Bio',
  has_external_url: 'Has External URL (0/1)',
  is_verified: 'Is Verified (0/1)',
  account_age_days: 'Account Age (days)',
  posts_count: 'Total Posts',
  posts_per_day: 'Posts Per Day',
  avg_post_likes: 'Avg Post Likes',
  avg_post_comments: 'Avg Post Comments',
  avg_post_shares: 'Avg Post Shares',
  story_frequency: 'Story Frequency (0–1)',
  live_frequency: 'Live Frequency (0–1)',
  followers_count: 'Followers Count',
  following_count: 'Following Count',
  follower_following_ratio: 'Follower/Following Ratio',
  mutual_friends_ratio: 'Mutual Friends Ratio (0–1)',
  fake_follower_pct: 'Fake Follower % (0–1)',
  avg_daily_posts: 'Avg Daily Posts',
  posting_hour_variance: 'Posting Hour Variance',
  night_posting_ratio: 'Night Posting Ratio (0–1)',
  weekend_posting_ratio: 'Weekend Posting Ratio (0–1)',
  hashtag_avg: 'Avg Hashtags per Post',
  mention_avg: 'Avg Mentions per Post',
  caption_length_avg: 'Avg Caption Length',
  emoji_usage_ratio: 'Emoji Usage Ratio (0–1)',
  uses_automation: 'Uses Automation (0/1)',
  content_diversity_score: 'Content Diversity (0–1)',
  repost_ratio: 'Repost Ratio (0–1)',
  original_content_ratio: 'Original Content (0–1)',
  spam_keyword_count: 'Spam Keyword Count',
  link_in_posts_ratio: 'Link in Posts Ratio (0–1)',
  sentiment_variance: 'Sentiment Variance (0–1)',
  engagement_rate: 'Engagement Rate (0–1)',
  comment_to_like_ratio: 'Comment/Like Ratio (0–1)',
  profile_completeness: 'Profile Completeness (0–1)',
  response_time_hours: 'Avg Response Time (hrs)'
};

/* ——— PARTICLES ——— */
function initParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  for (let i = 0; i < 18; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 5 + 2;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random()*100}%;
      top:${Math.random()*100}%;
      animation-duration:${Math.random()*12+8}s;
      animation-delay:${Math.random()*6}s;
      opacity:${Math.random()*0.5+0.1};
    `;
    container.appendChild(p);
  }
}

/* ——— TAB SWITCHING ——— */
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-page').forEach(p => p.classList.toggle('active', p.id === `page-${name}`));
  document.getElementById('hero').style.display = (name === 'dashboard') ? 'flex' : 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

/* ——— TOAST ——— */
function showToast(msg, duration = 3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), duration);
}

/* ——— FETCH MODEL STATS ——— */
async function loadModelStats() {
  try {
    const res = await fetch(`${API}/model_stats`);
    if (!res.ok) throw new Error('API not ready');
    modelStats = await res.json();
    renderDashboard();
    renderModelsTab();
  } catch (e) {
    showToast('⚠️ Backend not running. Run app.py first.');
  }
}

/* ——— DASHBOARD ——— */
function renderDashboard() {
  if (!modelStats) return;
  const best = modelStats.models[modelStats.best_model];

  // Hero stats
  document.getElementById('stat-auc').textContent = best.roc_auc.toFixed(3);
  document.getElementById('stat-acc').textContent = (best.accuracy * 100).toFixed(1) + '%';
  document.getElementById('stat-f1').textContent = best.f1_score.toFixed(3);

  // KPIs
  document.getElementById('kpi-model').textContent = modelStats.best_model;
  document.getElementById('kpi-auc').textContent = best.roc_auc.toFixed(4);
  document.getElementById('kpi-samples').textContent = (modelStats.train_samples + modelStats.test_samples).toLocaleString();
  document.getElementById('kpi-f1').textContent = best.f1_score.toFixed(4);

  // Feature groups
  const fg = document.getElementById('feature-groups');
  const groupSizes = { Identity: 9, Activity: 8, Network: 5, Behavioral: 9, Content: 6, Engagement: 3 };
  const totalFeats = 40;
  fg.innerHTML = Object.entries(FEATURE_GROUPS_META).map(([g, m]) => {
    const cnt = groupSizes[g] || 6;
    const pct = (cnt / totalFeats * 100).toFixed(0);
    return `<div class="fg-row">
      <div class="fg-dot" style="background:${m.color}"></div>
      <div class="fg-name">${m.icon} ${g}</div>
      <div class="fg-bar-wrap"><div class="fg-bar" style="width:${pct}%;background:${m.color}"></div></div>
      <div class="fg-count">${cnt}</div>
    </div>`;
  }).join('');

  // Leaderboard
  const tbody = document.getElementById('leaderboard-body');
  const sorted = Object.entries(modelStats.models).sort((a, b) => b[1].roc_auc - a[1].roc_auc);
  tbody.innerHTML = sorted.map(([name, m], i) => {
    const isBest = name === modelStats.best_model;
    const badge = isBest ? `<span class="badge-best">⭐ Best</span>` : `<span class="badge-good">Active</span>`;
    return `<tr>
      <td style="color:var(--text-muted)">${i + 1}</td>
      <td style="font-weight:700;color:${isBest ? 'var(--purple-light)' : 'inherit'}">${name}</td>
      <td style="font-family:'JetBrains Mono',monospace;color:var(--cyan)">${m.roc_auc.toFixed(4)}</td>
      <td style="font-family:'JetBrains Mono',monospace">${(m.accuracy * 100).toFixed(2)}%</td>
      <td style="font-family:'JetBrains Mono',monospace">${m.f1_score.toFixed(4)}</td>
      <td style="font-family:'JetBrains Mono',monospace">${m.precision.toFixed(4)}</td>
      <td style="font-family:'JetBrains Mono',monospace">${m.recall.toFixed(4)}</td>
      <td style="font-family:'JetBrains Mono',monospace;color:var(--text-sec)">${m.cv_auc_mean.toFixed(4)} ± ${m.cv_auc_std.toFixed(4)}</td>
      <td>${badge}</td>
    </tr>`;
  }).join('');
}

/* ——— MODELS TAB ——— */
function renderModelsTab() {
  if (!modelStats) return;
  const grid = document.getElementById('models-grid');
  const sorted = Object.entries(modelStats.models).sort((a, b) => b[1].roc_auc - a[1].roc_auc);
  grid.innerHTML = sorted.map(([name, m]) => {
    const isBest = name === modelStats.best_model;
    return `<div class="model-card ${isBest ? 'best-model' : ''}">
      <div class="model-name">${name}</div>
      ${isBest ? '<div class="model-best-tag">⭐ Best Performing Model</div>' : ''}
      <div class="model-metrics">
        <div class="model-metric">
          <div class="model-metric-val" style="color:var(--cyan)">${m.roc_auc.toFixed(4)}</div>
          <div class="model-metric-lbl">ROC-AUC</div>
        </div>
        <div class="model-metric">
          <div class="model-metric-val" style="color:var(--green)">${(m.accuracy*100).toFixed(2)}%</div>
          <div class="model-metric-lbl">Accuracy</div>
        </div>
        <div class="model-metric">
          <div class="model-metric-val" style="color:var(--purple-light)">${m.f1_score.toFixed(4)}</div>
          <div class="model-metric-lbl">F1-Score</div>
        </div>
        <div class="model-metric">
          <div class="model-metric-val" style="color:var(--amber)">${m.precision.toFixed(4)}</div>
          <div class="model-metric-lbl">Precision</div>
        </div>
      </div>
      <div class="model-cv">CV-AUC: ${m.cv_auc_mean.toFixed(4)} ± ${m.cv_auc_std.toFixed(4)}</div>
    </div>`;
  }).join('');
}

/* ——— ANALYZER FORM ——— */
function buildAnalyzerForm() {
  const groups = {
    Identity:   ['has_profile_picture','username_digit_ratio','username_length','has_bio','bio_length','bio_url_count','has_external_url','is_verified','profile_completeness'],
    Activity:   ['account_age_days','posts_count','posts_per_day','avg_post_likes','avg_post_comments','avg_post_shares','story_frequency','live_frequency'],
    Network:    ['followers_count','following_count','follower_following_ratio','mutual_friends_ratio','fake_follower_pct'],
    Behavioral: ['avg_daily_posts','posting_hour_variance','night_posting_ratio','weekend_posting_ratio','hashtag_avg','mention_avg','caption_length_avg','emoji_usage_ratio','uses_automation'],
    Content:    ['content_diversity_score','repost_ratio','original_content_ratio','spam_keyword_count','link_in_posts_ratio','sentiment_variance'],
    Engagement: ['engagement_rate','comment_to_like_ratio','response_time_hours']
  };

  const container = document.getElementById('form-sections');
  container.innerHTML = Object.entries(groups).map(([group, feats]) => {
    const meta = FEATURE_GROUPS_META[group];
    const fields = feats.map(f => `
      <div class="form-field">
        <label for="f_${f}">${FEATURE_LABELS[f] || f}</label>
        <input type="number" id="f_${f}" name="${f}" step="any" min="0" placeholder="0" value="0"/>
      </div>`).join('');
    return `<div class="form-group-section">
      <div class="form-group-title" style="color:${meta.color}">${meta.icon} ${group}</div>
      <div class="form-grid">${fields}</div>
    </div>`;
  }).join('');
}

function getFormValues() {
  const data = {};
  document.querySelectorAll('#analyzerForm input[name]').forEach(inp => {
    data[inp.name] = parseFloat(inp.value) || 0;
  });
  return data;
}

function clearForm() {
  document.querySelectorAll('#analyzerForm input').forEach(inp => inp.value = 0);
  document.getElementById('result-panel').querySelector('.result-placeholder').style.display = 'flex';
  document.getElementById('result-content').style.display = 'none';
}

function loadRandomProfile() {
  if (!demoProfiles.length) { showToast('Demo profiles not loaded yet'); return; }
  const p = demoProfiles[Math.floor(Math.random() * demoProfiles.length)];
  document.querySelectorAll('#analyzerForm input[name]').forEach(inp => {
    if (p[inp.name] !== undefined) inp.value = p[inp.name];
  });
  showToast(`✅ Loaded: ${p.name || p.username}`);
}

document.getElementById('analyzerForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('analyzeBtn');
  btn.innerHTML = '<span class="spinner"></span> Analyzing...';
  btn.disabled = true;

  try {
    const data = getFormValues();
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const result = await res.json();
    renderResult(result);
  } catch (err) {
    showToast('❌ Error: Backend not reachable');
  } finally {
    btn.innerHTML = '🔍 Analyze Profile';
    btn.disabled = false;
  }
});

function renderResult(result) {
  document.querySelector('.result-placeholder').style.display = 'none';
  const content = document.getElementById('result-content');
  content.style.display = 'flex';

  const colorMap = { low: '#10b981', medium: '#f59e0b', high: '#f97316', critical: '#ef4444' };
  const emojiMap = { low: '✅', medium: '⚠️', high: '🚨', critical: '🔴' };
  const color = colorMap[result.risk_level] || '#6366f1';

  const card = document.getElementById('verdict-card');
  card.style.borderColor = color + '60';
  card.style.background = `linear-gradient(135deg, rgba(7,11,20,0.95), ${color}12)`;

  document.getElementById('verdict-icon').textContent = emojiMap[result.risk_level] || '❓';
  const vl = document.getElementById('verdict-label');
  vl.textContent = result.verdict;
  vl.style.color = color;

  const pct = Math.round(result.probability * 100);
  document.getElementById('prob-bar').style.width = pct + '%';
  document.getElementById('prob-bar').style.background = color;
  document.getElementById('prob-text').textContent = `Fake Probability: ${pct}% — Confidence: ${Math.round(result.confidence * 100)}%`;

  document.getElementById('verdict-meta').innerHTML = `
    <span>🤖 Model: <strong>${result.model_used}</strong></span>
    <span>📊 Score: <strong>${result.probability.toFixed(4)}</strong></span>
  `;

  // Group scores
  const scoresBody = document.getElementById('group-scores-body');
  if (result.group_risk_scores) {
    scoresBody.innerHTML = Object.entries(result.group_risk_scores).map(([group, score]) => {
      const meta = FEATURE_GROUPS_META[group] || { color: '#6366f1' };
      const scoreColor = score > 60 ? '#ef4444' : score > 35 ? '#f59e0b' : '#10b981';
      return `<div class="score-row">
        <div class="score-name">${meta.icon || ''} ${group}</div>
        <div class="score-bar-wrap"><div class="score-bar" style="width:${Math.min(score,100)}%;background:${scoreColor}"></div></div>
        <div class="score-val" style="color:${scoreColor}">${score.toFixed(0)}%</div>
      </div>`;
    }).join('');
    document.getElementById('group-scores-card').style.display = 'block';
  }
}

/* ——— DEMO PROFILES TAB ——— */
async function loadDemoProfiles() {
  try {
    const res = await fetch(`${API}/demo_profiles`);
    demoProfiles = await res.json();
    renderDemoTab();
  } catch (e) {
    document.getElementById('demo-grid').innerHTML = '<p style="color:var(--text-sec);padding:20px">Backend not running.</p>';
  }
}

async function analyzeAndShowDemo(profile) {
  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile)
    });
    const result = await res.json();
    renderDemoTab(result, profile);
  } catch (e) {
    showToast('❌ Backend error');
  }
}

function renderDemoTab(liveResult, activeProfile) {
  const grid = document.getElementById('demo-grid');
  const typeColors = { 'Real User': '#10b981', 'Bot Account': '#ef4444', 'Impersonator': '#f97316', 'Real Influencer': '#6366f1' };
  const avatarEmojis = { 'Real User': '👩', 'Bot Account': '🤖', 'Impersonator': '🎭', 'Real Influencer': '⭐' };

  grid.innerHTML = demoProfiles.map((p, idx) => {
    const tColor = typeColors[p.type] || '#6366f1';
    const avatar = avatarEmojis[p.type] || '👤';
    return `<div class="demo-card" id="demo-card-${idx}">
      <div class="demo-card-top">
        <div class="demo-avatar" style="background:${tColor}22;color:${tColor}">${avatar}</div>
        <div>
          <div class="demo-name">${p.name}</div>
          <div class="demo-username">@${p.username}</div>
        </div>
        <div class="demo-type-badge" style="background:${tColor}22;color:${tColor}">${p.type}</div>
      </div>
      <div class="demo-metrics">
        <div class="demo-metric"><div class="demo-metric-val">${(p.followers_count||0).toLocaleString()}</div><div class="demo-metric-lbl">Followers</div></div>
        <div class="demo-metric"><div class="demo-metric-val">${p.posts_count||0}</div><div class="demo-metric-lbl">Posts</div></div>
        <div class="demo-metric"><div class="demo-metric-val">${p.account_age_days||0}d</div><div class="demo-metric-lbl">Age</div></div>
      </div>
      <div class="demo-verdict" id="verdict-${idx}" style="background:rgba(99,102,241,0.08);color:var(--text-sec)">
        Expected: ${p.expected} — Click Analyze to verify
      </div>
      <div class="demo-footer">
        <button class="btn-sm" style="background:rgba(99,102,241,0.15);color:var(--purple-light)" onclick="loadProfileToAnalyzer(${idx})">Load in Analyzer</button>
        <button class="btn-sm btn-primary" onclick="runDemoAnalysis(${idx})">🔍 Analyze</button>
      </div>
    </div>`;
  }).join('');
}

async function runDemoAnalysis(idx) {
  const profile = demoProfiles[idx];
  if (!profile) return;
  const verdict = document.getElementById(`verdict-${idx}`);
  verdict.textContent = 'Analyzing...';
  verdict.style.color = 'var(--text-sec)';

  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile)
    });
    const result = await res.json();
    const colorMap = { low: '#10b981', medium: '#f59e0b', high: '#f97316', critical: '#ef4444' };
    const color = colorMap[result.risk_level] || '#6366f1';
    const pct = Math.round(result.probability * 100);
    verdict.style.background = color + '18';
    verdict.style.color = color;
    verdict.textContent = `${result.verdict} — ${pct}% fake probability`;
  } catch (e) {
    verdict.textContent = '❌ Backend error';
  }
}

function loadProfileToAnalyzer(idx) {
  const p = demoProfiles[idx];
  if (!p) return;
  Object.keys(p).forEach(key => {
    const inp = document.getElementById(`f_${key}`);
    if (inp) inp.value = p[key];
  });
  switchTab('analyzer');
  showToast(`✅ Loaded "${p.name}" into analyzer`);
}

/* ——— INIT ——— */
async function init() {
  initParticles();
  buildAnalyzerForm();
  await loadModelStats();
  await loadDemoProfiles();
}

init();
