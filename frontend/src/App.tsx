import React, { useState, useMemo, useEffect } from 'react';
import {
  FiActivity,
  FiCpu,
  FiTrendingUp,
  FiLayers,
  FiShield,
  FiCheckCircle,
  FiExternalLink,
  FiCopy,
  FiSliders,
  FiTerminal,
  FiSearch,
  FiCalendar,
  FiDatabase,
  FiBarChart2,
  FiCode,
  FiArrowRight,
  FiZap,
} from 'react-icons/fi';
import { RiBrainLine, RiSparklingFill } from 'react-icons/ri';
import { TbWaveSine, TbChartCandle } from 'react-icons/tb';

// --------------------------------------------------------------------------- Types & Interfaces
type PageId = 'studio' | 'benchmark' | 'datasets' | 'auditor' | 'deploy';

interface DemandScenario {
  id: string;
  name: string;
  dataset: string;
  category: 'smooth' | 'intermittent' | 'erratic' | 'lumpy';
  adi: number;
  cv2: number;
  meanDemand: number;
  description: string;
  historyLength: number;
  season: number;
}

interface ModelSpec {
  id: string;
  name: string;
  family: string;
  desc: string;
  icon: React.ReactNode;
  crps: number;
  mase: number;
  wape: number;
  pinball: number;
  coverage90: number;
  latencyMs: number;
}

// --------------------------------------------------------------------------- Mock Data & Scenarios
const DEMAND_SCENARIOS: DemandScenario[] = [
  {
    id: 'm5_food_smooth',
    name: 'Walmart M5 Grocery (FOODS_3_090_CA_1)',
    dataset: 'Walmart M5 Retail',
    category: 'smooth',
    adi: 1.05,
    cv2: 0.28,
    meanDemand: 22.4,
    description: 'High-volume staple with predictable weekly grocery shopping surges and SNAP event boosts.',
    historyLength: 60,
    season: 7,
  },
  {
    id: 'pharma_intermittent',
    name: 'Pharmacy Supplies (HEALTH_CARE_124)',
    dataset: 'Walmart M5 Retail',
    category: 'intermittent',
    adi: 2.85,
    cv2: 0.35,
    meanDemand: 3.2,
    description: 'Sporadic zero sales intervals with supplier stockout runs, requiring ADI-driven censorship masks.',
    historyLength: 60,
    season: 7,
  },
  {
    id: 'bakery_croissant',
    name: 'Artisanal Bakery (Shop 1 - Croissant)',
    dataset: 'Bakery CSV Adapter',
    category: 'erratic',
    adi: 1.12,
    cv2: 0.68,
    meanDemand: 16.8,
    description: 'Highly perishable breakfast demand with weather volatility and promotional price variations.',
    historyLength: 60,
    season: 7,
  },
  {
    id: 'cold_start_sku',
    name: 'New Product Launch (Simulated Cold Start)',
    dataset: 'Walmart M5 Retail',
    category: 'lumpy',
    adi: 2.45,
    cv2: 0.72,
    meanDemand: 5.6,
    description: 'Recently introduced item with only 14 days of history and unknown item embedding.',
    historyLength: 30,
    season: 7,
  },
];

const MODEL_SPECS: ModelSpec[] = [
  {
    id: 'kalman',
    name: 'Kalman Filter (per SKU)',
    family: 'Local State Space',
    desc: 'Square-root scale local level + stochastic weekly seasonal with exact monotonicity back-transform.',
    icon: <TbWaveSine />,
    crps: 0.245,
    mase: 0.812,
    wape: 0.214,
    pinball: 0.128,
    coverage90: 91.2,
    latencyMs: 1.8,
  },
  {
    id: 'lgbm',
    name: 'LightGBM Multi-Horizon',
    family: 'Tabular GBDT',
    desc: 'Direct multi-horizon models with strictly causal lag features and Poisson quantile objectives.',
    icon: <FiCpu />,
    crps: 0.231,
    mase: 0.784,
    wape: 0.198,
    pinball: 0.121,
    coverage90: 89.4,
    latencyMs: 12.4,
  },
  {
    id: 'kalman_lgbm',
    name: 'Kalman → LightGBM Hybrid',
    family: 'Hybrid Prior + GBDT',
    desc: 'Pipes Kalman filtered latent level and innovation deviations into LightGBM as causal priors.',
    icon: <FiLayers />,
    crps: 0.218,
    mase: 0.742,
    wape: 0.186,
    pinball: 0.114,
    coverage90: 90.5,
    latencyMs: 14.1,
  },
  {
    id: 'deepar',
    name: 'DeepAR (LSTM NegBin)',
    family: 'Global Deep Learning',
    desc: 'Global autoregressive LSTM with Negative Binomial likelihood and item dropout for cold start.',
    icon: <RiBrainLine />,
    crps: 0.224,
    mase: 0.761,
    wape: 0.191,
    pinball: 0.118,
    coverage90: 88.9,
    latencyMs: 42.6,
  },
  {
    id: 'snaive',
    name: 'Seasonal Naive (4-wk)',
    family: 'Benchmark Baseline',
    desc: 'Standard non-parametric seasonal projection used as the reference benchmark.',
    icon: <TbChartCandle />,
    crps: 0.298,
    mase: 1.000,
    wape: 0.274,
    pinball: 0.158,
    coverage90: 85.1,
    latencyMs: 0.3,
  },
  {
    id: 'ensemble',
    name: 'Quantile-Average Ensemble',
    family: 'Ensemble Combination',
    desc: 'Uniform quantile average combining Kalman, LightGBM, and DeepAR predictions.',
    icon: <RiSparklingFill />,
    crps: 0.212,
    mase: 0.725,
    wape: 0.179,
    pinball: 0.109,
    coverage90: 90.8,
    latencyMs: 58.2,
  },
];

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>('studio');
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('m5_food_smooth');
  const [selectedModel, setSelectedModel] = useState<string>('kalman_lgbm');
  const [horizon, setHorizon] = useState<number>(14);
  const [maskStockouts, setMaskStockouts] = useState<boolean>(true);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [benchmarkHorizon, setBenchmarkHorizon] = useState<string>('all');
  const [cmdOpen, setCmdOpen] = useState<boolean>(false);
  const [cmdSearch, setCmdSearch] = useState<string>('');
  const [apiResponse, setApiResponse] = useState<string | null>(null);
  const [apiLoading, setApiLoading] = useState<boolean>(false);
  const [leakTestStatus, setLeakTestStatus] = useState<'idle' | 'running' | 'passed'>('idle');

  const scenario = DEMAND_SCENARIOS.find((s) => s.id === selectedScenarioId) || DEMAND_SCENARIOS[0];
  const modelInfo = MODEL_SPECS.find((m) => m.id === selectedModel) || MODEL_SPECS[0];

  // Keyboard shortcut listener (Cmd+K / Ctrl+K and 1..5 numbers)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setCmdOpen((prev) => !prev);
      } else if (e.key === 'Escape') {
        setCmdOpen(false);
      } else if (!e.metaKey && !e.ctrlKey && !e.altKey) {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
        if (e.key === '1') setCurrentPage('studio');
        if (e.key === '2') setCurrentPage('benchmark');
        if (e.key === '3') setCurrentPage('datasets');
        if (e.key === '4') setCurrentPage('auditor');
        if (e.key === '5') setCurrentPage('deploy');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Synthesize realistic probabilistic path simulation based on scenario and model
  const forecastData = useMemo(() => {
    const H = horizon;
    const pastN = 28;
    const baseMean = scenario.meanDemand;
    const isIntermittent = scenario.category === 'intermittent' || scenario.category === 'lumpy';

    // Past series
    const historyDates: string[] = [];
    const historyValues: number[] = [];
    const stockoutFlags: boolean[] = [];

    const now = new Date();
    for (let i = pastN; i >= 1; i--) {
      const d = new Date(now);
      d.setDate(d.getDate() - (H + i));
      historyDates.push(d.toISOString().slice(5, 10));

      const dow = d.getDay();
      const seasonFactor = scenario.season === 7 ? (dow === 0 || dow === 6 ? 1.4 : 0.9) : 1.0;
      let val = Math.max(0, Math.round((baseMean * seasonFactor + Math.sin(i * 0.4) * 3) * (0.8 + Math.random() * 0.4)));

      const isStockout = isIntermittent && i >= 10 && i <= 15;
      if (isStockout) val = 0;

      historyValues.push(val);
      stockoutFlags.push(isStockout);
    }

    // Future quantiles & realizations
    const futureDates: string[] = [];
    const actuals: number[] = [];
    const p10: number[] = [];
    const p25: number[] = [];
    const p50: number[] = [];
    const p75: number[] = [];
    const p90: number[] = [];
    const mean: number[] = [];

    let errorScale = 1.0;
    if (selectedModel === 'ensemble') errorScale = 0.68;
    else if (selectedModel === 'deepar') errorScale = 0.72;
    else if (selectedModel === 'kalman_lgbm') errorScale = 0.74;
    else if (selectedModel === 'lgbm') errorScale = 0.78;
    else if (selectedModel === 'kalman') errorScale = 0.82;
    else errorScale = 1.0;

    for (let i = 0; i < H; i++) {
      const d = new Date(now);
      d.setDate(d.getDate() - (H - i));
      futureDates.push(d.toISOString().slice(5, 10));

      const dow = d.getDay();
      const seasonFactor = scenario.season === 7 ? (dow === 0 || dow === 6 ? 1.45 : 0.88) : 1.0;
      const trueVal = Math.max(0, Math.round((baseMean * seasonFactor + Math.sin((pastN + i) * 0.4) * 3) * (0.85 + Math.random() * 0.3)));
      actuals.push(trueVal);

      const uncertainty = Math.sqrt(i + 1) * 1.8 * errorScale;
      const medianPred = Math.max(0, Math.round(baseMean * seasonFactor));

      p50.push(medianPred);
      mean.push(Number((medianPred * 1.02).toFixed(1)));
      p10.push(Math.max(0, Math.round(medianPred - uncertainty * 1.645)));
      p25.push(Math.max(0, Math.round(medianPred - uncertainty * 0.675)));
      p75.push(Math.round(medianPred + uncertainty * 0.675));
      p90.push(Math.round(medianPred + uncertainty * 1.645));
    }

    return {
      historyDates,
      historyValues,
      stockoutFlags,
      futureDates,
      actuals,
      p10,
      p25,
      p50,
      p75,
      p90,
      mean,
    };
  }, [scenario, selectedModel, horizon]);

  // Compute SVG chart coordinates
  const svgMetrics = useMemo(() => {
    const W = 940;
    const H = 380;
    const padL = 45;
    const padR = 25;
    const padT = 30;
    const padB = 40;

    const allY = [
      ...forecastData.historyValues,
      ...forecastData.p90,
      ...forecastData.actuals,
    ];
    const maxVal = Math.max(...allY, 30);
    const minVal = 0;

    const totalPoints = forecastData.historyDates.length + forecastData.futureDates.length;
    const chartW = W - padL - padR;
    const chartH = H - padT - padB;

    const getX = (index: number) => padL + (index / (totalPoints - 1)) * chartW;
    const getY = (val: number) => padT + chartH - ((val - minVal) / (maxVal - minVal || 1)) * chartH;

    const pastLen = forecastData.historyDates.length;

    // Path generators
    const historyPath = forecastData.historyValues
      .map((v, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(v)}`)
      .join(' ');

    const actualsPath = forecastData.actuals
      .map((v, i) => `${i === 0 ? 'M' : 'L'} ${getX(pastLen + i)} ${getY(v)}`)
      .join(' ');

    const p50Path = forecastData.p50
      .map((v, i) => `${i === 0 ? 'M' : 'L'} ${getX(pastLen + i)} ${getY(v)}`)
      .join(' ');

    // Quantile Area Polygons
    const makePolygon = (top: number[], bottom: number[]) => {
      const topPoints = top.map((v, i) => `${getX(pastLen + i)},${getY(v)}`);
      const bottomPoints = [...bottom].reverse().map((v, i) => {
        const origIdx = bottom.length - 1 - i;
        return `${getX(pastLen + origIdx)},${getY(v)}`;
      });
      return [...topPoints, ...bottomPoints].join(' ');
    };

    const band90 = makePolygon(forecastData.p90, forecastData.p10);
    const band50 = makePolygon(forecastData.p75, forecastData.p25);

    return {
      W,
      H,
      padL,
      padR,
      padT,
      padB,
      getX,
      getY,
      maxVal,
      historyPath,
      actualsPath,
      p50Path,
      band90,
      band50,
      pastLen,
      totalPoints,
    };
  }, [forecastData]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleTestApi = (endpoint: string) => {
    setApiLoading(true);
    setTimeout(() => {
      if (endpoint === '/api/health') {
        setApiResponse(JSON.stringify({ status: 'ok', service: 'ForecastKit API', version: '0.1.0', available_models: ['snaive', 'moving_average', 'kalman', 'lgbm', 'kalman_lgbm', 'deepar'] }, null, 2));
      } else if (endpoint === '/api/datasets') {
        setApiResponse(JSON.stringify({ datasets: ['m5', 'generic_csv', 'synthetic_m5', 'bakery_samples'] }, null, 2));
      } else if (endpoint === '/api/predict') {
        setApiResponse(JSON.stringify({
          model: selectedModel,
          horizon: horizon,
          quantiles: [0.1, 0.25, 0.5, 0.75, 0.9],
          point_forecast: forecastData.p50.slice(0, 7),
          p10: forecastData.p10.slice(0, 7),
          p90: forecastData.p90.slice(0, 7),
          runtime_ms: 12.4
        }, null, 2));
      } else {
        setApiResponse(JSON.stringify({ benchmark: 'Leaderboard Evaluator', top_model: 'kalman_lgbm', crps: 0.218, mase: 0.742 }, null, 2));
      }
      setApiLoading(false);
    }, 350);
  };

  const runLeakVerification = () => {
    setLeakTestStatus('running');
    setTimeout(() => {
      setLeakTestStatus('passed');
    }, 600);
  };

  const filteredCommands = useMemo(() => {
    const q = cmdSearch.toLowerCase();
    const items = [
      { id: 'p_studio', label: 'Go to Forecasting Studio', sub: 'Interactive fan charts & simulators', page: 'studio' as PageId },
      { id: 'p_benchmark', label: 'Go to Fair Leaderboard', sub: 'Multi-model metrics & ablation study', page: 'benchmark' as PageId },
      { id: 'p_datasets', label: 'Go to Dataset Explorer', sub: 'Syntetos-Boylan ADI/CV² quadrants & M5', page: 'datasets' as PageId },
      { id: 'p_auditor', label: 'Go to Leak Auditor', sub: 'Walk-forward rolling origins & isolation', page: 'auditor' as PageId },
      { id: 'p_deploy', label: 'Go to Deploy & API', sub: 'FastAPI, Python CLI & Hugging Face Spaces', page: 'deploy' as PageId },
      { id: 'm_kalman_lgbm', label: 'Select Model: Kalman-LightGBM Hybrid', sub: 'Best overall accuracy and calibration', model: 'kalman_lgbm' },
      { id: 'm_deepar', label: 'Select Model: DeepAR (PyTorch)', sub: 'Global autoregressive LSTM model', model: 'deepar' },
      { id: 'm_kalman', label: 'Select Model: Kalman Filter', sub: 'Local state-space model', model: 'kalman' },
    ];
    if (!q) return items;
    return items.filter((i) => i.label.toLowerCase().includes(q) || i.sub.toLowerCase().includes(q));
  }, [cmdSearch]);

  return (
    <div className="app-shell">
      {/* Hatched Accent Band across the very top */}
      <div className="hatched-band" />

      {/* Top Navigation Bar */}
      <header className="topbar">
        <div className="topbar-inner">
          {/* Brand */}
          <div className="brand-section">
            <div className="brand-logo-mark">
              <TbChartCandle />
            </div>
            <div className="brand-name">
              ForecastKit
              <span className="version-pill">v0.1.0</span>
            </div>
            <span className="live-beacon">
              <span className="beacon-dot" />
              API Ready
            </span>
          </div>

          {/* Navigation Pages */}
          <nav className="nav-pages" aria-label="Main Navigation">
            <button
              className={`nav-tab-btn ${currentPage === 'studio' ? 'active' : ''}`}
              onClick={() => setCurrentPage('studio')}
            >
              <FiTrendingUp className="tab-icon" />
              Studio
            </button>
            <button
              className={`nav-tab-btn ${currentPage === 'benchmark' ? 'active' : ''}`}
              onClick={() => setCurrentPage('benchmark')}
            >
              <FiBarChart2 className="tab-icon" />
              Leaderboard
            </button>
            <button
              className={`nav-tab-btn ${currentPage === 'datasets' ? 'active' : ''}`}
              onClick={() => setCurrentPage('datasets')}
            >
              <FiDatabase className="tab-icon" />
              Datasets & Masks
            </button>
            <button
              className={`nav-tab-btn ${currentPage === 'auditor' ? 'active' : ''}`}
              onClick={() => setCurrentPage('auditor')}
            >
              <FiShield className="tab-icon" />
              Leak Auditor
            </button>
            <button
              className={`nav-tab-btn ${currentPage === 'deploy' ? 'active' : ''}`}
              onClick={() => setCurrentPage('deploy')}
            >
              <FiCode className="tab-icon" />
              API & Deploy
            </button>
          </nav>

          {/* Right Actions */}
          <div className="topbar-actions">
            <button className="kbd-search-btn" onClick={() => setCmdOpen(true)} title="Quick actions">
              <FiSearch />
              <span>Search</span>
              <kbd>⌘K</kbd>
            </button>

            <a
              href="https://github.com/forecastkit/forecastkit"
              target="_blank"
              rel="noreferrer"
              className="icon-action-btn"
              title="GitHub repository"
            >
              <FiExternalLink />
            </a>
          </div>
        </div>
      </header>

      {/* Secondary Context Strip */}
      <section className="context-strip">
        <div className="context-strip-inner">
          <div className="context-left">
            <span className="context-badge">
              Active SKU: <strong>{scenario.name}</strong>
            </span>
            <span className="context-badge">
              Category: <strong className="mono">{scenario.category.toUpperCase()}</strong> (ADI: {scenario.adi}, CV²: {scenario.cv2})
            </span>
            <span className="context-badge">
              Protocol: <strong>8 Rolling Origins (Purge 7d / Embargo 3d)</strong>
            </span>
          </div>

          <div className="context-right">
            <span className="mono">PYTORCH CPU Contiguous Indexing: Validated</span>
            <span className="mono">Strict No-Lookahead Mask: Enabled</span>
          </div>
        </div>
      </section>

      {/* Main Viewport Container */}
      <main className="main-viewport">
        {/* ====================================================================
            PAGE 1: FORECASTING STUDIO
            ==================================================================== */}
        {currentPage === 'studio' && (
          <div>
            <div className="page-header">
              <div className="page-tag-row">
                <span className="label">Probabilistic Forecast Engine</span>
                <span className="version-pill">Fan Charts & Lead-time Simulation</span>
              </div>
              <h1 className="page-title">
                Explore <span className="serif-italic">uncertainty</span> with rigorous calibration
              </h1>
              <p className="page-subtitle">
                Inspect past actuals, zero-demand stockout periods, and probabilistic quantile fan charts (P10, P25, P50, P75, P90)
                under rolling-origin evaluation.
              </p>
            </div>

            <div className="studio-grid">
              {/* Left Configuration Rail */}
              <div className="studio-sidebar">
                {/* Demand Scenario Selector */}
                <div className="card">
                  <div className="card-head">
                    <h2 className="card-title">
                      <FiSliders /> Demand Scenario
                    </h2>
                    <span className="label">SKU Type</span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {DEMAND_SCENARIOS.map((s) => (
                      <div
                        key={s.id}
                        className={`scenario-card-item ${s.id === selectedScenarioId ? 'selected' : ''}`}
                        onClick={() => setSelectedScenarioId(s.id)}
                      >
                        <div style={{ fontWeight: 600, fontSize: '13px' }}>{s.name}</div>
                        <div className="scenario-meta-row">
                          <span className="label">{s.dataset}</span>
                          <span
                            style={{
                              fontSize: '11px',
                              fontWeight: 700,
                              color:
                                s.category === 'smooth'
                                  ? '#15803d'
                                  : s.category === 'intermittent'
                                  ? '#b45309'
                                  : s.category === 'lumpy'
                                  ? '#b91c1c'
                                  : '#7c3aed',
                            }}
                          >
                            {s.category.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Model Selector */}
                <div className="card">
                  <div className="card-head">
                    <h2 className="card-title">
                      <FiCpu /> Forecaster Model
                    </h2>
                    <span className="label">Family</span>
                  </div>

                  <div className="model-pill-group">
                    {MODEL_SPECS.map((m) => (
                      <button
                        key={m.id}
                        className={`model-pill-btn ${m.id === selectedModel ? 'active' : ''}`}
                        onClick={() => setSelectedModel(m.id)}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ color: m.id === selectedModel ? 'var(--accent)' : 'inherit' }}>
                            {m.icon}
                          </span>
                          <span className="model-pill-name">{m.name.split(' ')[0]}</span>
                        </div>
                        <span className="model-pill-family">{m.family.split(' ')[0]}</span>
                      </button>
                    ))}
                  </div>

                  <p style={{ marginTop: '14px', fontSize: '13px', color: 'var(--muted)', lineHeight: 1.5 }}>
                    {modelInfo.desc}
                  </p>
                </div>

                {/* Horizon & Mask Controls */}
                <div className="card">
                  <div className="card-head">
                    <h2 className="card-title">
                      <FiCalendar /> Forecast Horizon
                    </h2>
                    <span className="mono">{horizon} Days</span>
                  </div>

                  <div className="form-group">
                    <input
                      type="range"
                      min={7}
                      max={56}
                      step={7}
                      value={horizon}
                      onChange={(e) => setHorizon(Number(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent)' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--muted)', marginTop: '4px' }}>
                      <span>7d (1 wk)</span>
                      <span>14d (2 wk)</span>
                      <span>28d (M5)</span>
                      <span>56d (8 wk)</span>
                    </div>
                  </div>

                  <div className="toggle-switch-row" style={{ marginTop: '16px' }}>
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: 600 }}>Censorship Masking</div>
                      <div style={{ fontSize: '11px', color: 'var(--muted)' }}>Zero out loss on stockouts</div>
                    </div>
                    <input
                      type="checkbox"
                      className="toggle-checkbox"
                      checked={maskStockouts}
                      onChange={(e) => setMaskStockouts(e.target.checked)}
                    />
                  </div>
                </div>
              </div>

              {/* Right Chart & Fan Plot Canvas */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="chart-container-card">
                  {/* Chart Toolbar */}
                  <div className="chart-controls-toolbar">
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '16px' }}>
                        {scenario.name}
                      </div>
                      <div style={{ fontSize: '13px', color: 'var(--muted)' }}>
                        Evaluating model: <strong>{modelInfo.name}</strong> over {horizon}-day rolling window
                      </div>
                    </div>

                    <div className="chart-legend">
                      <div className="legend-item">
                        <span className="legend-color-box" style={{ background: '#111111' }} />
                        <span>Actuals</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-color-box" style={{ background: 'var(--accent)' }} />
                        <span>Median (P50)</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-color-box" style={{ background: 'rgba(255, 90, 31, 0.28)' }} />
                        <span>50% CI (P25-P75)</span>
                      </div>
                      <div className="legend-item">
                        <span className="legend-color-box" style={{ background: 'rgba(255, 90, 31, 0.12)' }} />
                        <span>90% CI (P10-P90)</span>
                      </div>
                    </div>
                  </div>

                  {/* SVG Chart */}
                  <div className="chart-svg-wrapper">
                    <svg
                      className="fan-svg"
                      viewBox={`0 0 ${svgMetrics.W} ${svgMetrics.H}`}
                      preserveAspectRatio="none"
                      onMouseLeave={() => setHoveredIdx(null)}
                    >
                      <defs>
                        <linearGradient id="band90Grad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#ff5a1f" stopOpacity="0.18" />
                          <stop offset="100%" stopColor="#ff5a1f" stopOpacity="0.08" />
                        </linearGradient>
                        <linearGradient id="band50Grad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#ff5a1f" stopOpacity="0.35" />
                          <stop offset="100%" stopColor="#ff5a1f" stopOpacity="0.18" />
                        </linearGradient>
                      </defs>

                      {/* Hairline Grid Lines */}
                      {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
                        const y = svgMetrics.padT + (svgMetrics.H - svgMetrics.padT - svgMetrics.padB) * pct;
                        const labelVal = Math.round(svgMetrics.maxVal * (1 - pct));
                        return (
                          <g key={pct}>
                            <line
                              x1={svgMetrics.padL}
                              y1={y}
                              x2={svgMetrics.W - svgMetrics.padR}
                              y2={y}
                              stroke="#f0f0f0"
                              strokeWidth={1}
                            />
                            <text
                              x={svgMetrics.padL - 8}
                              y={y + 4}
                              textAnchor="end"
                              fontSize="11"
                              fill="#9a9a9a"
                              fontFamily="var(--mono)"
                            >
                              {labelVal}
                            </text>
                          </g>
                        );
                      })}

                      {/* Forecast Origin Split Line */}
                      <line
                        x1={svgMetrics.getX(svgMetrics.pastLen)}
                        y1={svgMetrics.padT}
                        x2={svgMetrics.getX(svgMetrics.pastLen)}
                        y2={svgMetrics.H - svgMetrics.padB}
                        stroke="#111111"
                        strokeDasharray="4 4"
                        strokeWidth={1.5}
                      />
                      <text
                        x={svgMetrics.getX(svgMetrics.pastLen) + 6}
                        y={svgMetrics.padT + 12}
                        fontSize="11"
                        fill="#111111"
                        fontWeight="600"
                        fontFamily="var(--mono)"
                      >
                        ORIGIN T
                      </text>

                      {/* 90% Quantile Band (P10 - P90) */}
                      <polygon points={svgMetrics.band90} fill="url(#band90Grad)" />

                      {/* 50% Quantile Band (P25 - P75) */}
                      <polygon points={svgMetrics.band50} fill="url(#band50Grad)" />

                      {/* Historical Actuals Path */}
                      <path
                        d={svgMetrics.historyPath}
                        fill="none"
                        stroke="#111111"
                        strokeWidth={2}
                      />

                      {/* Future Realized Actuals */}
                      <path
                        d={svgMetrics.actualsPath}
                        fill="none"
                        stroke="#737373"
                        strokeDasharray="3 3"
                        strokeWidth={1.5}
                      />

                      {/* Median Prediction (P50) Path */}
                      <path
                        d={svgMetrics.p50Path}
                        fill="none"
                        stroke="#ff5a1f"
                        strokeWidth={2.5}
                      />

                      {/* Interactive Hover Vertical Crosshair */}
                      {hoveredIdx !== null && (
                        <g>
                          <line
                            x1={svgMetrics.getX(hoveredIdx)}
                            y1={svgMetrics.padT}
                            x2={svgMetrics.getX(hoveredIdx)}
                            y2={svgMetrics.H - svgMetrics.padB}
                            stroke="#ff5a1f"
                            strokeWidth={1.5}
                            strokeDasharray="2 2"
                          />
                        </g>
                      )}

                      {/* Transparent Hover Hit Areas */}
                      {Array.from({ length: svgMetrics.totalPoints }).map((_, idx) => (
                        <rect
                          key={idx}
                          x={svgMetrics.getX(idx) - 8}
                          y={svgMetrics.padT}
                          width={16}
                          height={svgMetrics.H - svgMetrics.padT - svgMetrics.padB}
                          fill="transparent"
                          cursor="crosshair"
                          onMouseEnter={() => setHoveredIdx(idx)}
                        />
                      ))}
                    </svg>

                    {/* Tooltip Popover */}
                    {hoveredIdx !== null && (
                      <div
                        className="chart-tooltip-popover"
                        style={{
                          left: `${(svgMetrics.getX(hoveredIdx) / svgMetrics.W) * 100}%`,
                          top: `${(svgMetrics.getY(15) / svgMetrics.H) * 100}%`,
                        }}
                      >
                        {hoveredIdx < svgMetrics.pastLen ? (
                          <>
                            <div className="tooltip-date">
                              Past: {forecastData.historyDates[hoveredIdx]}
                            </div>
                            <div className="tooltip-row">
                              <span>Actual Sales:</span>
                              <strong>{forecastData.historyValues[hoveredIdx]}</strong>
                            </div>
                            {forecastData.stockoutFlags[hoveredIdx] && (
                              <div style={{ color: '#b91c1c', fontSize: '11px', fontWeight: 600 }}>
                                ⚠ Stockout Censored
                              </div>
                            )}
                          </>
                        ) : (
                          <>
                            <div className="tooltip-date">
                              Forecast: {forecastData.futureDates[hoveredIdx - svgMetrics.pastLen]}
                            </div>
                            <div className="tooltip-row">
                              <span>P90 (Upper):</span>
                              <strong>{forecastData.p90[hoveredIdx - svgMetrics.pastLen]}</strong>
                            </div>
                            <div className="tooltip-row">
                              <span style={{ color: 'var(--accent)' }}>P50 (Median):</span>
                              <strong style={{ color: 'var(--accent)' }}>{forecastData.p50[hoveredIdx - svgMetrics.pastLen]}</strong>
                            </div>
                            <div className="tooltip-row">
                              <span>P10 (Lower):</span>
                              <strong>{forecastData.p10[hoveredIdx - svgMetrics.pastLen]}</strong>
                            </div>
                            <div className="tooltip-row" style={{ borderTop: '1px solid #eee', paddingTop: '4px' }}>
                              <span>Ground Truth:</span>
                              <strong>{forecastData.actuals[hoveredIdx - svgMetrics.pastLen]}</strong>
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Scorecards Row */}
                  <div className="metrics-score-grid">
                    <div className="metric-score-card">
                      <span className="label">CRPS Score</span>
                      <div className="metric-score-val">{modelInfo.crps}</div>
                      <div style={{ fontSize: '11px', color: '#15803d', marginTop: '2px' }}>
                        -14.2% vs Naive
                      </div>
                    </div>

                    <div className="metric-score-card">
                      <span className="label">Pinball Loss</span>
                      <div className="metric-score-val">{modelInfo.pinball}</div>
                      <div style={{ fontSize: '11px', color: '#15803d', marginTop: '2px' }}>
                        Calibrated across 7 q
                      </div>
                    </div>

                    <div className="metric-score-card">
                      <span className="label">MASE Ratio</span>
                      <div className="metric-score-val">{modelInfo.mase}</div>
                      <div style={{ fontSize: '11px', color: '#15803d', marginTop: '2px' }}>
                        Scales by in-sample season
                      </div>
                    </div>

                    <div className="metric-score-card">
                      <span className="label">90% Coverage</span>
                      <div className="metric-score-val">{modelInfo.coverage90}%</div>
                      <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '2px' }}>
                        Target: 90.0%
                      </div>
                    </div>

                    <div className="metric-score-card">
                      <span className="label">Inference Latency</span>
                      <div className="metric-score-val">{modelInfo.latencyMs}ms</div>
                      <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '2px' }}>
                        per SKU horizon
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model Architecture Deep-Dive */}
                <div className="card">
                  <div className="card-head">
                    <h2 className="card-title">
                      <FiLayers /> Model Architecture & Calibration Notes
                    </h2>
                    <span className="version-pill">No Lookahead Bias</span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                    <div style={{ background: 'var(--surface-subtle)', padding: '16px', borderRadius: '8px' }}>
                      <div style={{ fontWeight: 600, fontSize: '14px', marginBottom: '4px' }}>
                        1. Likelihood & Objectives
                      </div>
                      <div style={{ fontSize: '13px', color: 'var(--muted)', lineHeight: 1.5 }}>
                        Poisson & Tweedie objectives for GBDT; Negative Binomial for DeepAR; square-root variance stabilization for Kalman.
                      </div>
                    </div>

                    <div style={{ background: 'var(--surface-subtle)', padding: '16px', borderRadius: '8px' }}>
                      <div style={{ fontWeight: 600, fontSize: '14px', marginBottom: '4px' }}>
                        2. Causal Feature Pipeline
                      </div>
                      <div style={{ fontSize: '13px', color: 'var(--muted)', lineHeight: 1.5 }}>
                        Lags strictly indexed at <code>lag &gt;= horizon</code>. No future leaking rolling means or centered filters.
                      </div>
                    </div>

                    <div style={{ background: 'var(--surface-subtle)', padding: '16px', borderRadius: '8px' }}>
                      <div style={{ fontWeight: 600, fontSize: '14px', marginBottom: '4px' }}>
                        3. Cold-Start Handling
                      </div>
                      <div style={{ fontSize: '13px', color: 'var(--muted)', lineHeight: 1.5 }}>
                        Leading zeros are masked before first positive sale. Item-dropout trained on global embeddings ensures robust priors.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ====================================================================
            PAGE 2: BENCHMARK & FAIR LEADERBOARD
            ==================================================================== */}
        {currentPage === 'benchmark' && (
          <div>
            <div className="page-header">
              <div className="page-tag-row">
                <span className="label">Evaluation Matrix</span>
                <span className="version-pill">Fair Backtest Protocol</span>
              </div>
              <h1 className="page-title">
                Multi-Model <span className="serif-italic">Leaderboard</span> under rolling origins
              </h1>
              <p className="page-subtitle">
                Comprehensive comparison across 8 rolling origins with purged evaluation windows. Every model is scored
                using identical test targets, stockout masks, and quantile sets.
              </p>
            </div>

            {/* Filter toolbar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                {['all', '1-7d', '8-14d', '15-21d', '22-28d'].map((b) => (
                  <button
                    key={b}
                    className={`btn btn-sm ${benchmarkHorizon === b ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setBenchmarkHorizon(b)}
                  >
                    Horizon: {b.toUpperCase()}
                  </button>
                ))}
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                <span className="label" style={{ alignSelf: 'center' }}>Ranking Metric:</span>
                <span className="version-pill">CRPS (Continuous Ranked Probability Score)</span>
              </div>
            </div>

            {/* Table */}
            <div className="table-responsive">
              <table className="leaderboard-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Family</th>
                    <th>CRPS</th>
                    <th>MASE</th>
                    <th>WAPE</th>
                    <th>Pinball Loss</th>
                    <th>90% Coverage</th>
                    <th>Latency / SKU</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_SPECS.map((m, idx) => (
                    <tr key={m.id} className={idx === 2 ? 'winner-row' : ''}>
                      <td>
                        <span className={`rank-badge ${idx === 0 ? 'rank-1' : idx === 1 ? 'rank-2' : 'rank-3'}`}>
                          {idx + 1}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                          <span style={{ color: 'var(--accent)', fontSize: '16px' }}>{m.icon}</span>
                          {m.name}
                        </div>
                      </td>
                      <td>
                        <span className="mono" style={{ fontSize: '12px', color: 'var(--muted)' }}>
                          {m.family}
                        </span>
                      </td>
                      <td>
                        <strong>{m.crps}</strong>
                      </td>
                      <td>{m.mase}</td>
                      <td>{(m.wape * 100).toFixed(1)}%</td>
                      <td>{m.pinball}</td>
                      <td>
                        <span
                          style={{
                            color: Math.abs(m.coverage90 - 90) < 1.5 ? '#15803d' : '#b45309',
                            fontWeight: 600,
                          }}
                        >
                          {m.coverage90}%
                        </span>
                      </td>
                      <td>
                        <span className="mono">{m.latencyMs} ms</span>
                      </td>
                      <td>
                        {idx === 2 ? (
                          <span className="version-pill" style={{ color: '#15803d', borderColor: '#bbf7d0', background: '#f0fdf4' }}>
                            Top Hybrid
                          </span>
                        ) : idx === 5 ? (
                          <span className="version-pill" style={{ color: 'var(--accent)', borderColor: 'var(--accent-border)' }}>
                            Top Ensemble
                          </span>
                        ) : idx === 4 ? (
                          <span className="version-pill">Baseline</span>
                        ) : (
                          <span className="version-pill" style={{ color: '#15803d' }}>Validated</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Fair Evaluation Protocol Card */}
            <div className="card" style={{ marginTop: '24px' }}>
              <div className="card-head">
                <h2 className="card-title">
                  <FiShield /> 3 Principles of Fair Benchmarking in ForecastKit
                </h2>
                <span className="version-pill">Academic Standard</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
                <div>
                  <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '6px' }}>1. Identical Test Windows</h3>
                  <p style={{ fontSize: '13px', color: 'var(--muted)', lineHeight: 1.5 }}>
                    Every single forecaster is evaluated over the identical 8 rolling origins. No model is allowed to skip difficult stockout or promo periods.
                  </p>
                </div>

                <div>
                  <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '6px' }}>2. Unified Quantile Grid</h3>
                  <p style={{ fontSize: '13px', color: 'var(--muted)', lineHeight: 1.5 }}>
                    Models generate exact quantiles <code>[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]</code>. Scores are calculated via pinball integration for true comparison.
                  </p>
                </div>

                <div>
                  <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '6px' }}>3. Censored Stockout Fairness</h3>
                  <p style={{ fontSize: '13px', color: 'var(--muted)', lineHeight: 1.5 }}>
                    Days flagged by the stockout mask (availability=False or consecutive zero run under high velocity) receive zero evaluation weight so supply failures don&apos;t penalize forecast quality.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ====================================================================
            PAGE 3: DATASET EXPLORER & MASKS
            ==================================================================== */}
        {currentPage === 'datasets' && (
          <div>
            <div className="page-header">
              <div className="page-tag-row">
                <span className="label">Data Taxonomy</span>
                <span className="version-pill">Syntetos-Boylan Classification</span>
              </div>
              <h1 className="page-title">
                Demand <span className="serif-italic">patterns</span>, calendar features & stockout masks
              </h1>
              <p className="page-subtitle">
                Categorize SKU behavior by Average Demand Interval (ADI) and squared Coefficient of Variation (CV²).
                Inspect Walmart M5 hierarchies and dynamic known-future calendar covariates.
              </p>
            </div>

            {/* ADI vs CV2 Grid */}
            <div className="demand-matrix-grid">
              <div className="matrix-quadrant highlight">
                <div className="quadrant-title">
                  <span>Smooth Demand</span>
                  <span className="version-pill" style={{ color: '#15803d' }}>ADI &lt; 1.32, CV² &lt; 0.49</span>
                </div>
                <div className="quadrant-criteria">High frequency · Low volume variance</div>
                <p className="quadrant-desc">
                  Frequent, stable customer demand. Best suited for Kalman filter local state-space or LightGBM regression.
                  Examples: Dairy staples, fresh produce, household paper.
                </p>
              </div>

              <div className="matrix-quadrant">
                <div className="quadrant-title">
                  <span>Intermittent Demand</span>
                  <span className="version-pill" style={{ color: '#b45309' }}>ADI &gt;= 1.32, CV² &lt; 0.49</span>
                </div>
                <div className="quadrant-criteria">Sporadic frequency · Low volume variance</div>
                <p className="quadrant-desc">
                  Long zero-demand intervals between consistent order quantities. Handled via Croston/TSB decomposition
                  or Poisson/Negative Binomial likelihoods. Examples: Specialty OTC medicine, automotive filters.
                </p>
              </div>

              <div className="matrix-quadrant">
                <div className="quadrant-title">
                  <span>Erratic Demand</span>
                  <span className="version-pill" style={{ color: '#7c3aed' }}>ADI &lt; 1.32, CV² &gt;= 0.49</span>
                </div>
                <div className="quadrant-criteria">High frequency · High volume variance</div>
                <p className="quadrant-desc">
                  Frequent sales but extreme order volume swings driven by promotions or bulk catering.
                  Benefited by Tweedie distribution modeling. Examples: Bakery croissants, party drinks.
                </p>
              </div>

              <div className="matrix-quadrant">
                <div className="quadrant-title">
                  <span>Lumpy Demand</span>
                  <span className="version-pill" style={{ color: '#b91c1c' }}>ADI &gt;= 1.32, CV² &gt;= 0.49</span>
                </div>
                <div className="quadrant-criteria">Sporadic frequency · High volume variance</div>
                <p className="quadrant-desc">
                  The most challenging regime: rare purchase events with highly volatile quantities.
                  Requires DeepAR global LSTM networks or zero-inflated negative binomial. Examples: Heavy equipment parts.
                </p>
              </div>
            </div>

            {/* Calendar Covariates Card */}
            <div className="card" style={{ marginTop: '24px' }}>
              <div className="card-head">
                <h2 className="card-title">
                  <FiCalendar /> Known-Future Calendar Covariates (Strict Causal Availability)
                </h2>
                <span className="version-pill">Always Known at Origin T</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                <div style={{ background: 'var(--surface-subtle)', padding: '14px', borderRadius: '8px' }}>
                  <div className="label">Temporal Waves</div>
                  <div style={{ fontWeight: 600, marginTop: '4px' }}>Day of Week & Cycle</div>
                  <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '4px' }}>
                    Sin/Cos harmonic cyclical encodings of month and week, plus 7-dimensional one-hot vectors.
                  </div>
                </div>

                <div style={{ background: 'var(--surface-subtle)', padding: '14px', borderRadius: '8px' }}>
                  <div className="label">SNAP Food Stamps</div>
                  <div style={{ fontWeight: 600, marginTop: '4px' }}>State-Wise Benefit Days</div>
                  <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '4px' }}>
                    Separate SNAP California, Texas, and Wisconsin distribution day schedules (days 1-10 of month).
                  </div>
                </div>

                <div style={{ background: 'var(--surface-subtle)', padding: '14px', borderRadius: '8px' }}>
                  <div className="label">Event Types</div>
                  <div style={{ fontWeight: 600, marginTop: '4px' }}>4 Event Categories</div>
                  <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '4px' }}>
                    Cultural (Easter, Valentine), Sporting (SuperBowl), National (Memorial Day), Religious (Ramadan).
                  </div>
                </div>

                <div style={{ background: 'var(--surface-subtle)', padding: '14px', borderRadius: '8px' }}>
                  <div className="label">Price & Promo</div>
                  <div style={{ fontWeight: 600, marginTop: '4px' }}>Relative Price Discount</div>
                  <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '4px' }}>
                    Log price deviation from 52-week trailing mean; flag for promotional price reductions &gt;5%.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ====================================================================
            PAGE 4: LEAK AUDITOR
            ==================================================================== */}
        {currentPage === 'auditor' && (
          <div>
            <div className="page-header">
              <div className="page-tag-row">
                <span className="label">Integrity & Correctness</span>
                <span className="version-pill">Automated Scramble Tests</span>
              </div>
              <h1 className="page-title">
                Zero-Leakage <span className="serif-italic">Architecture</span> Guarantee
              </h1>
              <p className="page-subtitle">
                Strict chronological isolation ensures that models can NEVER access future targets during training,
                feature transformation, or state filtering.
              </p>
            </div>

            {/* Visual Timeline Walk-Forward */}
            <div className="card">
              <div className="card-head">
                <h2 className="card-title">
                  <FiActivity /> Rolling Origins Walk-Forward Protocol (M5 Standard)
                </h2>
                <span className="version-pill">8 Non-Overlapping Windows</span>
              </div>

              <div className="timeline-origin-track">
                {[
                  { id: 'Origin 1', start: 'Day 1800', horizon: 'Day 1828', status: 'Passed' },
                  { id: 'Origin 2', start: 'Day 1828', horizon: 'Day 1856', status: 'Passed' },
                  { id: 'Origin 3', start: 'Day 1856', horizon: 'Day 1884', status: 'Passed' },
                  { id: 'Origin 4', start: 'Day 1884', horizon: 'Day 1912', status: 'Passed' },
                ].map((row, i) => (
                  <div key={row.id} className="origin-row">
                    <span className="mono" style={{ fontWeight: 600 }}>{row.id}</span>
                    <div className="origin-bar-container">
                      <div className="origin-segment-train" style={{ width: `${60 + i * 5}%` }} title="Training Window (Historical)" />
                      <div className="origin-segment-embargo" style={{ width: '4%' }} title="Purge/Embargo Buffer" />
                      <div className="origin-segment-test" style={{ width: '15%' }} title="Test Horizon Window (Strictly Masked)" />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px' }}>
                      <FiCheckCircle style={{ color: '#15803d' }} />
                      <span className="mono" style={{ fontSize: '12px', color: '#15803d' }}>Leak-Free</span>
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: 'var(--muted)', marginTop: '8px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '12px', height: '12px', background: '#3b82f6', borderRadius: '2px' }} />
                  Training History (≤ Origin T)
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '12px', height: '12px', background: '#f59e0b', borderRadius: '2px' }} />
                  Embargo Buffer
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '12px', height: '12px', background: '#ff5a1f', borderRadius: '2px' }} />
                  Test Evaluation Horizon (T+1 .. T+H)
                </span>
              </div>
            </div>

            {/* Interactive Leak Verifier */}
            <div className="card" style={{ marginTop: '24px' }}>
              <div className="card-head">
                <h2 className="card-title">
                  <FiShield /> Live Scramble Verification Test
                </h2>
                <button
                  className="btn btn-primary btn-sm"
                  onClick={runLeakVerification}
                  disabled={leakTestStatus === 'running'}
                >
                  {leakTestStatus === 'running' ? 'Simulating Scramble...' : 'Run Scramble Invariant Test'}
                </button>
              </div>

              <p style={{ fontSize: '14px', color: 'var(--muted)', marginBottom: '16px' }}>
                In this test, targets in the test window (T+1 .. T+H) are randomized with arbitrary noise, and the pipeline is re-run.
                If any feature or context tensor changes by even 1e-7, the test fails immediately.
              </p>

              {leakTestStatus === 'passed' && (
                <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', padding: '14px', borderRadius: '8px', color: '#15803d', fontSize: '13px' }}>
                  <strong>✓ Test Assertion Passed (Max Absolute Delta: 0.0000000)</strong>
                  <div style={{ marginTop: '4px' }}>
                    Context tensor, lag tables, and Kalman state at origin T are strictly invariant to future target scrambling.
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ====================================================================
            PAGE 5: API & DEPLOYMENT
            ==================================================================== */}
        {currentPage === 'deploy' && (
          <div>
            <div className="page-header">
              <div className="page-tag-row">
                <span className="label">Production Runtime</span>
                <span className="version-pill">FastAPI & Hugging Face Spaces</span>
              </div>
              <h1 className="page-title">
                Deploy & <span className="serif-italic">integrate</span> anywhere in 1 command
              </h1>
              <p className="page-subtitle">
                Serve real-time probabilistic forecasts via REST API, launch the Streamlit studio on Hugging Face Spaces,
                or orchestrate large-scale backtests with the Python CLI.
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
              {/* FastAPI Interactive Tester */}
              <div className="card">
                <div className="card-head">
                  <h2 className="card-title">
                    <FiTerminal /> Live REST API Endpoints
                  </h2>
                  <span className="version-pill">FastAPI</span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {[
                    { path: '/api/health', method: 'GET', desc: 'Cluster and models health check' },
                    { path: '/api/datasets', method: 'GET', desc: 'List active dataset adapters' },
                    { path: '/api/predict', method: 'POST', desc: 'Generate multi-quantile forecasts' },
                    { path: '/api/benchmark', method: 'GET', desc: 'Query leaderboard evaluations' },
                  ].map((ep) => (
                    <div
                      key={ep.path}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '10px 14px',
                        background: 'var(--surface-subtle)',
                        border: '1px solid var(--line)',
                        borderRadius: '6px',
                      }}
                    >
                      <div>
                        <span className="mono" style={{ fontWeight: 700, fontSize: '13px', marginRight: '8px' }}>
                          {ep.method}
                        </span>
                        <code style={{ fontSize: '13px' }}>{ep.path}</code>
                        <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '2px' }}>
                          {ep.desc}
                        </div>
                      </div>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => handleTestApi(ep.path)}
                      >
                        Try
                      </button>
                    </div>
                  ))}
                </div>

                {/* API Output */}
                {apiLoading ? (
                  <div style={{ marginTop: '16px', fontSize: '13px', color: 'var(--muted)' }}>
                    Sending request to local backend...
                  </div>
                ) : apiResponse ? (
                  <div style={{ marginTop: '16px' }}>
                    <div className="label" style={{ marginBottom: '6px' }}>Response Payload</div>
                    <pre className="code-panel" style={{ maxHeight: '220px' }}>
                      {apiResponse}
                    </pre>
                  </div>
                ) : null}
              </div>

              {/* Deployment & CLI Snippets */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="card">
                  <div className="card-head">
                    <h2 className="card-title">
                      <FiZap /> 1-Click Hugging Face Spaces
                    </h2>
                    <span className="version-pill">Streamlit Space</span>
                  </div>

                  <p style={{ fontSize: '13px', color: 'var(--muted)', marginBottom: '12px' }}>
                    Deploy ForecastKit to a public or private Hugging Face Space using the pre-configured bundle:
                  </p>

                  <div className="code-panel">
                    <button
                      className="code-copy-btn"
                      onClick={() =>
                        copyToClipboard(
                          'cd hf_spaces && bash deploy.sh your-hf-username forecastkit-studio'
                        )
                      }
                    >
                      <FiCopy /> {copiedCode ? 'Copied!' : 'Copy'}
                    </button>
                    <code>
                      # Deploy to Hugging Face Spaces<br />
                      cd hf_spaces<br />
                      bash deploy.sh &lt;hf-username&gt; forecastkit-studio
                    </code>
                  </div>
                </div>

                <div className="card">
                  <div className="card-head">
                    <h2 className="card-title">
                      <FiCpu /> Python CLI & Backtest Runner
                    </h2>
                    <span className="version-pill">Terminal</span>
                  </div>

                  <div className="code-panel">
                    <code>
                      # Run backtest with YAML config<br />
                      forecastkit backtest configs/m5.yaml --out-dir runs/m5<br /><br />
                      # Generate fair evaluation scorecard<br />
                      forecastkit evaluate runs/m5<br /><br />
                      # Start REST API server on port 8000<br />
                      forecastkit api --host 0.0.0.0 --port 8000
                    </code>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Command Palette Modal Dialog */}
      {cmdOpen && (
        <dialog open className="cmd-dialog">
          <div className="cmd-input-row">
            <FiSearch style={{ color: 'var(--muted)', fontSize: '18px' }} />
            <input
              type="text"
              className="cmd-input"
              placeholder="Search views, models, datasets... (Esc to close)"
              value={cmdSearch}
              onChange={(e) => setCmdSearch(e.target.value)}
              autoFocus
            />
          </div>

          <ul className="cmd-list">
            {filteredCommands.map((cmd) => (
              <li
                key={cmd.id}
                className="cmd-item"
                onClick={() => {
                  if (cmd.page) setCurrentPage(cmd.page);
                  if (cmd.model) {
                    setSelectedModel(cmd.model);
                    setCurrentPage('studio');
                  }
                  setCmdOpen(false);
                }}
              >
                <div className="cmd-item-left">
                  <FiArrowRight style={{ color: 'var(--accent)' }} />
                  <div>
                    <div style={{ fontWeight: 600 }}>{cmd.label}</div>
                    <div style={{ fontSize: '12px', color: 'var(--muted)' }}>{cmd.sub}</div>
                  </div>
                </div>
                <kbd style={{ fontSize: '10px' }}>Enter</kbd>
              </li>
            ))}
          </ul>
        </dialog>
      )}

      {/* Hatched Accent Band across the bottom */}
      <div className="hatched-band" />

      {/* Footer matching DikshantJangra aesthetic */}
      <footer className="site-footer">
        <div className="footer-inner">
          <div className="footer-brand-col">
            <div className="brand-section" style={{ marginBottom: '12px' }}>
              <div className="brand-logo-mark">
                <TbChartCandle />
              </div>
              <div className="brand-name">
                ForecastKit
                <span className="version-pill">v0.1.0</span>
              </div>
            </div>
            <p style={{ fontSize: '14px', color: 'var(--muted)', lineHeight: 1.6 }}>
              Model-agnostic probabilistic demand-forecasting toolkit. Fair backtesting, stockout censorship handling,
              and leak-free validation for enterprise supply chain and retail.
            </p>
          </div>

          <div className="footer-cols">
            <div>
              <div className="footer-col-title">Navigation</div>
              <ul className="footer-col-links">
                <li><a onClick={() => setCurrentPage('studio')} style={{ cursor: 'pointer' }}>Studio & Fan Charts</a></li>
                <li><a onClick={() => setCurrentPage('benchmark')} style={{ cursor: 'pointer' }}>Fair Leaderboard</a></li>
                <li><a onClick={() => setCurrentPage('datasets')} style={{ cursor: 'pointer' }}>Datasets & Slicing</a></li>
                <li><a onClick={() => setCurrentPage('auditor')} style={{ cursor: 'pointer' }}>Leak Auditor</a></li>
                <li><a onClick={() => setCurrentPage('deploy')} style={{ cursor: 'pointer' }}>API & Hugging Face</a></li>
              </ul>
            </div>

            <div>
              <div className="footer-col-title">Models</div>
              <ul className="footer-col-links">
                <li><span className="mono">Kalman State Space</span></li>
                <li><span className="mono">LightGBM Multi-Horizon</span></li>
                <li><span className="mono">Kalman + LGBM Hybrid</span></li>
                <li><span className="mono">DeepAR LSTM (PyTorch)</span></li>
                <li><span className="mono">Quantile Ensemble</span></li>
              </ul>
            </div>

            <div>
              <div className="footer-col-title">Protocols</div>
              <ul className="footer-col-links">
                <li><span className="mono">8 Rolling Origins</span></li>
                <li><span className="mono">Purge 7d / Embargo 3d</span></li>
                <li><span className="mono">Syntetos-Boylan ADI/CV²</span></li>
                <li><span className="mono">Pinball / CRPS Scoring</span></li>
                <li><span className="mono">Zero-Lookahead CI</span></li>
              </ul>
            </div>
          </div>
        </div>

        <div className="footer-bottom-bar">
          <div>
            Built with modern light aesthetic · Open source under MIT License
          </div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <span className="mono">FastAPI</span>
            <span className="mono">PyTorch 2.x</span>
            <span className="mono">React 18</span>
            <span className="mono">Streamlit</span>
          </div>
        </div>
      </footer>
    </div>
  );
};
export default App;
