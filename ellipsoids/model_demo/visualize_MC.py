#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an interactive HTML demo for a 2D oddity-task Monte Carlo simulation.

The page includes:
1. A control pad for selecting covariance shape via axis ratio and rotation.
2. A scatter panel showing z_ref, z0, and z1 samples plus covariance ellipses.
3. A histogram panel showing the oddity-task decision margin.
4. Buttons for sample size and trace visibility, with immediate redraws.
"""

from __future__ import annotations
import json
from pathlib import Path
import jax
import numpy as np
from plotly.offline.offline import get_plotlyjs
from analysis.utils_load import get_path
from analysis.ellipses_tools import ellParamsQ_to_covMat
from core.oddity_task import simulate_oddity_one_trial


def compute_scatter_limits(config, heuristic_scale = 3):
    """
    Compute a single fixed viewing window for the scatter panel.

    The scatter plot is meant to feel stable while the user changes the control
    settings interactively. If we let the axis limits auto-rescale after every
    redraw, the sample cloud and covariance ellipses would visually "jump"
    around, which makes it much harder to compare conditions.
    
    Instead of adapting the axes on every frame, we estimate one panel size that
    is large enough to contain the widest expected spread of samples and
    ellipses over the entire allowed control range.

    """
    
    # Fixed reference center [x_ref, y_ref].
    ref_mean = np.array(config["ref_mean"], dtype=float)  
    
    # Smallest/largest y values reachable by x1 = [0, y].
    comp_y_min, comp_y_max = map(float, config["comp_y_range"])  
    
    # Midpoint of the allowed center locations. 
    center = np.array([ref_mean[0], 0.5 * (comp_y_min + comp_y_max)], dtype=float)  
    
    # Distance from that midpoint to either end of the slider range.
    max_center_offset = 0.5 * (comp_y_max - comp_y_min)  

    # Largest major/minor axis ratio the control pad allows.
    max_ratio = float(config["ratio_range"][1])  
    
    # Fixed semi-minor axis length used throughout the demo.
    minor_axis_length = float(config["minor_axis_length"])
    
    # Largest semi-major axis length.
    max_major_std = minor_axis_length * max_ratio  
    
    # Heuristic: add about 4.75 major-axis stds so the ellipse and most samples stay visible.
    radius = max_center_offset + heuristic_scale * max_major_std  

    # Symmetric horizontal and vertical limits around the chosen center.
    return {
        "x": [float(center[0] - radius), float(center[0] + radius)],  
        "y": [float(center[1] - radius), float(center[1] + radius)],  
    }


def compute_histogram_settings(config):
    """
    Estimate fixed histogram display settings across the full interactive control
    space.
    
    Overarching goal
    ----------------
    The decision-margin histogram should also remain visually stable while the
    user changes ratio, angle, or the `x1` position. If the histogram x-range,
    binning, or y-range changed from frame to frame, the panel would be hard to
    interpret because changes in bar height could reflect re-scaling rather than
    real changes in the underlying distribution.
    
    Strategy
    --------
    We therefore choose one global histogram specification that works well
    across the full set of allowed controls:
    - an x-range estimated from a coarse sweep over the control space,
    - a fixed bin width, and
    - a y-limit estimated from the same coarse sweep.
    
    To determine a reasonable y-limit, we do a coarse sweep over the control
    variables:
    - sample several axis-ratio values,
    - sample several rotation-angle values,
    - sample several allowed `x1` y-positions.
    
    For each coarse parameter combination, we:
    1. construct the corresponding covariance matrix,
    2. simulate a batch of Monte Carlo decision margins, and
    3. store those margins so we can estimate stable global histogram limits.

    After collecting those coarse samples, we:
    1. estimate a global x-range from the minimum and maximum sampled margins,
       snapping outward to the nearest clean bin edge,
    2. build the histogram bins from that x-range, and
    3. record the largest per-bin probability seen anywhere in the coarse sweep,
       then slightly inflate it to obtain a stable y-axis upper bound.

    """
    
    # load config info
    ref_mean = np.array(config["ref_mean"], dtype=float)  
    minor_axis_length = float(config["minor_axis_length"])  
    ratio_min, ratio_max = map(float, config["ratio_range"])  
    angle_min, angle_max = map(float, config["angle_range"])  
    comp_y_min, comp_y_max = map(float, config["comp_y_range"])  

    # Coarse sweep for ratios, angles and comp_y_values
    ratios = np.linspace(ratio_min, ratio_max, 7)  
    angles = np.linspace(angle_min, angle_max, 13)  
    comp_y_values = np.linspace(comp_y_min, comp_y_max, 9)  

    # Base random key so the coarse sweep is reproducible.
    base_key = jax.random.PRNGKey(int(config["seed"]))  

    # Store one sampled decision-margin array for each coarse parameter combination.
    sampled_margins = []  
    sweep_index = 0
    for ratio in ratios:
        for angle_deg in angles:
            for comp_y in comp_y_values:
                # Comparison center moves only along the y-axis.
                comp_mean = np.array([ref_mean[0], comp_y], dtype=float)  

                # Convert ratio into the semi-major axis length.
                major_axis_length = float(ratio) * minor_axis_length  

                # Build the covariance matrix for this coarse setting.
                covariance = ellParamsQ_to_covMat(
                    major_axis_length,
                    minor_axis_length,
                    float(angle_deg),
                )  

                # Use the covariance factorization expected by the core oddity simulator.
                covariance_factor = np.linalg.cholesky(covariance)  

                # Give each coarse parameter setting its own deterministic random key.
                trial_key = jax.random.fold_in(base_key, sweep_index)  
                sweep_index += 1

                sampled_margins.append(
                    np.asarray(
                        simulate_oddity_one_trial(
                            (ref_mean, comp_mean, covariance_factor, covariance_factor),
                            trial_key,
                            1000,
                            0.0,
                        )
                    )
                )

    # Fixed histogram bin width so the display remains stable across redraws.
    bin_width = 2.0  

    # Pool all coarse-sweep margins to estimate a global x-range.
    all_margins = np.concatenate(sampled_margins)  

    # Round the min max sampled margin down to the nearest bin edge.
    x_min = float(bin_width * np.floor(np.min(all_margins) / bin_width))  
    x_max = float(bin_width * np.ceil(np.max(all_margins) / bin_width))  

    # Build the shared bin edges for every histogram view.
    bin_edges = np.arange(x_min, x_max + bin_width, bin_width)  

    # Number of bins implied by those edges.
    n_bins = len(bin_edges) - 1  

    # Track the tallest per-bin probability seen anywhere in the coarse sweep.
    max_probability = 0.0  
    for margins in sampled_margins:
        # Histogram one coarse-sweep margin set using the shared bin edges.
        hist, _ = np.histogram(margins, bins=bin_edges)  

        # Convert the tallest bin count into probability and keep the worst case.
        max_probability = max(max_probability, float(hist.max() / margins.size))  

    return {
        "x_min": x_min,  # Global histogram lower bound.
        "x_max": x_max,  # Global histogram upper bound.
        "bin_width": bin_width,  # Shared histogram bin width.
        "n_bins": n_bins,  # Total number of bins.
        "y_max": float(max_probability),  # Worst-case per-bin probability.
    }


def default_config() -> dict[str, float | int | list[float]]:
    """Return the default means and covariance-derived control settings."""
    ref_mean = np.array([0, 0], dtype=float)
    default_comp_y = 4.0
    default_cov = np.array(
        [
            [0.010, 0.003],
            [0.003, 0.018],
        ],
        dtype=float,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(default_cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    default_ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[1]))
    default_angle_deg = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])) % 180.0)

    config = {
        "ref_mean": ref_mean.tolist(),
        "default_comp_y": default_comp_y,
        "comp_y_range": [0.0, 4.0],
        "minor_axis_length": 1.0,
        "default_ratio": default_ratio,
        "default_angle_deg": default_angle_deg,
        "default_sample_size": 200,
        "ellipse_scale": 2.0,
        "seed": 7,
        "ratio_range": [1.0, 3.0],
        "angle_range": [0.0, 180.0],
        "sample_size_options": [100, 200, 500, 1000, 2000],
    }
    config["scatter_limits"] = compute_scatter_limits(config)
    config["histogram"] = compute_histogram_settings(config)
    return config


def build_html(config: dict[str, float | int | list[float]]) -> str:
    """Build a self-contained interactive HTML document."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Monte Carlo simulations for approximating percent-correct performance</title>
  <style>
    :root {
      --bg: #eef0ea;
      --panel: rgba(255, 255, 255, 0.88);
      --edge: rgba(39, 53, 48, 0.14);
      --ink: #1f2b24;
      --muted: #5b695f;
      --accent: #275d38;
      --accent-soft: rgba(39, 93, 56, 0.12);
      --gray-soft: #d8d8d8;
      --gray-deep: #6b6b6b;
      --control-pad-left: 36px;
      --control-pad-right: 22px;
      --control-pad-top: 18px;
      --control-pad-bottom: 34px;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(108, 140, 119, 0.14), transparent 26%),
        radial-gradient(circle at top right, rgba(200, 210, 198, 0.22), transparent 24%),
        linear-gradient(180deg, #f5f6f2 0%, #ecefe8 100%);
    }
    .page {
      max-width: 1560px;
      margin: 0 auto;
      padding: 28px 22px 34px 22px;
    }
    .header {
      margin-bottom: 14px;
    }
    .title {
      margin: 0;
      font-size: 32px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .layout {
      display: grid;
      grid-template-columns: 560px minmax(0, 1fr);
      gap: 18px;
      align-items: stretch;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--edge);
      border-radius: 22px;
      box-shadow: 0 18px 40px rgba(59, 67, 62, 0.08);
      height: 100%;
    }
    .controls {
      padding: 18px;
      position: sticky;
      top: 18px;
      display: flex;
      flex-direction: column;
    }
    .controls h2,
    .plots h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .controls-shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 150px;
      gap: 18px;
      margin-top: 14px;
      align-items: start;
      flex: 1 1 auto;
    }
    .controls-main,
    .controls-side {
      min-width: 0;
    }
    .controls-main .section:first-child,
    .controls-side .section:first-child {
      margin-top: 0;
    }
    .section {
      margin-top: 18px;
    }
    .section-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .button-group {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .controls-side .button-group {
      flex-direction: column;
      align-items: flex-start;
    }
    button {
      border: 1px solid var(--edge);
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }
    .controls-side button {
      width: 132px;
      padding-left: 10px;
      padding-right: 10px;
      text-align: center;
    }
    button:hover {
      transform: translateY(-1px);
      border-color: rgba(39, 93, 56, 0.3);
    }
    button.active {
      background: var(--accent-soft);
      border-color: rgba(39, 93, 56, 0.38);
      color: var(--accent);
      font-weight: 700;
    }
    button.ghost-off {
      opacity: 0.58;
    }
    .readout-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1.2fr;
      gap: 10px;
      margin-top: 10px;
    }
    .readout {
      background: rgba(255, 255, 255, 0.58);
      border: 1px solid rgba(39, 53, 48, 0.08);
      border-radius: 14px;
      padding: 10px 12px;
    }
    .readout-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .readout-value {
      margin-top: 4px;
      font-size: 18px;
      font-weight: 700;
    }
    .readout-value-cov {
      font-size: 12px;
      line-height: 1.35;
      font-weight: 600;
      word-break: break-word;
      white-space: normal;
    }
    .slider-wrap {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .slider-input {
      width: 100%;
      accent-color: var(--accent);
    }
    .slider-scale {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      padding: 0 2px;
    }
    .control-pad-wrap {
      margin-top: 10px;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(39, 53, 48, 0.12);
      background: linear-gradient(180deg, rgba(39, 93, 56, 0.06), rgba(39, 93, 56, 0.01));
      padding: 10px;
    }
    #control-pad {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 14px;
      background: #ffffff;
      cursor: crosshair;
    }
    .control-pad-layout {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .control-pad-square-row {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      align-items: start;
    }
    .control-y-axis {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      height: calc(320px - var(--control-pad-top) - var(--control-pad-bottom));
      margin-top: var(--control-pad-top);
      margin-bottom: var(--control-pad-bottom);
      padding: 0;
    }
    .control-y-axis-title {
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      letter-spacing: 0.04em;
      text-transform: lowercase;
    }
    .control-axis {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
      margin-left: var(--control-pad-left);
      margin-right: var(--control-pad-right);
      padding: 0;
    }
    .control-axis-title {
      margin-top: 8px;
      text-align: center;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.04em;
    }
    .plots {
      padding: 18px;
      display: flex;
      flex-direction: column;
    }
    .plot-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(460px, 1fr);
      gap: 16px;
      margin-top: 14px;
      flex: 1 1 auto;
    }
    .plot-panel {
      border: 1px solid rgba(39, 53, 48, 0.10);
      border-radius: 18px;
      overflow: hidden;
      background: #ffffff;
    }
    .plot-head {
      padding: 14px 16px 0 16px;
    }
    .plot-head h3 {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
    }
    .plot {
      width: 100%;
      height: 620px;
    }
    @media (max-width: 1180px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .controls {
        position: static;
      }
      .controls-shell {
        grid-template-columns: 1fr;
      }
      .readout-grid {
        grid-template-columns: 1fr 1fr;
      }
      .plot-grid {
        grid-template-columns: 1fr;
      }
      .plot {
        height: 520px;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1 class="title">Monte Carlo simulations for approximating percent-correct performance</h1>
    </div>

    <div class="layout">
      <aside class="card controls">
        <h2>Control Panel</h2>

        <div class="controls-shell">
          <div class="controls-main">
            <div class="section">
              <div class="section-label">Covariance Generator</div>
              <div class="control-pad-wrap">
                <div class="control-pad-layout">
                  <div class="control-pad-square-row">
                    <div class="control-y-axis">
                      <span>180°</span>
                      <div class="control-y-axis-title">angle</div>
                      <span>0°</span>
                    </div>
                    <canvas id="control-pad" width="320" height="320" aria-label="Ratio and angle control pad"></canvas>
                  </div>
                  <div>
                    <div class="control-axis">
                      <span>1.0</span>
                      <span>3.0</span>
                    </div>
                    <div class="control-axis-title">ratio</div>
                  </div>
                </div>
              </div>

              <div class="readout-grid">
                <div class="readout">
                  <div class="readout-label">Axis Ratio</div>
                  <div class="readout-value" id="ratio-readout">2.00</div>
                </div>
                <div class="readout">
                  <div class="readout-label">Angle</div>
                  <div class="readout-value" id="angle-readout">0°</div>
                </div>
                <div class="readout">
                  <div class="readout-label">Cov</div>
                  <div class="readout-value readout-value-cov" id="cov-readout">[ ]</div>
                </div>
              </div>
            </div>

            <div class="section">
              <div class="section-label">Distance between x_ref and x_1</div>
              <div class="slider-wrap">
                <input id="comp-y-slider" class="slider-input" type="range" min="0" max="4" step="0.01" value="4" />
                <div class="slider-scale">
                  <span>0.0</span>
                  <span>1.0</span>
                  <span>2.0</span>
                  <span>3.0</span>
                  <span>4.0</span>
                </div>
              </div>
            </div>
          </div>

          <div class="controls-side">
            <div class="section">
              <div class="section-label">Sample Size</div>
              <div class="button-group" id="sample-size-buttons"></div>
            </div>

            <div class="section">
              <div class="section-label">Scatter Visibility</div>
              <div class="button-group">
                <button id="toggle-zref">Hide z_ref</button>
                <button id="toggle-z0">Hide z_0</button>
                <button id="toggle-z1">Hide z_1</button>
              </div>
            </div>

            <div class="section">
              <div class="section-label">Sampling</div>
              <div class="button-group">
                <button id="resample-button" class="active">Resample</button>
              </div>
            </div>

            <div class="section">
              <div class="section-label">Export PNG</div>
              <div class="button-group">
                <button id="save-scatter-png">Save left panel</button>
                <button id="save-hist-png">Save right panel</button>
              </div>
            </div>
          </div>
        </div>
      </aside>

      <main class="card plots">
        <h2>Live View</h2>

        <div class="plot-grid">
          <section class="plot-panel">
            <div class="plot-head">
              <h3>Simulated internal measurements</h3>
            </div>
            <div id="scatter-plot" class="plot"></div>
          </section>

          <section class="plot-panel">
            <div class="plot-head">
              <h3>Decision Margin</h3>
            </div>
            <div id="histogram-plot" class="plot"></div>
          </section>
        </div>
      </main>
    </div>
  </div>

  <script>__PLOTLY_JS__</script>
  <script>
    const CONFIG = __CONFIG_JSON__;

    const state = {
      ratio: CONFIG.default_ratio,
      angleDeg: CONFIG.default_angle_deg,
      compY: CONFIG.default_comp_y,
      sampleSize: CONFIG.default_sample_size,
      resampleNonce: 0,
      visibility: {
        z_ref: true,
        z0: true,
        z1: true,
      },
      currentData: null,
      dragActive: false,
      animationFrame: null,
    };

    const COLORS = {
      refEllipse: "rgba(140, 140, 140, 0.26)",
      refLine: "rgba(110, 110, 110, 0.55)",
      compEllipse: "rgba(34, 139, 34, 0.18)",
      compLine: "rgba(34, 139, 34, 0.48)",
      zRef: "#b3b3b3",
      z0: "dimgray",
      z1: "forestgreen",
      centerRef: "#404040",
      centerComp: "forestgreen",
      correct: "rgba(34, 139, 34, 0.70)",
      incorrect: "rgba(120, 120, 120, 0.65)",
    };

    const controlPad = document.getElementById("control-pad");
    const controlCtx = controlPad.getContext("2d");
    const ratioReadout = document.getElementById("ratio-readout");
    const angleReadout = document.getElementById("angle-readout");
    const covReadout = document.getElementById("cov-readout");
    const compYSlider = document.getElementById("comp-y-slider");
    const scatterPlot = document.getElementById("scatter-plot");
    const histogramPlot = document.getElementById("histogram-plot");
    const sampleSizeButtons = document.getElementById("sample-size-buttons");
    const saveScatterButton = document.getElementById("save-scatter-png");
    const saveHistButton = document.getElementById("save-hist-png");
    const toggleButtons = {
      z_ref: document.getElementById("toggle-zref"),
      z0: document.getElementById("toggle-z0"),
      z1: document.getElementById("toggle-z1"),
    };
    const resampleButton = document.getElementById("resample-button");

    const CONTROL_BOUNDS = {
      left: 36,
      right: controlPad.width - 22,
      top: 18,
      bottom: controlPad.height - 34,
    };

    function clamp(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function ratioToCanvasX(ratio) {
      const t = (ratio - CONFIG.ratio_range[0]) / (CONFIG.ratio_range[1] - CONFIG.ratio_range[0]);
      return lerp(CONTROL_BOUNDS.left, CONTROL_BOUNDS.right, t);
    }

    function angleToCanvasY(angleDeg) {
      const t = (angleDeg - CONFIG.angle_range[0]) / (CONFIG.angle_range[1] - CONFIG.angle_range[0]);
      return lerp(CONTROL_BOUNDS.bottom, CONTROL_BOUNDS.top, t);
    }

    function canvasXToRatio(x) {
      const t = (x - CONTROL_BOUNDS.left) / (CONTROL_BOUNDS.right - CONTROL_BOUNDS.left);
      return clamp(lerp(CONFIG.ratio_range[0], CONFIG.ratio_range[1], t), CONFIG.ratio_range[0], CONFIG.ratio_range[1]);
    }

    function canvasYToAngle(y) {
      const t = (CONTROL_BOUNDS.bottom - y) / (CONTROL_BOUNDS.bottom - CONTROL_BOUNDS.top);
      return clamp(lerp(CONFIG.angle_range[0], CONFIG.angle_range[1], t), CONFIG.angle_range[0], CONFIG.angle_range[1]);
    }

    function drawControlPad() {
      const ctx = controlCtx;
      ctx.clearRect(0, 0, controlPad.width, controlPad.height);

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, controlPad.width, controlPad.height);

      ctx.save();
      ctx.strokeStyle = "rgba(39, 53, 48, 0.08)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i += 1) {
        const gx = lerp(CONTROL_BOUNDS.left, CONTROL_BOUNDS.right, i / 4);
        const gy = lerp(CONTROL_BOUNDS.top, CONTROL_BOUNDS.bottom, i / 4);
        ctx.beginPath();
        ctx.moveTo(gx, CONTROL_BOUNDS.top);
        ctx.lineTo(gx, CONTROL_BOUNDS.bottom);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(CONTROL_BOUNDS.left, gy);
        ctx.lineTo(CONTROL_BOUNDS.right, gy);
        ctx.stroke();
      }
      ctx.restore();

      ctx.strokeStyle = "rgba(39, 53, 48, 0.22)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        CONTROL_BOUNDS.left,
        CONTROL_BOUNDS.top,
        CONTROL_BOUNDS.right - CONTROL_BOUNDS.left,
        CONTROL_BOUNDS.bottom - CONTROL_BOUNDS.top
      );

      const pinX = ratioToCanvasX(state.ratio);
      const pinY = angleToCanvasY(state.angleDeg);

      ctx.beginPath();
      ctx.moveTo(pinX, CONTROL_BOUNDS.top);
      ctx.lineTo(pinX, CONTROL_BOUNDS.bottom);
      ctx.strokeStyle = "rgba(39, 93, 56, 0.16)";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(CONTROL_BOUNDS.left, pinY);
      ctx.lineTo(CONTROL_BOUNDS.right, pinY);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(pinX, pinY, 8.5, 0, Math.PI * 2);
      ctx.fillStyle = "#275d38";
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      ctx.stroke();
    }

    function updateReadouts() {
      ratioReadout.textContent = state.ratio.toFixed(2);
      angleReadout.textContent = `${state.angleDeg.toFixed(0)}°`;
    }

    function currentCompMean() {
      return [CONFIG.ref_mean[0], state.compY];
    }

    function covarianceFromControl(ratio, angleDeg) {
      const majorAxis = CONFIG.minor_axis_length * ratio;
      const minorAxis = CONFIG.minor_axis_length;
      const theta = angleDeg * Math.PI / 180.0;
      const c = Math.cos(theta);
      const s = Math.sin(theta);
      return [
        [
          majorAxis * majorAxis * c * c + minorAxis * minorAxis * s * s,
          (majorAxis * majorAxis - minorAxis * minorAxis) * c * s,
        ],
        [
          (majorAxis * majorAxis - minorAxis * minorAxis) * c * s,
          majorAxis * majorAxis * s * s + minorAxis * minorAxis * c * c,
        ],
      ];
    }

    function ellipsePoints(mean, ratio, angleDeg, scale, nPts = 181) {
      const majorStd = CONFIG.minor_axis_length * ratio;
      const minorStd = CONFIG.minor_axis_length;
      const theta = angleDeg * Math.PI / 180.0;
      const c = Math.cos(theta);
      const s = Math.sin(theta);
      const x = [];
      const y = [];

      for (let i = 0; i < nPts; i += 1) {
        const t = (i / (nPts - 1)) * Math.PI * 2.0;
        const dx = scale * majorStd * Math.cos(t);
        const dy = scale * minorStd * Math.sin(t);
        x.push(mean[0] + c * dx - s * dy);
        y.push(mean[1] + s * dx + c * dy);
      }
      return { x, y };
    }

    function mulberry32(seed) {
      let t = seed >>> 0;
      return function() {
        t += 0x6D2B79F5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
      };
    }

    function createNormalSampler(rng) {
      let spare = null;
      return function() {
        if (spare !== null) {
          const value = spare;
          spare = null;
          return value;
        }
        let u = 0;
        let v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        const mag = Math.sqrt(-2.0 * Math.log(u));
        const z0 = mag * Math.cos(2.0 * Math.PI * v);
        const z1 = mag * Math.sin(2.0 * Math.PI * v);
        spare = z1;
        return z0;
      };
    }

    function cholesky2x2(cov) {
      const a = cov[0][0];
      const b = cov[0][1];
      const c = cov[1][1];
      const l11 = Math.sqrt(a);
      const l21 = b / l11;
      const l22 = Math.sqrt(Math.max(c - l21 * l21, 1e-12));
      return [
        [l11, 0],
        [l21, l22],
      ];
    }

    function inverse2x2(m) {
      const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
      return [
        [m[1][1] / det, -m[0][1] / det],
        [-m[1][0] / det, m[0][0] / det],
      ];
    }

    function sampleGaussian(mean, chol, n, randn) {
      const points = new Array(n);
      for (let i = 0; i < n; i += 1) {
        const u0 = randn();
        const u1 = randn();
        const x = mean[0] + chol[0][0] * u0;
        const y = mean[1] + chol[1][0] * u0 + chol[1][1] * u1;
        points[i] = [x, y];
      }
      return points;
    }

    function squaredMahalanobis(aPoints, bPoints, invCov) {
      const values = new Array(aPoints.length);
      for (let i = 0; i < aPoints.length; i += 1) {
        const dx = aPoints[i][0] - bPoints[i][0];
        const dy = aPoints[i][1] - bPoints[i][1];
        const qx = invCov[0][0] * dx + invCov[0][1] * dy;
        const qy = invCov[1][0] * dx + invCov[1][1] * dy;
        values[i] = dx * qx + dy * qy;
      }
      return values;
    }

    function unpack(points) {
      return {
        x: points.map((p) => p[0]),
        y: points.map((p) => p[1]),
      };
    }

    function computeData() {
      const covariance = covarianceFromControl(state.ratio, state.angleDeg);
      const chol = cholesky2x2(covariance);
      const invCov = inverse2x2(covariance);
      const compMean = currentCompMean();
      const seed =
        CONFIG.seed +
        state.resampleNonce * 10007 +
        Math.round(state.ratio * 1000) +
        Math.round(state.angleDeg * 10) +
        Math.round(state.compY * 100) * 17 +
        state.sampleSize * 13;
      const rng = mulberry32(seed);
      const randn = createNormalSampler(rng);

      const zRef = sampleGaussian(CONFIG.ref_mean, chol, state.sampleSize, randn);
      const z0 = sampleGaussian(CONFIG.ref_mean, chol, state.sampleSize, randn);
      const z1 = sampleGaussian(compMean, chol, state.sampleSize, randn);

      const dRefRef = squaredMahalanobis(zRef, z0, invCov);
      const dRefComp = squaredMahalanobis(zRef, z1, invCov);
      const dRef2Comp = squaredMahalanobis(z0, z1, invCov);
      const dClosest = dRefComp.map((value, i) => Math.min(value, dRef2Comp[i]));
      const decisionMargin = dRefRef.map((value, i) => value - dClosest[i]);
      const pCorrect = decisionMargin.filter((value) => value < 0).length / decisionMargin.length;

      return {
        covariance,
        zRef,
        z0,
        z1,
        dRefRef,
        dRefComp,
        dRef2Comp,
        dClosest,
        decisionMargin,
        pCorrect,
        refEllipse: ellipsePoints(CONFIG.ref_mean, state.ratio, state.angleDeg, CONFIG.ellipse_scale),
        compEllipse: ellipsePoints(compMean, state.ratio, state.angleDeg, CONFIG.ellipse_scale),
        compMean,
      };
    }

    function buildScatterTraces(data) {
      const refSamples = unpack(data.zRef);
      const z0Samples = unpack(data.z0);
      const z1Samples = unpack(data.z1);

      return [
        {
          x: data.refEllipse.x,
          y: data.refEllipse.y,
          mode: "lines",
          line: { color: COLORS.refLine, width: 2 },
          fill: "toself",
          fillcolor: COLORS.refEllipse,
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          x: data.compEllipse.x,
          y: data.compEllipse.y,
          mode: "lines",
          line: { color: COLORS.compLine, width: 2 },
          fill: "toself",
          fillcolor: COLORS.compEllipse,
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          x: [CONFIG.ref_mean[0]],
          y: [CONFIG.ref_mean[1]],
          mode: "markers",
          marker: { color: COLORS.centerRef, symbol: "cross", size: 10, line: { width: 1.6 } },
          name: "ref center",
          hovertemplate: "ref center<extra></extra>",
          showlegend: false,
        },
        {
          x: [data.compMean[0]],
          y: [data.compMean[1]],
          mode: "markers",
          marker: {
            color: COLORS.centerComp,
            symbol: "cross",
            size: 10,
            line: { width: 1.6, color: COLORS.centerComp },
          },
          name: "comp center",
          hovertemplate: "comp center<extra></extra>",
          showlegend: false,
        },
        {
          x: refSamples.x,
          y: refSamples.y,
          mode: "markers",
          marker: {
            color: COLORS.zRef,
            size: 7,
            opacity: 0.72,
            line: { color: "white", width: 0.5 },
            symbol: "circle",
          },
          name: "z_ref",
          visible: state.visibility.z_ref,
          hovertemplate: "z_ref<extra></extra>",
        },
        {
          x: z0Samples.x,
          y: z0Samples.y,
          mode: "markers",
          marker: {
            color: COLORS.z0,
            size: 8,
            opacity: 0.72,
            line: { color: "white", width: 0.5 },
            symbol: "triangle-up",
          },
          name: "z_0",
          visible: state.visibility.z0,
          hovertemplate: "z_0<extra></extra>",
        },
        {
          x: z1Samples.x,
          y: z1Samples.y,
          mode: "markers",
          marker: {
            color: COLORS.z1,
            size: 7.5,
            opacity: 0.72,
            line: { color: "white", width: 0.5 },
            symbol: "square",
          },
          name: "z_1",
          visible: state.visibility.z1,
          hovertemplate: "z_1<extra></extra>",
        },
      ];
    }

    function buildScatterLayout(data) {
      return {
        margin: { l: 24, r: 24, t: 14, b: 14 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        showlegend: false,
        font: { family: "Arial, sans-serif", color: "#1f2b24", size: 14 },
        xaxis: {
          range: CONFIG.scatter_limits.x,
          visible: false,
          fixedrange: true,
          showgrid: false,
          zeroline: false,
          showticklabels: false,
        },
        yaxis: {
          range: CONFIG.scatter_limits.y,
          visible: false,
          fixedrange: true,
          showgrid: false,
          zeroline: false,
          showticklabels: false,
          scaleanchor: "x",
          scaleratio: 1,
        },
        annotations: [
          {
            text: `Percent correct = ${(100 * data.pCorrect).toFixed(1)}%`,
            xref: "paper",
            yref: "paper",
            x: 0.02,
            y: 0.98,
            showarrow: false,
            font: { size: 15, color: "#275d38" },
            align: "left",
            bgcolor: "rgba(255,255,255,0.66)",
            bordercolor: "rgba(39, 93, 56, 0.15)",
            borderpad: 6,
          },
        ],
      };
    }

    function buildHistogramTraces(data) {
      const negatives = data.decisionMargin.filter((value) => value < 0);
      const positives = data.decisionMargin.filter((value) => value >= 0);
      const totalCount = data.decisionMargin.length;
      const binSize = CONFIG.histogram.bin_width;
      return [
        {
          x: negatives,
          y: negatives.map(() => 1 / totalCount),
          type: "histogram",
          marker: { color: COLORS.correct },
          opacity: 0.90,
          histfunc: "sum",
          xbins: {
            start: CONFIG.histogram.x_min,
            end: CONFIG.histogram.x_max,
            size: binSize,
          },
          name: "Mahalanobis distance between<br>reference measurements is the shortest",
          showlegend: true,
          hovertemplate: "correct margin<extra></extra>",
        },
        {
          x: positives,
          y: positives.map(() => 1 / totalCount),
          type: "histogram",
          marker: { color: COLORS.incorrect },
          opacity: 0.88,
          histfunc: "sum",
          xbins: {
            start: CONFIG.histogram.x_min,
            end: CONFIG.histogram.x_max,
            size: binSize,
          },
          name: "otherwise",
          showlegend: false,
          hovertemplate: "incorrect margin<extra></extra>",
        },
      ];
    }

    function buildHistogramLayout(data) {
      return {
        margin: { l: 62, r: 22, t: 14, b: 70 },
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        barmode: "overlay",
        showlegend: true,
        legend: {
          orientation: "h",
          x: 0.02,
          y: 1.18,
          xanchor: "left",
          yanchor: "bottom",
          bgcolor: "rgba(255,255,255,0.92)",
          bordercolor: "rgba(39, 53, 48, 0.10)",
          borderwidth: 1,
          font: { family: "Arial, sans-serif", size: 12, color: "#1f2b24" },
        },
        font: { family: "Arial, sans-serif", color: "#1f2b24", size: 14 },
        xaxis: {
          range: [CONFIG.histogram.x_min, CONFIG.histogram.x_max],
          showticklabels: false,
          ticks: "",
          showgrid: false,
          zeroline: false,
        },
        yaxis: {
          title: { text: "probability" },
          range: [0, CONFIG.histogram.y_max],
          gridcolor: "rgba(39, 53, 48, 0.08)",
          zeroline: false,
        },
        shapes: [
          {
            type: "line",
            x0: 0,
            x1: 0,
            y0: 0,
            y1: 1,
            yref: "paper",
            line: { color: "#18211c", width: 2 },
          },
        ],
        annotations: [],
      };
    }

    function renderScatter() {
      Plotly.react(
        scatterPlot,
        buildScatterTraces(state.currentData),
        buildScatterLayout(state.currentData),
        { responsive: true, displayModeBar: false }
      );
    }

    function renderHistogram() {
      Plotly.react(
        histogramPlot,
        buildHistogramTraces(state.currentData),
        buildHistogramLayout(state.currentData),
        { responsive: true, displayModeBar: false }
      );
    }

    function updateCovReadout() {
      const covariance = state.currentData.covariance;
      covReadout.innerHTML =
        `[${covariance[0][0].toFixed(3)}, ${covariance[0][1].toFixed(3)}]<br>` +
        `[${covariance[1][0].toFixed(3)}, ${covariance[1][1].toFixed(3)}]`;
    }

    function renderAll(recompute = true) {
      if (recompute || !state.currentData) {
        state.currentData = computeData();
      }
      updateReadouts();
      drawControlPad();
      renderScatter();
      renderHistogram();
      updateCovReadout();
      updateToggleButtonLabels();
      updateSampleSizeButtons();
    }

    function scheduleRender(recompute = true) {
      if (state.animationFrame !== null) {
        cancelAnimationFrame(state.animationFrame);
      }
      state.animationFrame = requestAnimationFrame(() => {
        renderAll(recompute);
        state.animationFrame = null;
      });
    }

    function updateToggleButtonLabels() {
      toggleButtons.z_ref.textContent = state.visibility.z_ref ? "Hide z_ref" : "Show z_ref";
      toggleButtons.z0.textContent = state.visibility.z0 ? "Hide z_0" : "Show z_0";
      toggleButtons.z1.textContent = state.visibility.z1 ? "Hide z_1" : "Show z_1";

      toggleButtons.z_ref.classList.toggle("ghost-off", !state.visibility.z_ref);
      toggleButtons.z0.classList.toggle("ghost-off", !state.visibility.z0);
      toggleButtons.z1.classList.toggle("ghost-off", !state.visibility.z1);
    }

    function updateSampleSizeButtons() {
      const buttons = sampleSizeButtons.querySelectorAll("button");
      buttons.forEach((button) => {
        const value = Number(button.dataset.size);
        button.classList.toggle("active", value === state.sampleSize);
      });
    }

    function buildSampleSizeButtons() {
      CONFIG.sample_size_options.forEach((size) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.size = String(size);
        button.textContent = String(size);
        button.addEventListener("click", () => {
          state.sampleSize = size;
          scheduleRender(true);
        });
        sampleSizeButtons.appendChild(button);
      });
    }

    compYSlider.addEventListener("input", () => {
      state.compY = Number(compYSlider.value);
      updateReadouts();
      scheduleRender(true);
    });

    function setStateFromControlEvent(event) {
      const rect = controlPad.getBoundingClientRect();
      const x = clamp(((event.clientX - rect.left) / rect.width) * controlPad.width, CONTROL_BOUNDS.left, CONTROL_BOUNDS.right);
      const y = clamp(((event.clientY - rect.top) / rect.height) * controlPad.height, CONTROL_BOUNDS.top, CONTROL_BOUNDS.bottom);
      state.ratio = canvasXToRatio(x);
      state.angleDeg = canvasYToAngle(y);
      updateReadouts();
      drawControlPad();
      scheduleRender(true);
    }

    controlPad.addEventListener("mousedown", (event) => {
      state.dragActive = true;
      setStateFromControlEvent(event);
    });

    window.addEventListener("mousemove", (event) => {
      if (!state.dragActive) return;
      setStateFromControlEvent(event);
    });

    window.addEventListener("mouseup", () => {
      state.dragActive = false;
    });

    controlPad.addEventListener("click", (event) => {
      setStateFromControlEvent(event);
    });

    toggleButtons.z_ref.addEventListener("click", () => {
      state.visibility.z_ref = !state.visibility.z_ref;
      renderAll(false);
    });
    toggleButtons.z0.addEventListener("click", () => {
      state.visibility.z0 = !state.visibility.z0;
      renderAll(false);
    });
    toggleButtons.z1.addEventListener("click", () => {
      state.visibility.z1 = !state.visibility.z1;
      renderAll(false);
    });

    resampleButton.addEventListener("click", () => {
      state.resampleNonce += 1;
      scheduleRender(true);
    });

    saveScatterButton.addEventListener("click", () => {
      Plotly.downloadImage(scatterPlot, {
        format: "png",
        filename: "simulated_measurements",
        width: 1100,
        height: 880,
        scale: 2,
      });
    });

    saveHistButton.addEventListener("click", () => {
      Plotly.downloadImage(histogramPlot, {
        format: "png",
        filename: "decision_margin_histogram",
        width: 900,
        height: 880,
        scale: 2,
      });
    });

    buildSampleSizeButtons();
    compYSlider.value = String(state.compY);
    renderAll(true);
  </script>
</body>
</html>
"""

    plotly_js = get_plotlyjs()
    return (
        html
        .replace("__PLOTLY_JS__", plotly_js)
        .replace("__CONFIG_JSON__", json.dumps(config))
    )


def main() -> None:
    output_dir = Path(get_path("dropbox_root_mac_elps")) / "WishartPractice_FigFiles" / "MC_simulations"
    output_name = 'MC_interactive.html'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_html = output_dir / output_name
    html = build_html(default_config())
    output_html.write_text(html, encoding="utf-8")
    print(f"Wrote interactive HTML to: {output_html}")


if __name__ == "__main__":
    main()
