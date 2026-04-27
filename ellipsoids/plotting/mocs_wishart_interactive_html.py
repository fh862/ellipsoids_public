#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive HTML plotter for MOCS psychometric functions and WPPM regression.
"""

import json

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline.offline import get_plotlyjs

from plotting.wishart_predictions_plotting import WishartPredictionsVisualization_html


class MOCSWishartInteractiveHTMLPlotter:
    def __init__(self, reference_payloads, regression_payload, condition_panel_payload, page_title):
        self.reference_payloads = reference_payloads
        self.regression_payload = regression_payload
        self.condition_panel_payload = condition_panel_payload
        self.page_title = page_title

    @staticmethod
    def rgb_to_css(rgb):
        return WishartPredictionsVisualization_html.to_rgb_str(rgb)

    @staticmethod
    def ref_label(payload):
        vals = ", ".join(f"{v:.2f}" for v in payload["xref"])
        return f"[{vals}]"

    @staticmethod
    def default_plane_outline():
        return np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=float,
        )

    @staticmethod
    def default_cube_edges():
        cube_vertices = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        cube_edge_idx = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        return [cube_vertices[list(edge)] for edge in cube_edge_idx]

    def make_condition_figure(self):
        payload = self.condition_panel_payload
        fig = go.Figure()
        ndims = payload["ndims"]

        if ndims == 2:
            plane_outline = np.asarray(
                payload.get("plane_outline", self.default_plane_outline()),
                dtype=float,
            )
            fig.add_trace(
                go.Scatter(
                    x=plane_outline[:, 0],
                    y=plane_outline[:, 1],
                    mode="lines",
                    line=dict(color="rgba(90, 90, 90, 0.45)", width=1.8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            cube_edges = payload.get("cube_edges", self.default_cube_edges())
            for edge in cube_edges:
                edge = np.asarray(edge, dtype=float)
                fig.add_trace(
                    go.Scatter3d(
                        x=edge[:, 0],
                        y=edge[:, 1],
                        z=edge[:, 2],
                        mode="lines",
                        line=dict(color="rgba(90, 90, 90, 0.38)", width=4),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        for ref_payload in payload["references"]:
            color = ref_payload["color_css"]
            xref = np.asarray(ref_payload["xref"], dtype=float)
            comp_arr = np.asarray(ref_payload["comp_points"], dtype=float)
            comp_arr = np.reshape(comp_arr, (-1, xref.size))
            if comp_arr.shape[0] == 0:
                continue
            dist = np.linalg.norm(comp_arr - xref[None], axis=1)
            farthest_comp = comp_arr[int(np.argmax(dist))]
            hover_lines = "<br>".join(
                [
                    ref_payload["ref_name"],
                    f"Conditions: {ref_payload['n_comp']}",
                ]
            )
            if ndims == 2:
                fig.add_trace(
                    go.Scatter(
                        x=[xref[0], farthest_comp[0]],
                        y=[xref[1], farthest_comp[1]],
                        mode="lines",
                        line=dict(color=color, width=1.8),
                        opacity=0.28,
                        meta=ref_payload["ref_key"],
                        hovertemplate=hover_lines + "<extra></extra>",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[xref[0]],
                        y=[xref[1]],
                        mode="markers",
                        marker=dict(symbol="cross", size=10, color=color, line=dict(width=0)),
                        opacity=0.55,
                        name="Reference",
                        meta=ref_payload["ref_key"],
                        hovertemplate=hover_lines + "<extra></extra>",
                        showlegend=ref_payload is payload["references"][0],
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=comp_arr[:, 0],
                        y=comp_arr[:, 1],
                        mode="markers",
                        marker=dict(symbol="circle", size=4, color=color, line=dict(width=0)),
                        opacity=0.42,
                        name="Comparison",
                        meta=ref_payload["ref_key"],
                        hovertemplate=hover_lines + "<extra></extra>",
                        showlegend=ref_payload is payload["references"][0],
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=[xref[0], farthest_comp[0]],
                        y=[xref[1], farthest_comp[1]],
                        z=[xref[2], farthest_comp[2]],
                        mode="lines",
                        line=dict(color=color, width=4),
                        opacity=0.22,
                        meta=ref_payload["ref_key"],
                        hovertemplate=hover_lines + "<extra></extra>",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=[xref[0]],
                        y=[xref[1]],
                        z=[xref[2]],
                        mode="markers",
                        marker=dict(symbol="cross", size=6, color=color, line=dict(width=0)),
                        opacity=0.55,
                        name="Reference",
                        meta=ref_payload["ref_key"],
                        hovertemplate=hover_lines + "<extra></extra>",
                        showlegend=ref_payload is payload["references"][0],
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=comp_arr[:, 0],
                        y=comp_arr[:, 1],
                        z=comp_arr[:, 2],
                        mode="markers",
                        marker=dict(symbol="circle", size=2.6, color=color, line=dict(width=0)),
                        opacity=0.38,
                        name="Comparison",
                        meta=ref_payload["ref_key"],
                        hovertemplate=hover_lines + "<extra></extra>",
                        showlegend=ref_payload is payload["references"][0],
                    )
                )
        if ndims == 2:
            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Arial", size=13),
                margin=dict(l=52, r=24, t=24, b=52),
                xaxis=dict(
                    title="Model space dimension 1",
                    range=[-1.0, 1.0],
                    tickvals=np.linspace(-0.7, 0.7, 5),
                    gridcolor="rgba(30, 30, 30, 0.14)",
                    zeroline=False,
                    scaleanchor="y",
                    scaleratio=1,
                ),
                yaxis=dict(
                    title="Model space dimension 2",
                    range=[-1.0, 1.0],
                    tickvals=np.linspace(-0.7, 0.7, 5),
                    gridcolor="rgba(30, 30, 30, 0.14)",
                    zeroline=False,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.28,
                    xanchor="center",
                    x=0.5,
                ),
                uirevision="mocs-condition-panel",
            )
        else:
            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(family="Arial", size=13),
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(
                    xaxis=dict(
                        title="Model space dimension 1",
                        range=[-1.0, 1.0],
                        tickvals=np.linspace(-0.7, 0.7, 5),
                        showbackground=False,
                        gridcolor="rgba(30, 30, 30, 0.24)",
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title="Model space dimension 2",
                        range=[-1.0, 1.0],
                        tickvals=np.linspace(-0.7, 0.7, 5),
                        showbackground=False,
                        gridcolor="rgba(30, 30, 30, 0.24)",
                        zeroline=False,
                    ),
                    zaxis=dict(
                        title="Model space dimension 3",
                        range=[-1.0, 1.0],
                        tickvals=np.linspace(-0.7, 0.7, 5),
                        showbackground=False,
                        gridcolor="rgba(30, 30, 30, 0.24)",
                        zeroline=False,
                    ),
                    aspectmode="cube",
                    camera=dict(
                        eye=dict(x=1.55, y=1.45, z=1.1),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0),
                    ),
                    bgcolor="#ffffff",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                ),
                uirevision="mocs-condition-panel",
            )
        return fig

    @staticmethod
    def make_pmf_figure(payload):
        color = payload["color_css"]
        fig = go.Figure()
        stim_dist = np.asarray(payload["stim_dist"], dtype=float)
        pc_per_level = np.asarray(payload["pc_per_level"], dtype=float)
        obs_mask = ~np.isclose(stim_dist, 0.0)

        fig.add_trace(
            go.Scatter(
                x=payload["fine_val"],
                y=payload["mocs_ci_upper"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=payload["fine_val"],
                y=payload["mocs_ci_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(31, 41, 51, 0.14)",
                name="95% bootstrap CI of Weibull function",
                hovertemplate="MOCS CI<br>x=%{x:.3f}<br>pC=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=payload["fine_val"],
                y=payload["wppm_ci_upper"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=payload["fine_val"],
                y=payload["wppm_ci_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=payload["wppm_ci_fill_css"],
                name="95% bootstrap CI of WPPM prediction",
                hovertemplate="WPPM CI<br>x=%{x:.3f}<br>pC=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=payload["fine_val"],
                y=payload["mocs_fit"],
                mode="lines",
                line=dict(color="#111111", width=3.2),
                name="Best-fit Weibull function to MOCS trials",
                hovertemplate="Weibull fit<br>x=%{x:.3f}<br>pC=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=stim_dist[obs_mask],
                y=pc_per_level[obs_mask],
                mode="markers",
                marker=dict(size=12, color="#111111"),
                name="Observed MOCS data",
                hovertemplate="Observed<br>x=%{x:.3f}<br>pC=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=payload["fine_val"],
                y=payload["wppm_curve"],
                mode="lines",
                line=dict(color=color, width=2.4),
                name="WPPM prediction",
                hovertemplate="WPPM<br>x=%{x:.3f}<br>pC=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[payload["mocs_threshold"]],
                y=[payload["target_pc"]],
                mode="markers",
                marker=dict(size=0.1, color="rgba(0,0,0,0)"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[payload["mocs_threshold_err"][1]],
                    arrayminus=[payload["mocs_threshold_err"][0]],
                    thickness=6.0,
                    width=8,
                    color="#111111",
                ),
                hovertemplate=(
                    "Validation threshold CI<br>"
                    "x=%{x:.3f}<br>"
                    "pC=%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[payload["wppm_threshold"]],
                y=[payload["target_pc"]],
                mode="markers",
                marker=dict(size=0.1, color="rgba(0,0,0,0)"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[payload["wppm_threshold_err"][1]],
                    arrayminus=[payload["wppm_threshold_err"][0]],
                    thickness=3.0,
                    width=4,
                    color=color,
                ),
                hovertemplate=(
                    "WPPM threshold CI<br>"
                    "x=%{x:.3f}<br>"
                    "pC=%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        xmax = 1.05 * float(np.nanmax(stim_dist))
        y_candidates = np.concatenate(
            [
                np.asarray(payload["pc_per_level"], dtype=float),
                np.asarray(payload["mocs_ci_lower"], dtype=float),
                np.asarray(payload["wppm_ci_lower"], dtype=float),
                np.asarray([payload["target_pc"]], dtype=float),
            ]
        )
        ymin = float(np.nanmin(y_candidates))
        ymin = min(ymin - 0.03, 0.33)
        ymin = max(0.0, ymin)
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Arial", size=13),
            margin=dict(l=58, r=24, t=74, b=110),
            title=None,
            xaxis=dict(
                title="Euclidean distance between reference and comparison stimuli in model space",
                range=[0.0, xmax],
                gridcolor="rgba(30, 30, 30, 0.14)",
                zeroline=False,
                tickformat=".2f",
            ),
            yaxis=dict(
                title="Proportion correct",
                range=[ymin, 1.02],
                tickvals=[0.33, 0.67, 1.0],
                gridcolor="rgba(30, 30, 30, 0.14)",
                zeroline=False,
            ),
            showlegend=False,
            uirevision=f"pmf-{payload['ref_key']}",
        )
        return fig

    @staticmethod
    def make_regression_figure(payload, selected_key):
        fig = go.Figure()
        x_bds = payload["axis_bounds"]
        tickvals = np.linspace(x_bds[0], x_bds[1], 6)
        band_color = "rgba(31, 41, 51, 0.12)"

        fig.add_trace(
            go.Scatter(
                x=x_bds + x_bds[::-1],
                y=[
                    x_bds[0] * payload["slope_ci"][0],
                    x_bds[1] * payload["slope_ci"][0],
                    x_bds[1] * payload["slope_ci"][1],
                    x_bds[0] * payload["slope_ci"][1],
                ],
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=band_color,
                name="95% bootstrap CI of line fit",
                hovertemplate="Slope CI<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_bds,
                y=[x * payload["slope_org"] for x in x_bds],
                mode="lines",
                line=dict(color="#6b7280", width=2.4),
                name="Best line fit",
                hovertemplate="Best fit<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_bds,
                y=x_bds,
                mode="lines",
                line=dict(color="#111111", width=1.5, dash="dash"),
                name="Identity line",
                hovertemplate="Identity<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            )
        )

        selected_idx = payload["ref_keys"].index(selected_key)
        for idx, ref_key in enumerate(payload["ref_keys"]):
            is_selected = idx == selected_idx
            fig.add_trace(
            go.Scatter(
                x=[payload["x"][idx]],
                y=[payload["y"][idx]],
                mode="markers",
                marker=dict(
                        size=13 if is_selected else 10,
                        color=payload["colors_css"][idx],
                        line=dict(
                            color="#111111" if is_selected else "rgba(17, 17, 17, 0.2)",
                            width=1.8 if is_selected else 0.6,
                        ),
                        opacity=1.0 if is_selected else 0.85,
                    ),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[payload["x_err"][idx][1]],
                        arrayminus=[payload["x_err"][idx][0]],
                        thickness=3.0 if is_selected else 1.5,
                        width=4,
                        color=payload["colors_css"][idx],
                    ),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[payload["y_err"][idx][1]],
                        arrayminus=[payload["y_err"][idx][0]],
                        thickness=3.0 if is_selected else 1.5,
                        width=4,
                        color=payload["colors_css"][idx],
                    ),
                    name="Selected reference",
                    showlegend=is_selected,
                    hovertemplate=(
                        f"{payload['ref_names'][idx]}<br>"
                        "Validation threshold=%{x:.3f}<br>"
                        "WPPM threshold=%{y:.3f}<extra></extra>"
                    ),
                )
            )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.03,
            y=0.97,
            showarrow=False,
            align="left",
            font=dict(size=12, color="#1f2933"),
            text=(
                f"Corr coef = {payload['corr_org']:.2f}; 95% CI: "
                f"[{payload['corr_ci'][0]:.2f}, {payload['corr_ci'][1]:.2f}]<br>"
                f"Slope = {payload['slope_org']:.2f}; 95% CI: "
                f"[{payload['slope_ci'][0]:.2f}, {payload['slope_ci'][1]:.2f}]<br>"
                f"Overlapped CIs: {payload['num_overlaps']} / {len(payload['x'])}"
            ),
        )

        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Arial", size=13),
            margin=dict(l=62, r=24, t=74, b=60),
            title=None,
            xaxis=dict(
                title="Threshold distance (validation)",
                range=x_bds,
                tickmode="array",
                tickvals=tickvals,
                gridcolor="rgba(30, 30, 30, 0.14)",
                zeroline=False,
                tickformat=".2f",
            ),
            yaxis=dict(
                title="Threshold distance (WPPM)",
                range=x_bds,
                tickmode="array",
                tickvals=tickvals,
                gridcolor="rgba(30, 30, 30, 0.14)",
                zeroline=False,
                tickformat=".2f",
                scaleanchor="x",
                scaleratio=1,
                constrain="domain",
            ),
            autosize=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0.0),
            uirevision=f"regression-{selected_key}",
        )
        return fig

    def write_interactive_html(self, output_html):
        initial_key = self.reference_payloads[0]["ref_key"]
        condition_fig = self.make_condition_figure()
        pmf_figs = {
            payload["ref_key"]: pio.to_json(self.make_pmf_figure(payload), pretty=False)
            for payload in self.reference_payloads
        }
        regression_figs = {
            payload["ref_key"]: pio.to_json(
                self.make_regression_figure(self.regression_payload, payload["ref_key"]),
                pretty=False,
            )
            for payload in self.reference_payloads
        }

        plotly_js = get_plotlyjs()
        condition_fig_json = pio.to_json(condition_fig, pretty=False)
        pmf_figs_json = json.dumps({k: json.loads(v) for k, v in pmf_figs.items()})
        regression_figs_json = json.dumps({k: json.loads(v) for k, v in regression_figs.items()})
        ref_keys = [payload["ref_key"] for payload in self.reference_payloads]
        ref_colors = {payload["ref_key"]: payload["color_css"] for payload in self.reference_payloads}

        html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self.page_title}</title>
  <style>
    :root {{
      --page-bg: #e9e9e9;
      --panel-bg: #eeeeee;
      --ink: #1f2933;
      --muted: #5b6870;
      --edge: rgba(31, 41, 51, 0.12);
      --accent: #1e5f74;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--ink);
      background: #e9e9e9;
    }}
    .page {{
      min-height: 100vh;
      padding: 20px;
      box-sizing: border-box;
    }}
    .header {{
      margin: 0 0 14px 0;
    }}
    .title {{
      margin: 0;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .subtitle {{
      margin: 6px 0 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(420px, 1fr) minmax(420px, 1fr);
      gap: 18px;
      align-items: stretch;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid var(--edge);
      border-radius: 18px;
      box-shadow: 0 18px 40px rgba(73, 54, 27, 0.08);
      backdrop-filter: blur(10px);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .panel-header {{
      padding: 12px 16px 0 16px;
      background: rgba(238, 238, 238, 0.9);
    }}
    .panel-title {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }}
    .panel-copy {{
      margin: 6px 0 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .button-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 2px;
      padding-bottom: 6px;
      align-items: center;
    }}
    .control-button {{
      border: 1px solid rgba(31, 41, 51, 0.18);
      background: rgba(31, 41, 51, 0.08);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      line-height: 1;
      cursor: pointer;
      transition: all 140ms ease;
    }}
    .control-button:hover {{
      transform: translateY(-1px);
      background: rgba(31, 41, 51, 0.12);
    }}
    .control-button.active {{
      background: var(--ink);
      color: #ffffff;
      border-color: var(--ink);
    }}
    .ref-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 2px;
      padding-bottom: 6px;
      align-items: center;
    }}
    .ref-pills-label {{
      font-size: 13px;
      color: var(--muted);
      margin-right: 6px;
    }}
    .ref-pill {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 2px solid var(--ink);
      background: #ffffff;
      padding: 0;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease, box-shadow 120ms ease;
    }}
    .ref-pill:hover {{
      transform: translateY(-1px);
      box-shadow: 0 2px 8px rgba(31, 41, 51, 0.18);
    }}
    .ref-pill.active {{
      box-shadow: 0 0 0 2px rgba(31, 41, 51, 0.16);
    }}
    .plot {{
      width: 100%;
      height: 68vh;
      min-height: 520px;
      background: #ffffff;
    }}
    .custom-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      align-items: center;
      padding: 0 18px 14px 18px;
      color: var(--ink);
      font-size: 13px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      user-select: none;
      opacity: 1;
      transition: opacity 120ms ease;
    }}
    .legend-item.inactive {{
      opacity: 0.35;
    }}
    .legend-swatch {{
      width: 30px;
      height: 14px;
      display: inline-block;
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .plot {{
        height: 54vh;
        min-height: 460px;
      }}
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1 class="title">{self.page_title}</h1>
    </div>
    <div class="grid">
      <section class="panel">
        <div class="panel-header">
          <h2 class="panel-title">MOCS Conditions</h2>
          <div class="button-bar">
            <p class="panel-copy">Click</p>
            <button id="auto-sweep-button" type="button" class="control-button">auto sweep</button>
            <p class="panel-copy">to see all validation conditions.</p>
          </div>
          <div id="ref-pills" class="ref-pills">
            <span class="ref-pills-label">Inspect individual condition</span>
          </div>
        </div>
        <div id="conditions-panel" class="plot"></div>
      </section>
      <section class="panel">
        <div class="panel-header">
          <h2 class="panel-title">Psychometric Function</h2>
          <p class="panel-copy">MOCS data, Weibull fit, and WPPM prediction for the selected reference.</p>
        </div>
        <div id="pmf-panel" class="plot"></div>
        <div id="pmf-custom-legend" class="custom-legend">
          <span class="legend-item" data-key="wppm_prediction">
            <svg class="legend-swatch" viewBox="0 0 30 14" aria-hidden="true">
              <line x1="3" y1="7" x2="27" y2="7" stroke="#2b33b8" stroke-width="3"></line>
            </svg>
            <span>WPPM prediction</span>
          </span>
          <span class="legend-item" data-key="mocs_fit">
            <svg class="legend-swatch" viewBox="0 0 30 14" aria-hidden="true">
              <line x1="3" y1="7" x2="27" y2="7" stroke="#111111" stroke-width="4"></line>
            </svg>
            <span>Best-fit Weibull function to MOCS trials</span>
          </span>
          <span class="legend-item" data-key="mocs_ci">
            <svg class="legend-swatch" viewBox="0 0 30 14" aria-hidden="true">
              <rect x="3" y="4" width="24" height="6" fill="rgba(31,41,51,0.14)"></rect>
            </svg>
            <span>95% bootstrap CI of Weibull function</span>
          </span>
          <span class="legend-item" data-key="observed">
            <svg class="legend-swatch" viewBox="0 0 30 14" aria-hidden="true">
              <circle cx="15" cy="7" r="4.5" fill="#111111"></circle>
            </svg>
            <span>Observed MOCS data</span>
          </span>
          <span class="legend-item" data-key="wppm_ci">
            <svg class="legend-swatch" viewBox="0 0 30 14" aria-hidden="true">
              <rect x="3" y="4" width="24" height="6" fill="rgba(43,51,184,0.28)"></rect>
            </svg>
            <span>95% bootstrap CI of WPPM prediction</span>
          </span>
          <span class="legend-item" data-key="validation_ci">
            <svg class="legend-swatch" viewBox="0 0 30 14" aria-hidden="true">
              <line x1="6" y1="7" x2="24" y2="7" stroke="#111111" stroke-width="4"></line>
              <line x1="6" y1="3" x2="6" y2="11" stroke="#111111" stroke-width="3"></line>
              <line x1="24" y1="3" x2="24" y2="11" stroke="#111111" stroke-width="3"></line>
            </svg>
            <span>Validation threshold CI</span>
          </span>
          <span class="legend-item" data-key="wppm_threshold_ci">
            <svg class="legend-swatch" viewBox="0 0 30 14" aria-hidden="true">
              <line x1="6" y1="7" x2="24" y2="7" stroke="#2b33b8" stroke-width="3"></line>
              <line x1="6" y1="4" x2="6" y2="10" stroke="#2b33b8" stroke-width="2"></line>
              <line x1="24" y1="4" x2="24" y2="10" stroke="#2b33b8" stroke-width="2"></line>
            </svg>
            <span>WPPM threshold CI</span>
          </span>
        </div>
      </section>
      <section class="panel">
        <div class="panel-header">
          <h2 class="panel-title">Linear Regression</h2>
          <p class="panel-copy">Validation thresholds are plotted against WPPM thresholds with bootstrap confidence intervals.</p>
        </div>
        <div id="regression-panel" class="plot"></div>
      </section>
    </div>
  </div>
  <script>
    const conditionFigure = {condition_fig_json};
    const pmfFigures = {pmf_figs_json};
    const regressionFigures = {regression_figs_json};
    const refKeys = {json.dumps(ref_keys)};
    const refColors = {json.dumps(ref_colors)};
    let selectedRef = {json.dumps(initial_key)};
    let autoSweepTimer = null;
    let autoSweepIndex = 0;
    const pmfLegendState = {{
      wppm_prediction: true,
      mocs_fit: true,
      mocs_ci: true,
      observed: true,
      wppm_ci: true,
      validation_ci: true,
      wppm_threshold_ci: true
    }};
    const pmfLegendTraceMap = {{
      mocs_ci: [0, 1],
      wppm_ci: [2, 3],
      mocs_fit: [4],
      observed: [5],
      wppm_prediction: [6],
      validation_ci: [7],
      wppm_threshold_ci: [8]
    }};

    function drawConditions() {{
      return Plotly.newPlot(
        "conditions-panel",
        conditionFigure.data,
        conditionFigure.layout,
        {{responsive: true, displaylogo: false}}
      );
    }}

    function drawPMF(refKey) {{
      const fig = pmfFigures[refKey];
      return Plotly.react(
        "pmf-panel",
        fig.data,
        fig.layout,
        {{responsive: true, displaylogo: false}}
      ).then(() => {{
        applyPMFLegendState();
      }});
    }}

    function drawRegression(refKey) {{
      const fig = regressionFigures[refKey];
      return Plotly.react(
        "regression-panel",
        fig.data,
        fig.layout,
        {{responsive: true, displaylogo: false}}
      );
    }}

    function applyPMFLegendState() {{
      Object.entries(pmfLegendTraceMap).forEach(([key, traceIdxs]) => {{
        Plotly.restyle(
          "pmf-panel",
          {{visible: pmfLegendState[key] ? true : "legendonly"}},
          traceIdxs
        );
      }});
      document.querySelectorAll("#pmf-custom-legend .legend-item").forEach((item) => {{
        const key = item.dataset.key;
        item.classList.toggle("inactive", key ? !pmfLegendState[key] : false);
      }});
    }}

    function updateRefPills() {{
      document.querySelectorAll(".ref-pill").forEach((button) => {{
        const key = button.dataset.refKey;
        const color = refColors[key];
        const isActive = key === selectedRef;
        button.classList.toggle("active", isActive);
        button.style.borderColor = color;
        button.style.background = isActive ? color : "#ffffff";
      }});
    }}

    function updateConditionHighlight(refKey) {{
      const ndims = {self.condition_panel_payload["ndims"]};
      const tracesPerRef = 3;
      const refsStart = conditionFigure.data.length - (refKeys.length * tracesPerRef);
      refKeys.forEach((key, idx) => {{
        const isActive = key === refKey;
        const base = refsStart + idx * tracesPerRef;
        if (ndims === 2) {{
          Plotly.restyle("conditions-panel", {{
            "line.width": isActive ? 3.2 : 1.8,
            opacity: isActive ? 0.9 : 0.28
          }}, [base]);
          Plotly.restyle("conditions-panel", {{
            "marker.size": isActive ? 13 : 10,
            opacity: isActive ? 1.0 : 0.55
          }}, [base + 1]);
          Plotly.restyle("conditions-panel", {{
            "marker.size": isActive ? 8 : 6,
            opacity: isActive ? 0.9 : 0.42
          }}, [base + 2]);
          Plotly.restyle("conditions-panel", {{
            "line.width": isActive ? 3.2 : 1.8,
            opacity: isActive ? 0.95 : 0.55
          }}, [base + 3]);
        }} else {{
          Plotly.restyle("conditions-panel", {{
            "line.width": isActive ? 7 : 4,
            opacity: isActive ? 0.78 : 0.22
          }}, [base]);
          Plotly.restyle("conditions-panel", {{
            "marker.size": isActive ? 8 : 6,
            opacity: isActive ? 1.0 : 0.55
          }}, [base + 1]);
          Plotly.restyle("conditions-panel", {{
            "marker.size": isActive ? 5.5 : 3.5,
            opacity: isActive ? 0.82 : 0.38
          }}, [base + 2]);
        }}
      }});
    }}

    function bindConditionClick() {{
      const panel = document.getElementById("conditions-panel");
      panel.on("plotly_click", (ev) => {{
        const point = ev.points && ev.points[0];
        if (!point || !point.data || !point.data.meta) {{
          return;
        }}
        selectReference(point.data.meta);
      }});
    }}

    function updateAutoSweepButton(isActive) {{
      const button = document.getElementById("auto-sweep-button");
      button.classList.toggle("active", isActive);
      button.textContent = isActive ? "stop sweep" : "auto sweep";
    }}

    function selectReference(refKey) {{
      selectedRef = refKey;
      Promise.resolve().then(() => {{
        updateConditionHighlight(refKey);
        updateRefPills();
        drawPMF(refKey);
        drawRegression(refKey);
      }});
    }}

    function stopAutoSweep() {{
      if (autoSweepTimer !== null) {{
        window.clearInterval(autoSweepTimer);
        autoSweepTimer = null;
      }}
      updateAutoSweepButton(false);
    }}

    function startAutoSweep() {{
      stopAutoSweep();
      autoSweepIndex = 0;
      updateAutoSweepButton(true);
      selectReference(refKeys[autoSweepIndex]);
      autoSweepIndex += 1;
      autoSweepTimer = window.setInterval(() => {{
        if (autoSweepIndex >= refKeys.length) {{
          stopAutoSweep();
          return;
        }}
        selectReference(refKeys[autoSweepIndex]);
        autoSweepIndex += 1;
      }}, 1000);
    }}

    function buildControls() {{
      document.getElementById("auto-sweep-button").addEventListener("click", () => {{
        if (autoSweepTimer === null) {{
          startAutoSweep();
        }} else {{
          stopAutoSweep();
        }}
      }});
      document.querySelectorAll("#pmf-custom-legend .legend-item").forEach((item) => {{
        const key = item.dataset.key;
        if (!key) {{
          return;
        }}
        item.addEventListener("click", () => {{
          pmfLegendState[key] = !pmfLegendState[key];
          applyPMFLegendState();
        }});
      }});
      const pills = document.getElementById("ref-pills");
      refKeys.forEach((refKey) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ref-pill";
        button.dataset.refKey = refKey;
        button.title = refKey;
        button.setAttribute("aria-label", refKey);
        button.addEventListener("click", () => {{
          stopAutoSweep();
          selectReference(refKey);
        }});
        pills.appendChild(button);
      }});
      updateRefPills();
    }}

    buildControls();
    drawConditions().then(() => {{
      selectReference(selectedRef);
      bindConditionClick();
    }});
  </script>
</body>
</html>
"""

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html_str)
