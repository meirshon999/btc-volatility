"""Interactive Plotly figures for the Streamlit dashboard and the HTML exports."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src import config
from src.style import BLUE_RAMP, DARK, DIVERGING, LIGHT, model_color, plotly_template

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _base(fig: go.Figure, title: str, dark: bool, height: int = 420) -> go.Figure:
    fig.update_layout(template=plotly_template(dark), title=title, height=height)
    return fig


def _colors(dark: bool) -> dict:
    return DARK if dark else LIGHT


def _halving_marks(fig: go.Figure, start, end, dark: bool, rows: list[int] | None = None) -> None:
    """Thin vertical lines at the halvings (shapes + annotations; works on date axes)."""
    c = _colors(dark)
    for h in config.HALVINGS[1:]:
        day = pd.Timestamp(h["start_date"])
        if not (pd.Timestamp(start) <= day <= pd.Timestamp(end)):
            continue
        for row in rows or [None]:
            yref = "paper" if row is None else ("y domain" if row == 1 else f"y{row} domain")
            fig.add_shape(type="line", x0=day, x1=day, y0=0, y1=1, xref="x", yref=yref,
                          line=dict(color=c["axis"], width=1))
        fig.add_annotation(x=day, y=1, xref="x", yref="paper", text=f"halving {h['epoch']}",
                           showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=10, color=c["muted"]))


def event_trace(daily: pd.DataFrame, column: str, dark: bool) -> go.Scatter | None:
    """Numbered markers at well-known market shocks, with the day's move in the tooltip."""
    c = _colors(dark)
    rows = []
    for i, event in enumerate(config.MARKET_EVENTS, start=1):
        day = pd.Timestamp(event["date"])
        if day in daily.index:
            rows.append((i, day, event["label"], daily.loc[day]))
    if not rows:
        return None
    return go.Scatter(
        x=[r[1] for r in rows], y=[r[3][column] for r in rows], mode="markers+text", name="events",
        text=[str(r[0]) for r in rows], textposition="middle center",
        textfont=dict(size=10, color=c["text"], family="system-ui, sans-serif"),
        marker=dict(size=19, color=c["surface"], line=dict(color=c["text"], width=1.4)),
        customdata=[[r[2], r[3]["log_return"], r[3]["realized_vol"]] for r in rows],
        hovertemplate="<b>%{customdata[0]}</b><br>%{x|%d %b %Y}<br>return %{customdata[1]:+.1f} %"
                      "<br>volatility %{customdata[2]:.1f} %<extra></extra>",
        showlegend=False)


def price_chart(daily: pd.DataFrame, log_scale: bool = True, dark: bool = False) -> go.Figure:
    c = _colors(dark)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["close"], name="close",
                             line=dict(color=model_color("LightGBM", dark), width=1.6),
                             hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"), row=1, col=1)
    events = event_trace(daily, "close", dark)
    if events is not None:
        fig.add_trace(events, row=1, col=1)
    rule = "W-MON" if len(daily) > 400 else "D"
    volume = daily["volume"].resample(rule).sum()
    fig.add_trace(go.Bar(x=volume.index, y=volume, name="volume", marker_color=BLUE_RAMP[2],
                         marker_line_width=0,
                         hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} BTC<extra></extra>"), row=2, col=1)
    _halving_marks(fig, daily.index[0], daily.index[-1], dark, rows=[1, 2])
    if log_scale:      # only the powers of ten: $1, $10, ..., $100k
        fig.update_yaxes(type="log", dtick=1, tickformat="~s", tickprefix="$", row=1, col=1)
    else:
        fig.update_yaxes(tickformat="~s", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="BTC / week" if rule != "D" else "BTC / day", row=2, col=1)
    fig.update_layout(showlegend=False, hovermode="closest", bargap=0.1)
    fig.update_xaxes(showspikes=True, spikecolor=c["axis"], spikethickness=1, spikedash="solid")
    return _base(fig, "Bitcoin price (USD) and traded volume", dark, height=520)


def volatility_chart(daily: pd.DataFrame, window: int = 30, dark: bool = False) -> go.Figure:
    c = _colors(dark)
    rolling = daily["realized_vol"].rolling(window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily["realized_vol"], name="daily",
                             line=dict(color=c["neutral"], width=1),
                             hovertemplate="%{y:.2f} %<extra>daily</extra>"))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling, name=f"{window}-day average",
                             line=dict(color=model_color("LightGBM", dark), width=2),
                             hovertemplate="%{y:.2f} %<extra>" + f"{window}-day avg</extra>"))
    events = event_trace(daily, "realized_vol", dark)
    if events is not None:
        fig.add_trace(events)
    _halving_marks(fig, daily.index[0], daily.index[-1], dark)
    fig.update_layout(hovermode="closest", yaxis_title="Daily realised volatility, %")
    return _base(fig, "Realised volatility (from 5-minute returns)", dark)


def calendar_heatmap(daily: pd.DataFrame, year: int, dark: bool = False) -> go.Figure:
    """One year as a GitHub-style calendar: columns are weeks, rows are weekdays."""
    c = _colors(dark)
    part = daily.loc[str(year)]
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    column = ((part.index - jan1).days + jan1.day_of_week) // 7
    z = np.full((7, 54), np.nan)
    text = np.full((7, 54), "", dtype=object)
    z[part.index.day_of_week, column] = part["realized_vol"].to_numpy()
    for day, row, col in zip(part.index, part.index.day_of_week, column):
        r = part.loc[day]
        text[row, col] = f"{day:%a %d %b %Y}<br>volatility {r['realized_vol']:.2f} %<br>return {r['log_return']:+.1f} %"
    ramp = BLUE_RAMP[::-1] if dark else BLUE_RAMP
    month_cols = [((pd.Timestamp(year=year, month=m, day=1) - jan1).days + jan1.day_of_week) // 7
                  for m in range(1, 13)]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(54)), y=WEEKDAYS, zmin=1, zmax=7, xgap=3, ygap=3,
        colorscale=[[i / (len(ramp) - 1), color] for i, color in enumerate(ramp)],
        customdata=text, hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title=dict(text="% / day", side="right"), thickness=12, outlinewidth=0)))
    fig.update_xaxes(tickvals=month_cols, ticktext=[pd.Timestamp(2021, m, 1).strftime("%b") for m in range(1, 13)],
                     showgrid=False, ticks="", side="top")
    fig.update_yaxes(autorange="reversed", showgrid=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor=c["surface"])
    shade = "brighter" if dark else "darker"
    fig.update_layout(margin=dict(l=48, r=24, t=70, b=10))
    return _base(fig, f"Volatility calendar {year}: one square per day, {shade} = stormier", dark, height=250)


def weekday_chart(daily: pd.DataFrame, dark: bool = False) -> go.Figure:
    means = daily["realized_vol"].groupby(daily.index.day_of_week).mean().reindex(range(7))
    colors = [model_color("LightGBM", dark)] * 5 + [model_color("HAR-RV", dark)] * 2
    fig = go.Figure(go.Bar(x=WEEKDAYS, y=means, marker_color=colors, marker_line_width=0,
                           text=[f"{v:.2f} %" for v in means], textposition="outside",
                           hovertemplate="%{x}: %{y:.2f} %<extra></extra>"))
    fig.update_layout(yaxis_title="Mean daily volatility, %", showlegend=False)
    return _base(fig, "Volatility by weekday (weekends in orange)", dark, height=380)


def intraday_heatmap(profile: pd.DataFrame, dark: bool = False) -> go.Figure:
    grid = profile.pivot(index="weekday", columns="hour", values="avg_abs_return_pct")
    ramp = BLUE_RAMP[::-1] if dark else BLUE_RAMP
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(), x=[f"{h:02d}:00" for h in grid.columns], y=WEEKDAYS,
        colorscale=[[i / (len(ramp) - 1), color] for i, color in enumerate(ramp)],
        colorbar=dict(title=dict(text="% / hour", side="right"), thickness=12, outlinewidth=0),
        xgap=2, ygap=2,
        hovertemplate="%{y} %{x} UTC<br>mean |return| %{z:.3f} %<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, ticks="")
    return _base(fig, "Mean absolute hourly return by weekday and hour (UTC)", dark, height=360)


def return_histogram(daily: pd.DataFrame, dark: bool = False) -> go.Figure:
    r = daily["log_return"].dropna()
    grid = np.linspace(-20, 20, 400)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r.clip(-20, 20), nbinsx=120, histnorm="probability density",
                               name="daily returns", marker_color=model_color("LightGBM", dark),
                               marker_line_width=0, opacity=0.85,
                               hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=grid, y=stats.norm.pdf(grid, r.mean(), r.std()), name="normal fit",
                             line=dict(color=model_color("HAR-RV", dark), width=2), hoverinfo="skip"))
    fig.update_layout(xaxis_title="Daily log return, %", yaxis_title="Density", bargap=0)
    return _base(fig, f"Daily returns: excess kurtosis {r.kurt():.1f}", dark, height=380)


def correlation_heatmap(corr: pd.DataFrame, dark: bool = False) -> go.Figure:
    c = _colors(dark)
    values = corr.to_numpy().copy()
    values[np.triu_indices_from(values, k=0)] = np.nan
    text = np.where(np.isnan(values), "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(values)))
    scale = [[p, color] for p, color in DIVERGING]
    if dark:
        scale[2] = [0.5, "#383835"]
    fig = go.Figure(go.Heatmap(z=values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=scale, text=text, texttemplate="%{text}",
                               textfont=dict(size=11, color=c["text"]), xgap=2, ygap=2,
                               colorbar=dict(thickness=12, outlinewidth=0),
                               hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _base(fig, "Spearman correlations", dark, height=480)


def rolling_correlation_chart(rolling: pd.Series, dark: bool = False) -> go.Figure:
    c = _colors(dark)
    fig = go.Figure(go.Scatter(x=rolling.index, y=rolling, line=dict(color=model_color("LightGBM", dark), width=2),
                               fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
                               hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=c["axis"], width=1))
    fig.update_layout(yaxis_title="correlation", showlegend=False)
    return _base(fig, "Rolling 1-year correlation of BTC volatility with the VIX", dark, height=380)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def forecast_chart(predictions: pd.DataFrame, models: list[str], dark: bool = False,
                   intervals: pd.DataFrame | None = None, show_days: int = 120,
                   selector: bool = True, height: int = 480,
                   title: str = "One-step-ahead forecasts vs realised volatility (test period)") -> go.Figure:
    c = _colors(dark)
    fig = go.Figure()
    if intervals is not None and "LightGBM" in models:
        band = intervals.reindex(predictions.index)
        fig.add_trace(go.Scatter(x=band.index, y=band["upper"], line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=band.index, y=band["lower"], line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(model_color("LightGBM", dark), 0.16), name="80 % range",
                                 customdata=band[["upper"]].to_numpy(),
                                 hovertemplate="%{y:.2f} – %{customdata[0]:.2f} %<extra>80 % range</extra>"))
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions["actual"], name="actual",
                             mode="lines+markers", line=dict(color=c["text"], width=1.4),
                             marker=dict(size=4), hovertemplate="%{y:.2f} %<extra>actual</extra>"))
    for name in models:
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions[name], name=name,
                                 line=dict(color=model_color(name, dark), width=1.8),
                                 hovertemplate="%{y:.2f} %<extra>" + name + "</extra>"))
    # open on the last months (readable); buttons zoom out to the whole test period
    last = predictions.index[-1]
    first_shown = max(predictions.index[0], last - pd.Timedelta(days=show_days))
    fig.update_xaxes(range=[first_shown, last + pd.Timedelta(days=2)])
    if selector:
        fig.update_xaxes(rangeselector=dict(
            buttons=[dict(count=1, label="1 mo", step="month", stepmode="backward"),
                     dict(count=3, label="3 mo", step="month", stepmode="backward"),
                     dict(count=6, label="6 mo", step="month", stepmode="backward"),
                     dict(step="all", label="All")],
            bgcolor=c["surface"], activecolor=c["grid"], bordercolor=c["axis"], borderwidth=1,
            font=dict(color=c["text_secondary"]), x=1, xanchor="right", y=1.02, yanchor="bottom"))
    fig.update_layout(hovermode="x unified", yaxis_title="Daily volatility, %")
    return _base(fig, title, dark, height=height)


def skill_chart(metrics: pd.DataFrame, dark: bool = False) -> go.Figure:
    """How much smaller each model's error is than the naive forecast's (in %)."""
    c = _colors(dark)
    data = metrics[metrics["model"] != "Naive"].assign(skill=lambda d: -d["mae_vs_naive_pct"])
    data = data.sort_values("skill")
    best = data["model"].iloc[-1]
    colors = [model_color(m, dark) if m == best else c["neutral"] for m in data["model"]]
    fig = go.Figure(go.Bar(x=data["skill"], y=data["model"], orientation="h", marker_color=colors,
                           marker_line_width=0, text=[f"{v:+.0f} %" for v in data["skill"]],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:+.1f} % vs naive<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=c["text_secondary"], width=1))
    fig.update_layout(xaxis_title="error reduction vs naive forecast, % (right = better)",
                      showlegend=False, margin=dict(l=160, r=60))
    fig.update_xaxes(showgrid=True, gridcolor=c["grid"], ticksuffix=" %", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return _base(fig, "How much better than “tomorrow = today”?", dark, height=360)


def validation_timeline(cv: pd.DataFrame, last_day: pd.Timestamp, dark: bool = False) -> go.Figure:
    """Train / validation / test periods of every fold on one time axis."""
    c = _colors(dark)
    start, test_start = pd.Timestamp(config.MODEL_START), pd.Timestamp(config.TEST_START)
    folds = cv.groupby("fold")[["val_start", "val_end"]].first()
    rows = [(f"CV fold {k}", pd.Timestamp(r["val_start"]), pd.Timestamp(r["val_end"]), "validate")
            for k, r in folds.iterrows()]
    rows.append(("Final test", test_start, last_day, "test"))
    fig = go.Figure()
    shown = set()
    for label, eval_start, eval_end, kind in rows:
        for part, a, b, color in (("train", start, eval_start, BLUE_RAMP[1]),
                                  (kind, eval_start, eval_end + pd.Timedelta(days=1),
                                   model_color("LightGBM", dark) if kind == "validate" else model_color("HAR-RV", dark))):
            fig.add_trace(go.Bar(
                y=[label], x=[(b - a).total_seconds() * 1000], base=[a], orientation="h",
                marker_color=color, marker_line_width=0, name=part, showlegend=part not in shown,
                hovertemplate=f"{label} · {part}: {a:%d %b %Y} – {(b - pd.Timedelta(days=1)):%d %b %Y}<extra></extra>"))
            shown.add(part)
    fig.update_xaxes(type="date", showgrid=True, gridcolor=c["grid"])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(barmode="overlay", bargap=0.35)
    return _base(fig, "Validation: always train on the past, test on the future", dark, height=330)


def regime_gauge(value: float, low: float, high: float, top: float, dark: bool = False) -> go.Figure:
    """Where today's 30-day volatility sits between calm, normal and stormy (history since 2017)."""
    c = _colors(dark)
    fig = go.Figure()
    bands = [(0, low, BLUE_RAMP[0], "calm", "#0b0b0b"), (low, high, BLUE_RAMP[2], "normal", "#0b0b0b"),
             (high, top, BLUE_RAMP[4], "stormy", "#ffffff")]
    for start, stop, color, label, ink in bands:
        fig.add_shape(type="rect", x0=start, x1=stop, y0=0.12, y1=0.52, fillcolor=color,
                      line=dict(width=2, color=c["surface"]), layer="below")
        fig.add_annotation(x=(start + stop) / 2, y=0.32, text=label, showarrow=False,
                           font=dict(size=12, color=ink))
    marker = min(value, top)
    fig.add_trace(go.Scatter(x=[marker], y=[0.6], mode="markers", hoverinfo="skip",
                             marker=dict(symbol="triangle-down", size=14, color=c["text"])))
    fig.add_annotation(x=marker, y=0.7, text=f"today {value:.0f} %", showarrow=False, yanchor="bottom",
                       font=dict(size=12, color=c["text"]))
    fig.update_xaxes(range=[0, top], showgrid=False, zeroline=False, showline=False, ticks="",
                     tickvals=[0, low, high, top],
                     ticktext=["0 %", f"{low:.0f} %", f"{high:.0f} %", f"{top:.0f} %+"])
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(template=plotly_template(dark), height=130, showlegend=False,
                      margin=dict(l=12, r=12, t=4, b=26))
    return fig


def metric_bars(metrics: pd.DataFrame, metric: str, dark: bool = False) -> go.Figure:
    c = _colors(dark)
    order = metrics.sort_values(metric, ascending=False)
    focus = metrics.sort_values(metric).iloc[0]["model"]
    colors = [model_color(m, dark) if m == focus else c["neutral"] for m in order["model"]]
    fig = go.Figure(go.Bar(x=order[metric], y=order["model"], orientation="h", marker_color=colors,
                           marker_line_width=0, text=[f"{v:.3f}" for v in order[metric]],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(xaxis_title=f"{metric.upper()} (lower is better)", showlegend=False,
                      margin=dict(l=160, r=60))
    fig.update_xaxes(showgrid=True, gridcolor=c["grid"])
    fig.update_yaxes(showgrid=False)
    return _base(fig, f"Test {metric.upper()} by model", dark, height=360)


def monthly_error_chart(by_month: pd.DataFrame, models: list[str], dark: bool = False) -> go.Figure:
    fig = go.Figure()
    x = pd.PeriodIndex(by_month.index, freq="M").to_timestamp()
    for name in models:
        fig.add_trace(go.Scatter(x=x, y=by_month[name], name=name, mode="lines+markers",
                                 line=dict(color=model_color(name, dark), width=2), marker=dict(size=6),
                                 hovertemplate="%{x|%b %Y}: %{y:.3f}<extra>" + name + "</extra>"))
    fig.update_layout(hovermode="x unified", yaxis_title="MAE, percentage points")
    return _base(fig, "Forecast error by month", dark, height=380)
