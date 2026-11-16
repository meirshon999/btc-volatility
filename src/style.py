"""One visual language for every chart: matplotlib (report) and Plotly (dashboard).

Colours come from a colour-blind-checked categorical palette, assigned in a
fixed order and always to the same model, so a model keeps its colour in
every figure. Gridlines and axes are thin and quiet; the data stand out.
"""
from __future__ import annotations

import matplotlib as mpl
import plotly.graph_objects as go
import plotly.io as pio

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

LIGHT = {"surface": "#fcfcfb", "page": "#f9f9f7", "text": "#0b0b0b", "text_secondary": "#52514e",
         "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "neutral": "#c3c2b7"}
DARK = {"surface": "#1a1a19", "page": "#0d0d0d", "text": "#ffffff", "text_secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "neutral": "#5d5c58"}

# sequential (one hue, light -> dark) and diverging (blue <-> gray <-> red) scales
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"], [0.75, "#ec8c8b"], [1.0, "#b52f2f"]]

# every model keeps one colour everywhere (fixed order, never re-assigned)
MODEL_ORDER = ["LightGBM", "HAR-RV", "GARCH(1,1)", "Naive", "Seasonal naive", "Moving average (30d)",
               "Rolling std (30d)"]
MODEL_COLORS = dict(zip(MODEL_ORDER, PALETTE))
MODEL_COLORS_DARK = dict(zip(MODEL_ORDER, PALETTE_DARK))

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MPL_FONTS = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def model_color(name: str, dark: bool = False) -> str:
    colors = MODEL_COLORS_DARK if dark else MODEL_COLORS
    return colors.get(name, (DARK if dark else LIGHT)["muted"])


# --- matplotlib -----------------------------------------------------------------

def use_matplotlib_style() -> None:
    """Quiet axes, horizontal hairline grid, no chart junk."""
    c = LIGHT
    mpl.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"], "figure.dpi": 110, "savefig.dpi": 200,
        "font.family": "sans-serif", "font.sans-serif": MPL_FONTS, "font.size": 10,
        "text.color": c["text"], "axes.labelcolor": c["text_secondary"],
        "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 22, "axes.labelsize": 9.5,
        "axes.edgecolor": c["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": c["grid"],
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.color": c["muted"], "ytick.color": c["muted"],
        "xtick.labelcolor": c["text_secondary"], "ytick.labelcolor": c["text_secondary"],
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 3, "ytick.major.size": 0,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.constrained_layout.use": True,
    })


# --- Plotly ---------------------------------------------------------------------------

def plotly_template(dark: bool = False) -> go.layout.Template:
    c = DARK if dark else LIGHT
    axis = dict(gridcolor=c["grid"], gridwidth=1, linecolor=c["axis"], zeroline=False, automargin=True,
                tickfont=dict(color=c["text_secondary"], size=11),
                title=dict(font=dict(color=c["text_secondary"], size=12)), ticks="")
    return go.layout.Template(layout=go.Layout(
        font=dict(family=FONT, color=c["text"], size=12),
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        colorway=PALETTE_DARK if dark else PALETTE,
        title=dict(font=dict(size=15, color=c["text"]), x=0, xanchor="left", y=0.97),
        xaxis={**axis, "showgrid": False, "ticks": "outside", "tickcolor": c["axis"]},
        yaxis={**axis, "showgrid": True},
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                    font=dict(color=c["text_secondary"], size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=c["surface"], bordercolor=c["axis"],
                        font=dict(family=FONT, color=c["text"], size=12)),
        margin=dict(l=56, r=24, t=64, b=48),
    ))


def register_plotly_templates() -> None:
    pio.templates["btc_light"] = plotly_template(dark=False)
    pio.templates["btc_dark"] = plotly_template(dark=True)
    pio.templates.default = "btc_light"
