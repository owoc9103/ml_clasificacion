"""Gráficos estilo mesa de trading (Plotly)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TRADING_COLORS = {
    "bg": "#0b0f14",
    "panel": "#121820",
    "grid": "#1e293b",
    "up": "#00c896",
    "down": "#ff5c6c",
    "accent": "#38bdf8",
    "warn": "#fbbf24",
    "text": "#e6edf5",
    "muted": "#64748b",
}

LAYOUT_BASE = dict(
    paper_bgcolor=TRADING_COLORS["bg"],
    plot_bgcolor=TRADING_COLORS["panel"],
    font=dict(color=TRADING_COLORS["text"], size=12),
    margin=dict(l=48, r=24, t=48, b=40),
    hovermode="x unified",
)


def _apply_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(showgrid=True, gridcolor=TRADING_COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=TRADING_COLORS["grid"], zeroline=False)
    return fig


def fig_allocation_treemap(ok: pd.DataFrame) -> go.Figure:
    df = ok[ok["target_weight"].astype(float) > 0.001].copy()
    df["pct"] = df["target_weight"].astype(float) * 100
    fig = go.Figure(
        go.Treemap(
            labels=df["symbol"],
            parents=[""] * len(df),
            values=df["pct"],
            texttemplate="<b>%{label}</b><br>%{value:.1f}%",
            marker=dict(
                colors=df["ml_proba"].astype(float),
                colorscale=[
                    [0, TRADING_COLORS["down"]],
                    [0.45, TRADING_COLORS["warn"]],
                    [1, TRADING_COLORS["up"]],
                ],
                showscale=True,
                colorbar=dict(title="ML prob.", tickformat=".0%"),
            ),
        )
    )
    fig.update_layout(**LAYOUT_BASE, title="Treemap de cartera sugerida (color = convicción ML)")
    return fig


def fig_weight_lollipop(ok: pd.DataFrame) -> go.Figure:
    df = ok.sort_values("target_weight", ascending=True).tail(12)
    y = df["symbol"].astype(str)
    x = df["target_weight"].astype(float) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0] * len(df),
            y=y,
            mode="markers",
            marker=dict(size=8, color=TRADING_COLORS["muted"]),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            line=dict(color=TRADING_COLORS["accent"], width=2),
            marker=dict(size=14, color=TRADING_COLORS["up"], symbol="circle"),
            name="Peso %",
            text=[f"{v:.1f}%" for v in x],
            hovertemplate="%{y}<br>Peso: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title="Lollipop — pesos objetivo (%)",
        xaxis_title="Asignación sugerida",
        height=max(320, len(df) * 36),
    )
    return _apply_axes(fig)


def fig_signal_heatmap(ok: pd.DataFrame) -> go.Figure:
    metrics = {
        "Señal": "signal",
        "ML prob.": "ml_proba",
        "Momentum": "momo_sig",
        "Mean rev.": "meanrev_sig",
    }
    cols = [c for c in metrics.values() if c in ok.columns]
    if not cols:
        return go.Figure()
    mat = ok.set_index("symbol")[cols].astype(float)
    # normalizar 0-1 por columna para comparar en heatmap
    norm = (mat - mat.min()) / (mat.max() - mat.min()).replace(0, 1)
    norm.columns = [k for k, v in metrics.items() if v in cols]
    fig = go.Figure(
        data=go.Heatmap(
            z=norm.values.T,
            x=norm.index.tolist(),
            y=norm.columns.tolist(),
            colorscale=[
                [0, TRADING_COLORS["panel"]],
                [0.5, TRADING_COLORS["warn"]],
                [1, TRADING_COLORS["up"]],
            ],
            showscale=True,
            colorbar=dict(title="Intensidad rel."),
            hovertemplate="Ticker: %{x}<br>%{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(**LAYOUT_BASE, title="Heatmap de señales (normalizado por fila)", height=280)
    return fig


def fig_conviction_gauge(symbol: str, proba: float, weight_pct: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": f"{symbol}<br><span style='font-size:0.75em'>Peso {weight_pct:.1f}%</span>"},
            delta={"reference": 45, "increasing": {"color": TRADING_COLORS["up"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": TRADING_COLORS["up"] if proba >= 0.45 else TRADING_COLORS["warn"]},
                "steps": [
                    {"range": [0, 45], "color": "#1a2332"},
                    {"range": [45, 70], "color": "#163d32"},
                    {"range": [70, 100], "color": "#0f4d3a"},
                ],
                "threshold": {
                    "line": {"color": TRADING_COLORS["accent"], "width": 3},
                    "thickness": 0.8,
                    "value": 45,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=TRADING_COLORS["bg"],
        font=dict(color=TRADING_COLORS["text"]),
        height=260,
        margin=dict(l=20, r=20, t=60, b=10),
    )
    return fig


def fig_candlestick(df: pd.DataFrame, symbol: str) -> go.Figure:
    if df.empty or not {"open", "high", "low", "close"}.issubset(df.columns):
        return go.Figure()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color=TRADING_COLORS["up"],
                decreasing_line_color=TRADING_COLORS["down"],
                increasing_fillcolor=TRADING_COLORS["up"],
                decreasing_fillcolor=TRADING_COLORS["down"],
                name=symbol,
            )
        ]
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Velas — {symbol}",
        xaxis_rangeslider_visible=False,
        height=360,
    )
    return _apply_axes(fig)


def fig_rsi_panel(df: pd.DataFrame, symbol: str) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)

    if "close" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"],
                name="Close",
                line=dict(color=TRADING_COLORS["accent"], width=1.5),
            ),
            row=1,
            col=1,
        )

    if "rsi_14" in df.columns:
        rsi = df["rsi_14"].astype(float)
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, name="RSI 14", line=dict(color=TRADING_COLORS["warn"], width=1.5)),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color=TRADING_COLORS["down"], opacity=0.6, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=TRADING_COLORS["up"], opacity=0.6, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor=TRADING_COLORS["up"], opacity=0.08, row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor=TRADING_COLORS["down"], opacity=0.08, row=2, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Precio + RSI — {symbol}",
        height=420,
        showlegend=False,
    )
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return _apply_axes(fig)


def fig_macd_panel(df: pd.DataFrame, symbol: str) -> go.Figure:
    if df.empty or "macd_hist" not in df.columns:
        return go.Figure()
    hist = df["macd_hist"].astype(float)
    colors = [TRADING_COLORS["up"] if v >= 0 else TRADING_COLORS["down"] for v in hist]
    fig = go.Figure(
        go.Bar(x=df.index, y=hist, marker_color=colors, name="MACD hist")
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"MACD histograma — {symbol}",
        height=220,
        bargap=0.1,
    )
    return _apply_axes(fig)


def fig_weight_history(hist: pd.DataFrame, symbol: str | None = None) -> go.Figure:
    if hist.empty or "session_date" not in hist.columns:
        return go.Figure()
    df = hist.copy()
    df["session_date"] = pd.to_datetime(df["session_date"])
    if symbol:
        df = df[df["symbol"] == symbol]
    pivot = df.pivot_table(
        index="session_date", columns="symbol", values="target_weight", aggfunc="last"
    )
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[col].astype(float) * 100,
                mode="lines+markers",
                name=str(col),
                line=dict(width=2),
            )
        )
    fig.update_layout(
        **LAYOUT_BASE,
        title="Evolución de pesos sugeridos (%)" if symbol is None else f"Historial peso — {symbol}",
        yaxis_title="Peso %",
        height=360,
        legend=dict(orientation="h", y=1.12),
    )
    return _apply_axes(fig)
