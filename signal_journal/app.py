"""
Briefing matutino — Streamlit (sin ejecutar órdenes).
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from src.charts_trading import (
    fig_allocation_treemap,
    fig_candlestick,
    fig_conviction_gauge,
    fig_macd_panel,
    fig_rsi_panel,
    fig_signal_heatmap,
    fig_weight_history,
    fig_weight_lollipop,
)
from src.market_data import load_price_history
from src.signals import build_daily_report, load_symbols
from src.storage import append_snapshot, has_run_for_date, latest_session, load_history
from src.ui_theme import (
    INDICATOR_HELP,
    inject_trading_css,
    prepare_ok_frame,
    render_chart_guide,
    render_full_guide_page,
    render_header,
    render_pick_cards,
    sidebar_playbook,
    styled_signals_table,
)

st.set_page_config(
    page_title="Briefing Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_trading_css()
sidebar_playbook()

tab_desk, tab_charts, tab_tech, tab_hist, tab_guide = st.tabs(
    [
        "🎯 Mesa de decisión",
        "📈 Gráficos",
        "📐 Radar técnico",
        "🗓 Historial",
        "💡 Guía de lectura",
    ]
)

with st.sidebar:
    session = st.date_input("Fecha de sesión", value=datetime.now(timezone.utc).date())
    notional = st.number_input("Capital referencia (USD)", value=100_000.0, step=10_000.0)
    exclude_vix = st.checkbox("Excluir ^VIX del ranking", value=True)
    symbols = load_symbols(exclude_vix=exclude_vix)
    chart_bars = st.slider("Velas (sesiones)", min_value=30, max_value=180, value=90)

    refresh = st.button("🔄 Actualizar briefing del día", type="primary", width="stretch")

if refresh:
    with st.spinner("Sincronizando modelo e indicadores…"):
        try:
            report = build_daily_report(
                symbols,
                session_date=session,
                notional_account=notional,
            )
            append_snapshot(report, replace_session=has_run_for_date(session))
            st.session_state["last_report"] = report
            st.toast("Briefing listo. Revisa el Top 5 antes de operar.", icon="✅")
        except Exception as exc:
            st.error(f"No se pudo calcular el briefing: {exc}")

report = st.session_state.get("last_report")
if report is None:
    latest = latest_session()
    if latest is not None:
        report = latest

model_name = None
if report is not None and not report.empty and "model" in report.columns:
    model_name = str(report["model"].iloc[0])

render_header(str(session), model_name)

if report is None or report.empty:
    st.markdown(
        """
> **Empieza aquí:** en la barra lateral pulsa **Actualizar briefing del día**.  
> Si falla, primero actualiza datos de mercado en tu pipeline quant.
        """
    )
    st.stop()

ok = prepare_ok_frame(report)

with tab_desk:
    if ok.empty:
        st.warning(
            "No hay features para mostrar. Actualiza precios e indicadores en tu pipeline "
            "y vuelve a generar el briefing."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        top1 = ok.iloc[0]
        invested = ok["target_weight"].astype(float).sum() * 100
        c1.metric("Líder del día", str(top1["symbol"]))
        c2.metric("Peso sugerido #1", f"{float(top1['target_weight']) * 100:.1f}%")
        c3.metric("Exposición total sug.", f"{invested:.0f}%")
        c4.metric("Activos con señal", f"{len(ok[ok['signal'].astype(float) > 0.05])}")

        st.markdown("#### 🏆 Concentración sugerida (Top 5)")
        render_chart_guide("top5")
        render_pick_cards(ok)

        g_left, g_right = st.columns([1.1, 1])
        with g_left:
            render_chart_guide("treemap")
            st.plotly_chart(fig_allocation_treemap(ok), width="stretch", key="desk_treemap")
        with g_right:
            render_chart_guide("lollipop")
            st.plotly_chart(fig_weight_lollipop(ok), width="stretch", key="desk_lollipop")

        render_chart_guide("heatmap")
        st.plotly_chart(fig_signal_heatmap(ok), width="stretch", key="desk_heatmap")

        st.markdown("#### 📋 Ranking completo")
        render_chart_guide("ranking_table")
        desk_cols = [
            c
            for c in [
                "rank",
                "symbol",
                "signal",
                "ml_proba",
                "momo_sig",
                "meanrev_sig",
                "target_weight",
                "suggested_notional_usd",
            ]
            if c in ok.columns
        ]
        styled_signals_table(ok[desk_cols])

        st.info(
            "**Decisión manual:** este panel no envía órdenes. Opera en paper solo si confirmas "
            "convicción, horario NY y riesgo."
        )

with tab_charts:
    if ok.empty:
        st.warning("Genera un briefing con datos válidos para ver gráficos de mercado.")
    else:
        st.markdown("#### Velas — líderes del ranking")
        render_chart_guide("candlestick")
        top3 = ok.head(3)["symbol"].tolist()
        cols = st.columns(3)
        for col, sym in zip(cols, top3):
            with col:
                ohlc = load_price_history(str(sym), bars=chart_bars)
                st.plotly_chart(
                    fig_candlestick(ohlc, str(sym)),
                    width="stretch",
                    key=f"charts_candle_{sym}",
                )

        st.markdown("#### Convicción ML (gauge)")
        render_chart_guide("gauge")
        gcols = st.columns(min(3, len(ok.head(3))))
        for col, (_, row) in zip(gcols, ok.head(3).iterrows()):
            sym = str(row["symbol"])
            with col:
                st.plotly_chart(
                    fig_conviction_gauge(
                        sym,
                        float(row.get("ml_proba") or 0),
                        float(row.get("target_weight") or 0) * 100,
                    ),
                    width="stretch",
                    key=f"charts_gauge_{sym}",
                )

        st.markdown("#### Análisis del ticker seleccionado")
        pick = st.selectbox(
            "Ticker",
            ok["symbol"].astype(str).tolist(),
            index=0,
            key="charts_ticker_select",
        )
        hist_df = load_price_history(pick, bars=chart_bars)
        c1, c2 = st.columns([1.4, 1])
        with c1:
            render_chart_guide("rsi_panel")
            st.plotly_chart(fig_rsi_panel(hist_df, pick), width="stretch", key="charts_rsi_panel")
        with c2:
            render_chart_guide("macd")
            st.plotly_chart(fig_macd_panel(hist_df, pick), width="stretch", key="charts_macd_panel")
            st.plotly_chart(
                fig_candlestick(hist_df.tail(40), pick),
                width="stretch",
                key="charts_candle_focus",
            )

with tab_tech:
    st.markdown("#### Lectura técnica por activo")

    ind_cols = [c for c in report.columns if c.startswith("ind_")]
    if ind_cols:
        with st.expander("Glosario de indicadores", expanded=False):
            for col in ind_cols:
                title, desc = INDICATOR_HELP.get(col, (col.replace("ind_", ""), ""))
                st.markdown(f"**{title}** — {desc}")

        tech = ok[["rank", "symbol", "signal", "ml_proba"] + ind_cols].copy()
        pretty = tech.rename(
            columns={c: INDICATOR_HELP.get(c, (c.replace("ind_", ""),))[0] for c in ind_cols}
        )
        st.dataframe(pretty, width="stretch", hide_index=True)

with tab_hist:
    hist = load_history()
    if hist.empty:
        st.markdown("Aún no hay sesiones archivadas. Genera tu primer briefing desde la barra lateral.")
    else:
        dates = sorted(hist["session_date"].astype(str).unique(), reverse=True)
        pick_date = st.selectbox("Sesión archivada", dates, key="hist_session_select")
        part = hist[hist["session_date"].astype(str) == pick_date].copy()
        if "rank" in part.columns:
            part = part.sort_values("rank")

        render_chart_guide("weight_history")
        st.plotly_chart(fig_weight_history(hist), width="stretch", key="hist_weights_all")

        sym_hist = st.selectbox(
            "Evolución por ticker",
            sorted(hist["symbol"].astype(str).unique()),
            key="hist_symbol_select",
        )
        st.plotly_chart(
            fig_weight_history(hist, symbol=sym_hist),
            width="stretch",
            key="hist_weights_symbol",
        )

        st.dataframe(part, width="stretch", hide_index=True)

with tab_guide:
    render_full_guide_page()
