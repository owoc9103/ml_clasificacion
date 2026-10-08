"""Estilos y componentes visuales tipo terminal de trading."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

CHART_GUIDES: dict[str, dict[str, str]] = {
    "top5": {
        "title": "Top 5 sugerido",
        "body": """
- **Rank (#1, #2…):** orden de preferencia **relativa** hoy; #1 es el activo con mayor peso sugerido.
- **Porcentaje:** cuánto del capital *podrías* dedicar a ese ETF si armaras la cartera del modelo.
- **USD:** mismo concepto en dólares, usando el capital de referencia de la barra lateral.
- **Señal:** fuerza combinada (modelo + reglas técnicas). Más alto = más inclinación a tener exposición.
- **ML %:** confianza del modelo en un escenario favorable; **no** es “probabilidad de ganar seguro”.
        """,
    },
    "treemap": {
        "title": "Treemap de cartera",
        "body": """
- Cada **rectángulo** es un ETF; **cuanto más grande**, mayor **peso sugerido** en cartera.
- El **color** va de rojo/ámbar a verde según la **convicción ML** (más verde = mayor probabilidad según el modelo).
- Sirve para ver de un vistazo **dónde concentraría** el sistema el capital; tú decides si te parece prudente.
        """,
    },
    "lollipop": {
        "title": "Lollipop de pesos",
        "body": """
- Es una **barra horizontal** por ticker: la bolita marca el **% sugerido** de inversión.
- Comparar longitudes es más fácil que leer muchos números: los tickers “largos” son los preferidos hoy.
- Si todos son muy parecidos, la estrategia está **repartiendo**; si uno domina, hay **concentración**.
        """,
    },
    "heatmap": {
        "title": "Heatmap de señales",
        "body": """
- **Filas:** tipos de señal (ML, momentum, mean-reversion…). **Columnas:** cada ticker.
- El color indica **intensidad relativa** (no el valor exacto): más verde = más fuerte *frente a los demás* ese día.
- Útil para ver **coherencia**: un ticker muy verde en ML y en “Señal” refuerza la idea; colores mixtos = señales contradictorias.
        """,
    },
    "candlestick": {
        "title": "Velas japonesas",
        "body": """
- Cada **vela** = un día de trading. **Verde** = el cierre fue **mayor** que la apertura (día alcista). **Rojo** = cierre **menor** (día bajista).
- La **mecha** muestra el máximo y mínimo del día: mechas largas = mucha volatilidad intradía.
- Mira la **tendencia reciente** (velas subiendo o bajando), no una sola vela: el modelo usa contexto, tú también.
- Esto **no** dice “compra ahora”; solo te da **contexto visual** del precio antes de decidir.
        """,
    },
    "gauge": {
        "title": "Gauge de convicción ML",
        "body": """
- El **medidor** muestra la probabilidad ML (0–100 %) de escenario favorable según el entrenamiento histórico.
- La **marca en 45 %** es el umbral interno del sistema: por debajo, el momentum ML se considera débil.
- **Delta** compara con ese 45 %: positivo = por encima del mínimo que usa el motor.
- Un gauge alto **no obliga** a operar; combínalo con velas, RSI y tu tolerancia al riesgo.
        """,
    },
    "rsi_panel": {
        "title": "Precio + RSI",
        "body": """
- Arriba: **precio de cierre** diario. Abajo: **RSI (14)** — oscilador entre 0 y 100.
- **Zona verde (0–30):** mercado “sobrevendido” (caída fuerte); a veces buscan rebotes, pero puede seguir cayendo.
- **Zona roja (70–100):** “sobrecomprado” (subida fuerte); puede corregir o seguir en tendencia.
- El RSI **no predice** el futuro solo; ayuda a entender si la señal ML va **a favor o contra** el movimiento reciente.
        """,
    },
    "macd": {
        "title": "MACD histograma",
        "body": """
- Barras **verdes:** momentum alcista (MACD por encima de su señal). **Rojas:** momentum bajista.
- Barras **más altas** = fuerza del movimiento; cuando se **achican**, la tendencia puede estar perdiendo fuel.
- Confirma o cuestiona el ranking: un #1 con MACD rojo reciente es una señal **mixta**, no un error del modelo.
        """,
    },
    "ranking_table": {
        "title": "Tabla ranking",
        "body": """
- **Rank:** posición sugerida. **Señal:** score final 0–1. **ML prob.:** salida del clasificador.
- **Mom. ML / Mean rev.:** mezcla 30 % / 70 % aprox. (momentum del modelo vs reglas tipo RSI).
- **Peso obj. %:** fracción de cartera; **USD sug.:** monto en dólares de referencia.
        """,
    },
    "weight_history": {
        "title": "Historial de pesos",
        "body": """
- Cada **línea** es un ETF; el eje vertical es el **% sugerido** en días pasados en que guardaste el briefing.
- Si una línea **sube** varios días seguidos, el modelo fue aumentando preferencia; si **baja**, la fue reduciendo.
- Sirve para ver **estabilidad** de la idea: cambios bruscos diarios pueden indicar mercado volátil o datos recién actualizados.
        """,
    },
    "general": {
        "title": "Reglas de oro (principiantes)",
        "body": """
1. Todo aquí es **sugerencia académica**, no asesoría financiera personalizada.  
2. **Paper trading** primero; nunca operes live por impulso.  
3. Horario útil USA: aprox. **9:30–16:00 ET** (Nueva York).  
4. Si gráfico y tabla **no coinciden** con tu criterio, **no operes** — esperar también es decisión.  
5. Actualiza datos (`fetch` + `dataset`) antes del briefing para no decidir con información vieja.
        """,
    },
}


def render_chart_guide(slug: str, *, expanded: bool = False) -> None:
    """Muestra ayuda interpretativa bajo cada bloque de gráficos."""
    guide = CHART_GUIDES.get(slug)
    if not guide:
        return
    with st.expander(f"💡 Cómo interpretar: {guide['title']}", expanded=expanded):
        st.markdown(guide["body"].strip())


def render_full_guide_page() -> None:
    """Pestaña con todas las guías juntas."""
    st.markdown("#### Guía para leer el briefing sin ser experto")
    st.caption("Lectura recomendada la primera vez; luego usa los desplegables junto a cada gráfico.")
    for slug in (
        "general",
        "top5",
        "treemap",
        "lollipop",
        "heatmap",
        "candlestick",
        "gauge",
        "rsi_panel",
        "macd",
        "ranking_table",
        "weight_history",
    ):
        guide = CHART_GUIDES[slug]
        with st.expander(guide["title"], expanded=(slug == "general")):
            st.markdown(guide["body"].strip())


INDICATOR_HELP = {
    "ind_rsi_2": ("RSI (2)", "Momentum muy corto. <30 suele leerse sobrevendido."),
    "ind_rsi_14": ("RSI (14)", "Clásico de mean-reversion. Contexto de sobrecompra/venta."),
    "ind_ema20_z": ("EMA20 Z", "Distancia normalizada vs media. Negativo = precio bajo la media."),
    "ind_r_5": ("Ret. 5d", "Rendimiento reciente a 5 sesiones."),
    "ind_r_20": ("Ret. 20d", "Tendencia del último mes aprox."),
    "ind_macd_hist": ("MACD hist.", "Fuerza del cruce MACD. Positivo favorece momentum."),
    "ind_adx_14": ("ADX", "Fuerza de tendencia (no dirección). >25 tendencia clara."),
    "ind_sharpe_20": ("Sharpe 20", "Retorno ajustado por vol reciente."),
    "ind_%b": ("Bollinger %B", "Posición dentro de bandas. Cerca de 0 = banda inferior."),
    "ind_vol_ratio": ("Vol ratio", "Volumen vs media. Picos = participación."),
    "ind_atr_14": ("ATR", "Rango/volatilidad absoluta del activo."),
    "ind_close": ("Precio", "Último cierre en el dataset (referencia)."),
}


def inject_trading_css() -> None:
    st.markdown(
        """
<style>
    .block-container { padding-top: 1.2rem; max-width: 1400px; }
    .trade-header {
        background: linear-gradient(135deg, #0d3d32 0%, #0b0f14 45%, #1a2744 100%);
        border: 1px solid #1e3a34;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .trade-header h1 { margin: 0; font-size: 1.75rem; color: #f0f6ff; letter-spacing: -0.02em; }
    .trade-header p { margin: 0.35rem 0 0; color: #8b9cb3; font-size: 0.95rem; }
    .step-box {
        background: #121820;
        border-left: 3px solid #00c896;
        padding: 0.65rem 0.85rem;
        margin: 0.45rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #c5d0de;
    }
    .pick-card {
        background: linear-gradient(180deg, #151c28 0%, #10151d 100%);
        border: 1px solid #243044;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        min-height: 148px;
    }
    .pick-rank {
        font-size: 0.72rem; font-weight: 700; color: #00c896;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
    .pick-symbol { font-size: 1.65rem; font-weight: 800; color: #fff; margin: 0.1rem 0; }
    .pick-meta { font-size: 0.82rem; color: #94a3b8; }
    .pick-weight { font-size: 1.25rem; font-weight: 700; color: #5eead4; }
    .badge-long {
        display: inline-block; background: rgba(0,200,150,0.15);
        color: #34d399; border: 1px solid rgba(52,211,153,0.35);
        padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600;
    }
    .badge-wait {
        display: inline-block; background: rgba(251,191,36,0.12);
        color: #fbbf24; border: 1px solid rgba(251,191,36,0.35);
        padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background: #121820; border: 1px solid #1e293b; border-radius: 10px; padding: 0.5rem;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header(session_str: str, model: str | None) -> None:
    model_txt = html.escape(model or "—")
    st.markdown(
        f"""
<div class="trade-header">
  <h1>📊 Briefing matutino — decisión de cartera</h1>
  <p>Sesión <strong>{html.escape(session_str)}</strong> · Modelo activo: <strong>{model_txt}</strong> · 
  Solo lectura: <span class="badge-long">NO ejecuta órdenes</span></p>
</div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_playbook() -> None:
    with st.sidebar:
        st.markdown("### 🧭 Qué hacer cada mañana")
        st.markdown(
            """
<div class="step-box"><b>1.</b> Actualiza mercado (terminal):<br/>
<code>fetch_data</code> → <code>make_dataset</code></div>
<div class="step-box"><b>2.</b> Pulsa <b>Actualizar briefing</b> aquí.</div>
<div class="step-box"><b>3.</b> Revisa el <b>Top 5</b> y la convicción ML.</div>
<div class="step-box"><b>4.</b> Contrasta RSI / tendencia en la tabla técnica.</div>
<div class="step-box"><b>5.</b> Tú decides: operar en paper, esperar o pasar.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            "**Convicción ML (`ml_proba`)**  \n"
            "Probabilidad de escenario favorable (etiqueta binaria entrenada).  \n"
            "No es garantía de ganancia."
        )
        st.markdown(
            "**Peso objetivo**  \n"
            "Reparto sugerido tras combinar momentum (ML) y mean-reversion."
        )
        st.markdown("---")
        st.markdown("¿Primera vez? Abre la pestaña **Guía de lectura** arriba.")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.lower().isin(("true", "1", "yes"))
    return series.fillna(False).astype(bool)


def prepare_ok_frame(report: pd.DataFrame) -> pd.DataFrame:
    df = report.copy()
    if "data_ok" in df.columns:
        df = df[_coerce_bool(df["data_ok"])]
    if "rank" in df.columns:
        df = df.sort_values("rank", ascending=True)
    return df


def render_pick_cards(top: pd.DataFrame) -> None:
    cols = st.columns(min(5, len(top)))
    for i, (_, row) in enumerate(top.head(5).iterrows()):
        sym = html.escape(str(row.get("symbol", "—")))
        w = float(row.get("target_weight") or 0) * 100
        usd = float(row.get("suggested_notional_usd") or 0)
        proba = float(row.get("ml_proba") or 0)
        sig = float(row.get("signal") or 0)
        rank = int(row.get("rank") or i + 1)
        badge = "badge-long" if proba >= 0.48 else "badge-wait"
        label = "Alta convicción" if proba >= 0.48 else "Convicción moderada"
        with cols[i]:
            st.markdown(
                f"""
<div class="pick-card">
  <div class="pick-rank">#{rank} sugerido</div>
  <div class="pick-symbol">{sym}</div>
  <div class="pick-weight">{w:.1f}% · ${usd:,.0f}</div>
  <div class="pick-meta">Señal {sig:.2f} · ML {proba:.0%}</div>
  <span class="{badge}">{label}</span>
</div>
                """,
                unsafe_allow_html=True,
            )


def styled_signals_table(df: pd.DataFrame) -> None:
    show = df.copy()
    pct_cols = ["target_weight", "ml_proba", "momo_sig", "meanrev_sig", "signal"]
    for c in pct_cols:
        if c in show.columns:
            if c == "target_weight":
                show[c] = show[c].astype(float)
            elif c == "ml_proba":
                show[c] = show[c].astype(float)

    column_config = {}
    if "target_weight" in show.columns:
        show["target_weight"] = show["target_weight"].astype(float) * 100
        column_config["target_weight"] = st.column_config.ProgressColumn(
            "Peso obj. %", format="%.1f", min_value=0, max_value=35
        )
    if "ml_proba" in show.columns:
        show["ml_proba"] = show["ml_proba"].astype(float) * 100
        column_config["ml_proba"] = st.column_config.ProgressColumn(
            "ML prob. %", format="%.0f", min_value=0, max_value=100
        )
    if "suggested_notional_usd" in show.columns:
        column_config["suggested_notional_usd"] = st.column_config.NumberColumn(
            "USD sug.", format="$%.0f"
        )
    if "signal" in show.columns:
        column_config["signal"] = st.column_config.NumberColumn("Señal", format="%.3f")

    rename = {
        "rank": "Rank",
        "symbol": "Ticker",
        "momo_sig": "Mom. ML",
        "meanrev_sig": "Mean rev.",
    }
    show = show.rename(columns={k: v for k, v in rename.items() if k in show.columns})

    st.dataframe(
        show,
        width="stretch",
        hide_index=True,
        column_config=column_config or None,
    )
