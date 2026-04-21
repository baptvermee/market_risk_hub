"""
Page Streamlit — Bond Pricer
Pricing d'obligations : taux fixe, zéro coupon, amortissable.
Même design que les autres pages (dark trading desk style).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.bond_pricer import (
    generate_cash_flows,
    bond_price,
    clean_dirty_price,
    yield_to_maturity,
    macaulay_duration,
    modified_duration,
    convexity,
    rate_sensitivity_analysis,
    price_yield_curve,
)

# =========================
# Page config & CSS
# =========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0a0f1c 0%, #111827 100%);
        border: 1px solid #1e293b;
        padding: 16px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    div[data-testid="stMetric"] label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #64748b;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    h1 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        border-bottom: 2px solid #34d399;
        padding-bottom: 8px;
    }

    h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.3px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1c 0%, #0f172a 100%);
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
    }

    .trading-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #34d399, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    .tag-premium {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .tag-discount {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .tag-par {
        background-color: #1e3a5f;
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Bond Pricer — Fixed Income")
st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Sidebar — Paramètres
# =========================
st.sidebar.header("PARAMÈTRES")

st.sidebar.markdown("#### Obligation")

bond_type = st.sidebar.selectbox(
    "Type d'obligation",
    ["fixed", "zero", "amortizing"],
    format_func=lambda x: {"fixed": "Taux fixe (in fine)", "zero": "Zéro coupon", "amortizing": "Amortissable"}[x],
)

face_value = st.sidebar.number_input("Nominal", min_value=100.0, value=1000.0, step=100.0)
maturity = st.sidebar.slider("Maturité (années)", 1, 30, 10, 1)

if bond_type != "zero":
    coupon_pct = st.sidebar.slider("Coupon annuel %", 0.0, 15.0, 5.0, 0.25)
    coupon_rate = coupon_pct / 100
else:
    coupon_rate = 0.0
    st.sidebar.info("Zéro coupon : pas de coupon")

frequency = st.sidebar.selectbox(
    "Fréquence des coupons",
    [1, 2, 4],
    index=1,
    format_func=lambda x: {1: "Annuel", 2: "Semestriel", 4: "Trimestriel"}[x],
)

st.sidebar.markdown("#### Marché")
ytm_pct = st.sidebar.slider("Yield to Maturity %", 0.1, 15.0, 4.0, 0.1)
ytm = ytm_pct / 100

st.sidebar.markdown("#### Intérêts courus")
days_since = st.sidebar.slider("Jours depuis dernier coupon", 0, 182, 0, 1)

# =========================
# Calculs
# =========================
result = clean_dirty_price(face_value, coupon_rate, maturity, ytm, frequency, bond_type, days_since)
d_mac = macaulay_duration(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
d_mod = modified_duration(face_value, coupon_rate, maturity, ytm, frequency, bond_type)
conv = convexity(face_value, coupon_rate, maturity, ytm, frequency, bond_type)

# Premium / Discount / Par
if abs(result["dirty_price"] - face_value) < 0.01:
    price_status = "PAR"
    status_class = "tag-par"
elif result["dirty_price"] > face_value:
    price_status = "PREMIUM"
    status_class = "tag-premium"
else:
    price_status = "DISCOUNT"
    status_class = "tag-discount"

bond_type_labels = {"fixed": "Taux fixe", "zero": "Zéro coupon", "amortizing": "Amortissable"}

# =========================
# Ligne 1 — KPI Prix
# =========================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Dirty Price", f"{result['dirty_price']:.4f}")
col2.metric("Clean Price", f"{result['clean_price']:.4f}")
col3.metric("Accrued Interest", f"{result['accrued_interest']:.4f}")
col4.metric("Prix / Nominal", f"{result['dirty_price']/face_value*100:.2f}%")

st.markdown(
    f'<span class="{status_class}">{price_status}</span> — '
    f'{bond_type_labels[bond_type]} — Nominal {face_value:.0f} — '
    f'Coupon {coupon_rate:.2%} — Maturité {maturity}ans — YTM {ytm:.2%}',
    unsafe_allow_html=True,
)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Ligne 2 — KPI Risque
# =========================
st.subheader("Mesures de risque")

col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)

col_r1.metric("Duration Macaulay", f"{d_mac:.4f} ans" if not np.isnan(d_mac) else "N/A")
col_r2.metric("Duration Modifiée", f"{d_mod:.4f}" if not np.isnan(d_mod) else "N/A")
col_r3.metric("Convexité", f"{conv:.4f}" if not np.isnan(conv) else "N/A")

# DV01 = changement de prix pour 1bp de hausse de taux
# DV01 ≈ D_mod × Prix × 0.0001
dv01 = d_mod * result["dirty_price"] * 0.0001 if not np.isnan(d_mod) else np.nan
col_r4.metric("DV01", f"{dv01:.4f}" if not np.isnan(dv01) else "N/A")

# BPV = basis point value (même chose que DV01 mais nommé différemment)
col_r5.metric("YTM", f"{ytm:.2%}")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graph 1 — Échéancier des flux
# =========================
st.subheader("Échéancier des flux")

cf = result["cash_flows"]

fig_cf = go.Figure()

# Barres des coupons
fig_cf.add_trace(go.Bar(
    x=cf["times"], y=cf["coupons"],
    name="Coupons",
    marker_color="#22d3ee",
    opacity=0.8,
))

# Barres du principal
fig_cf.add_trace(go.Bar(
    x=cf["times"], y=cf["principals"],
    name="Remboursement principal",
    marker_color="#a78bfa",
    opacity=0.8,
))

fig_cf.update_layout(
    barmode="stack",
    title=f"Échéancier — {bond_type_labels[bond_type]}",
    xaxis_title="Années",
    yaxis_title="Montant",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=400,
)

st.plotly_chart(fig_cf, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graph 2 — Courbe Prix vs Taux
# =========================
st.subheader("Courbe Prix — Taux")

st.markdown(
    "> **La relation fondamentale du fixed income.** "
    "Le prix est une fonction décroissante et convexe du taux. "
    "La tangente au point actuel = -Duration modifiée × Prix."
)

py_curve = price_yield_curve(face_value, coupon_rate, maturity, frequency, bond_type)

fig_py = go.Figure()

# Courbe prix-taux
fig_py.add_trace(go.Scatter(
    x=py_curve["ytm_values"] * 100,
    y=py_curve["prices"],
    mode="lines",
    name="Prix exact",
    line=dict(color="#34d399", width=2.5),
))

# Point actuel
fig_py.add_trace(go.Scatter(
    x=[ytm * 100],
    y=[result["dirty_price"]],
    mode="markers",
    name="Position actuelle",
    marker=dict(color="#f43f5e", size=12, symbol="diamond"),
))

# Tangente (approximation duration)
if not np.isnan(d_mod):
    ytm_tangent = np.linspace(max(0.001, ytm - 0.03), ytm + 0.03, 50)
    tangent_prices = result["dirty_price"] * (1 - d_mod * (ytm_tangent - ytm))

    fig_py.add_trace(go.Scatter(
        x=ytm_tangent * 100,
        y=tangent_prices,
        mode="lines",
        name="Approx. Duration (tangente)",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
    ))

# Ligne du pair
fig_py.add_hline(
    y=face_value, line_dash="dot", line_color="#64748b",
    annotation_text=f"Par ({face_value:.0f})",
)

fig_py.update_layout(
    title="Prix vs Yield — la convexité est visible dans la courbure",
    xaxis_title="Yield to Maturity (%)",
    yaxis_title="Prix",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=500,
)

st.plotly_chart(fig_py, use_container_width=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graph 3 — Analyse de sensibilité
# =========================
st.subheader("Analyse de sensibilité aux taux")

st.markdown(
    "> On compare le prix exact avec l'approximation par la duration seule, "
    "et par duration + convexité. La convexité corrige l'erreur pour les gros chocs."
)

sensitivity = rate_sensitivity_analysis(
    face_value, coupon_rate, maturity, ytm, frequency, bond_type,
)

shocks_df = pd.DataFrame(sensitivity["shocks"])

fig_sens = go.Figure()

# Prix exact
fig_sens.add_trace(go.Scatter(
    x=shocks_df["shock_bps"], y=shocks_df["exact_price"],
    mode="markers+lines", name="Prix exact",
    line=dict(color="#34d399", width=2.5),
    marker=dict(size=8),
))

# Approximation duration
fig_sens.add_trace(go.Scatter(
    x=shocks_df["shock_bps"], y=shocks_df["duration_approx"],
    mode="lines", name="Approx. Duration",
    line=dict(color="#f59e0b", width=2, dash="dash"),
))

# Approximation duration + convexité
fig_sens.add_trace(go.Scatter(
    x=shocks_df["shock_bps"], y=shocks_df["full_approx"],
    mode="lines", name="Approx. Duration + Convexité",
    line=dict(color="#a78bfa", width=2, dash="dot"),
))

fig_sens.update_layout(
    title="Impact d'un choc de taux — Duration vs Duration + Convexité",
    xaxis_title="Choc de taux (bps)",
    yaxis_title="Prix",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=450,
)

st.plotly_chart(fig_sens, use_container_width=True)

# Tableau de sensibilité
st.markdown("#### Détail des chocs")

display_sens = shocks_df[["shock_bps", "new_ytm", "exact_price", "exact_change_pct", "duration_error", "full_error"]].copy()
display_sens.columns = ["Choc (bps)", "Nouveau YTM", "Prix exact", "Variation %", "Erreur Duration", "Erreur Dur+Conv"]
display_sens["Nouveau YTM"] = display_sens["Nouveau YTM"].apply(lambda x: f"{x:.2%}")
display_sens["Prix exact"] = display_sens["Prix exact"].apply(lambda x: f"{x:.4f}")
display_sens["Variation %"] = display_sens["Variation %"].apply(lambda x: f"{x:+.2f}%")
display_sens["Erreur Duration"] = display_sens["Erreur Duration"].apply(lambda x: f"{x:+.4f}")
display_sens["Erreur Dur+Conv"] = display_sens["Erreur Dur+Conv"].apply(lambda x: f"{x:+.4f}")

st.dataframe(display_sens, use_container_width=True, hide_index=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Graph 4 — Comparaison des 3 types
# =========================
st.subheader("Comparaison — Fixe vs Zéro Coupon vs Amortissable")

st.markdown(
    "> À mêmes paramètres, les 3 types d'obligations ont des profils très différents. "
    "Le zéro coupon a la duration la plus longue (= le plus risqué aux taux). "
    "L'amortissable a la duration la plus courte (remboursement progressif)."
)

# Calcul pour les 3 types
types = ["fixed", "zero", "amortizing"]
labels = {"fixed": "Taux fixe", "zero": "Zéro coupon", "amortizing": "Amortissable"}
colors = {"fixed": "#22d3ee", "zero": "#a78bfa", "amortizing": "#f59e0b"}

comparison = []
for bt in types:
    cr = coupon_rate if bt != "zero" else 0.0
    bp = bond_price(face_value, cr, maturity, ytm, frequency, bt)
    dm = macaulay_duration(face_value, cr, maturity, ytm, frequency, bt)
    dmod = modified_duration(face_value, cr, maturity, ytm, frequency, bt)
    cv = convexity(face_value, cr, maturity, ytm, frequency, bt)

    comparison.append({
        "Type": labels[bt],
        "Prix": bp["dirty_price"],
        "Duration Macaulay": dm,
        "Duration Modifiée": dmod,
        "Convexité": cv,
    })

comp_df = pd.DataFrame(comparison)

# Graphique comparatif : courbes prix-taux
fig_comp = go.Figure()

for bt in types:
    cr = coupon_rate if bt != "zero" else 0.0
    curve = price_yield_curve(face_value, cr, maturity, frequency, bt)
    fig_comp.add_trace(go.Scatter(
        x=curve["ytm_values"] * 100,
        y=curve["prices"],
        mode="lines",
        name=labels[bt],
        line=dict(color=colors[bt], width=2),
    ))

fig_comp.add_vline(x=ytm * 100, line_dash="dot", line_color="#f43f5e", annotation_text=f"YTM={ytm:.1%}")
fig_comp.add_hline(y=face_value, line_dash="dot", line_color="#64748b", annotation_text=f"Par ({face_value:.0f})")

fig_comp.update_layout(
    title="Courbes Prix-Taux — les 3 types d'obligations",
    xaxis_title="Yield (%)",
    yaxis_title="Prix",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,15,28,0.8)",
    font=dict(family="JetBrains Mono"),
    height=500,
)

st.plotly_chart(fig_comp, use_container_width=True)

# Tableau comparatif
st.markdown("#### Métriques comparées")

display_comp = comp_df.copy()
display_comp["Prix"] = display_comp["Prix"].apply(lambda x: f"{x:.4f}")
display_comp["Duration Macaulay"] = display_comp["Duration Macaulay"].apply(lambda x: f"{x:.4f} ans" if not np.isnan(x) else "N/A")
display_comp["Duration Modifiée"] = display_comp["Duration Modifiée"].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
display_comp["Convexité"] = display_comp["Convexité"].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

st.dataframe(display_comp, use_container_width=True, hide_index=True)

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# YTM inverse
# =========================
st.subheader("📡 Yield to Maturity — Reverse Engineering")

st.markdown(
    "Entre un prix de marché et le pricer retrouve le YTM implicite. "
    "C'est l'équivalent fixed income de la vol implicite pour les options."
)

ytm_col1, ytm_col2 = st.columns(2)

with ytm_col1:
    market_price_input = st.number_input(
        "Prix de marché observé",
        min_value=1.0,
        value=round(result["dirty_price"], 2),
        step=1.0,
    )

with ytm_col2:
    implied_ytm = yield_to_maturity(market_price_input, face_value, coupon_rate, maturity, frequency, bond_type)

    if np.isnan(implied_ytm):
        st.error("Pas de convergence — le prix est peut-être hors bornes.")
    else:
        st.metric("YTM implicite", f"{implied_ytm:.4%}")

        diff_ytm = implied_ytm - ytm
        if abs(diff_ytm) < 0.0001:
            st.success("Le YTM implicite correspond au YTM du pricer — cohérent.")
        elif diff_ytm > 0:
            st.warning(f"YTM implicite ({implied_ytm:.2%}) > votre YTM ({ytm:.2%}) → le marché price un rendement plus élevé (prix plus bas).")
        else:
            st.info(f"YTM implicite ({implied_ytm:.2%}) < votre YTM ({ytm:.2%}) → le marché price un rendement plus faible (prix plus haut).")

st.markdown('<div class="trading-divider"></div>', unsafe_allow_html=True)

# =========================
# Tableau récapitulatif
# =========================
st.subheader("Récapitulatif")

recap = pd.DataFrame({
    "Paramètre": [
        "Type", "Nominal", "Coupon", "Maturité", "Fréquence", "YTM",
        "Dirty Price", "Clean Price", "Accrued Interest",
        "Duration Macaulay", "Duration Modifiée", "Convexité", "DV01",
    ],
    "Valeur": [
        bond_type_labels[bond_type], f"{face_value:.0f}", f"{coupon_rate:.2%}",
        f"{maturity} ans", f"{frequency}x/an", f"{ytm:.2%}",
        f"{result['dirty_price']:.4f}", f"{result['clean_price']:.4f}", f"{result['accrued_interest']:.4f}",
        f"{d_mac:.4f} ans" if not np.isnan(d_mac) else "N/A",
        f"{d_mod:.4f}" if not np.isnan(d_mod) else "N/A",
        f"{conv:.4f}" if not np.isnan(conv) else "N/A",
        f"{dv01:.4f}" if not np.isnan(dv01) else "N/A",
    ],
})

st.dataframe(recap, use_container_width=True, hide_index=True)