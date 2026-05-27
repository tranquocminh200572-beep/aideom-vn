"""
Bài 6: TOPSIS xếp hạng 6 vùng kinh tế Việt Nam theo mức độ ưu tiên đầu tư AI
Bao gồm: Entropy weights, phân tích độ nhạy trọng số AI Readiness
FIX: Toàn bộ biểu đồ dùng Plotly - không dùng matplotlib để tránh lỗi font tiếng Việt
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

CRITERIA = [
    'grdp_per_capita_million_VND', 'fdi_registered_billion_USD',
    'digital_index_0_100', 'ai_readiness_0_100',
    'trained_labor_pct', 'rd_intensity_pct',
    'internet_penetration_pct', 'gini_coef'
]
IS_BENEFIT = [True, True, True, True, True, True, True, False]
W_EXPERT = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

REGION_SHORT = {
    'Trung du miền núi phía Bắc': 'Trung du MN Bắc',
    'Đồng bằng sông Hồng':        'ĐB sông Hồng',
    'Bắc Trung Bộ + DH Trung Bộ': 'BTB + DHMT',
    'Tây Nguyên':                  'Tây Nguyên',
    'Đông Nam Bộ':                 'Đông Nam Bộ',
    'Đồng bằng sông Cửu Long':    'ĐB sông Cửu Long',
}

def shorten(name):
    return REGION_SHORT.get(name, name)

REGION_NAMES = [
    'Trung du miền núi phía Bắc',
    'Đồng bằng sông Hồng',
    'Bắc Trung Bộ + DH Trung Bộ',
    'Tây Nguyên',
    'Đông Nam Bộ',
    'Đồng bằng sông Cửu Long'
]

DATA_FALLBACK = np.array([
    [57.0,  3.5,  38, 22, 21.5, 0.18, 72, 0.405],
    [152.3, 20.0, 78, 68, 36.8, 0.85, 92, 0.358],
    [87.5,  8.2,  55, 40, 27.5, 0.32, 84, 0.372],
    [68.9,  0.8,  32, 18, 18.2, 0.15, 68, 0.412],
    [158.9, 18.5, 82, 75, 42.5, 0.78, 94, 0.385],
    [80.5,  2.1,  48, 30, 16.8, 0.22, 78, 0.392],
])

COLORS = ['#1976D2','#388E3C','#F57C00','#7B1FA2','#D32F2F','#0097A7']


def load_data():
    """Load data từ xlsx. Luôn dùng REGION_NAMES hardcode để tránh lỗi encoding."""
    for search_path in [
        Path('data/vietnam_regions_2024.xlsx'),
        Path('/mnt/project/vietnam_regions_2024.xlsx'),
    ]:
        if search_path.exists():
            try:
                df = pd.read_excel(search_path)
                X = df[CRITERIA].values.astype(float)
                # Dùng REGION_NAMES hardcode thay vì đọc từ file (tránh lỗi encoding Excel)
                return X, REGION_NAMES
            except Exception:
                pass
    return DATA_FALLBACK.copy(), REGION_NAMES


def topsis(X, w, is_benefit):
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms = np.where(norms == 0, 1e-12, norms)
    R = X / norms
    V = R * w
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg  = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg  = np.sqrt(((V - A_neg)  ** 2).sum(axis=1))
    denom  = np.where(S_star + S_neg == 0, 1e-12, S_star + S_neg)
    return S_neg / denom


def entropy_weights(X):
    X_pos = np.abs(X) + 1e-12
    P = X_pos / X_pos.sum(axis=0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = np.where(1 - E < 0, 0, 1 - E)
    return d / d.sum() if d.sum() > 0 else np.ones(len(d)) / len(d)


def plot_ranking_bar(scores_exp, scores_ent, region_names):
    """Biểu đồ so sánh xếp hạng - Plotly."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Trong so chuyen gia (w_AI=0.20)',
                                        'Trong so Entropy (khach quan)'])
    for col_idx, (scores, title) in enumerate(
        zip([scores_exp, scores_ent], ['Chuyen gia', 'Entropy']), start=1
    ):
        order      = np.argsort(scores)[::-1]
        names_ord  = [shorten(region_names[i]) for i in order]
        scores_ord = scores[order]
        bar_colors = [COLORS[i % len(COLORS)] for i in order]

        fig.add_trace(go.Bar(
            x=scores_ord[::-1],
            y=names_ord[::-1],
            orientation='h',
            marker_color=bar_colors[::-1],
            text=[f'#{6-j} {v:.3f}' if (6-j-1) < 3 else f'{v:.3f}'
                  for j, v in enumerate(scores_ord[::-1])],
            textposition='outside',
            name=title,
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_xaxes(title_text='Diem TOPSIS (C*)', range=[0, 1.2])
    fig.update_layout(height=420, margin=dict(t=60, b=20))
    return fig


def plot_entropy_bar(w_entropy, crit_labels):
    """Biểu đồ trọng số Entropy - Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=crit_labels, y=w_entropy,
        marker_color='#1976D2',
        text=[f'{w:.3f}' for w in w_entropy],
        textposition='outside',
    ))
    fig.add_hline(y=1/8, line_dash='dash', line_color='red',
                  annotation_text='Dong deu (1/8)')
    fig.update_layout(
        title='Trong so Entropy cho 8 tieu chi',
        xaxis_title='Tieu chi', yaxis_title='Trong so',
        height=350, margin=dict(t=50, b=80),
        xaxis_tickangle=-20,
    )
    return fig


def plot_sensitivity(X, region_names):
    """Heatmap độ nhạy w_AI - Plotly."""
    w_ai_range  = np.arange(0.05, 0.45, 0.05)
    n_r         = len(region_names)
    rank_matrix = np.zeros((n_r, len(w_ai_range)))

    for j, w_ai in enumerate(w_ai_range):
        w = W_EXPERT.copy()
        w[3] = w_ai
        other_idx = [i for i in range(len(w)) if i != 3]
        w_other   = w[other_idx] / w[other_idx].sum() * (1 - w_ai)
        w[other_idx] = w_other
        scores = topsis(X, w, IS_BENEFIT)
        rank_matrix[:, j] = n_r - scores.argsort().argsort()

    short_names = [shorten(n) for n in region_names]
    x_labels    = [f'{w:.2f}' for w in w_ai_range]

    fig = go.Figure(data=go.Heatmap(
        z=rank_matrix,
        x=x_labels,
        y=short_names,
        colorscale='RdYlGn_r',
        zmin=1, zmax=6,
        text=rank_matrix.astype(int),
        texttemplate='%{text}',
        textfont=dict(size=13, color='white'),
        colorbar=dict(title='Hang'),
    ))
    fig.update_layout(
        title='Phan tich do nhay: Hang vung theo w_AI',
        xaxis_title='Trong so w_AI (AI Readiness)',
        yaxis_title='Vung',
        height=380, margin=dict(t=50, b=50),
    )
    return fig


def plot_radar(X, w, region_names):
    """Radar chart - Plotly."""
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
    X_norm[:, 7] = 1 - X_norm[:, 7]

    crit_labels = ['GRDP/nguoi', 'FDI', 'Digital', 'AI Ready',
                   'LD dao tao', 'R&D', 'Internet', 'Binh dang']
    short_names = [shorten(n) for n in region_names]

    fig = go.Figure()
    for i, (vals, color, name) in enumerate(zip(X_norm, COLORS, short_names)):
        v = vals.tolist() + [vals[0]]
        c = crit_labels + [crit_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=v, theta=c,
            fill='toself', opacity=0.2,
            line=dict(color=color, width=2),
            name=name,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='So sanh 6 vung theo 8 tieu chi (chuan hoa [0,1])',
        height=500,
        legend=dict(x=1.05, y=1),
    )
    return fig


def render():
    st.header("Bai 6: TOPSIS Xep hang 6 Vung Kinh te theo Uu tien Dau tu AI")

    with st.expander("Ly thuyet TOPSIS", expanded=False):
        st.markdown("""
        **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution):
        1. Chuan hoa vector
        2. Ma tran co trong so
        3. Ly tuong duong/am
        4. Khoang cach Euclide
        5. He so gan gui C* (cang gan 1 cang tot)
        """)

    X, region_names = load_data()

    st.subheader("Dieu chinh trong so chuyen gia")
    st.caption("Tong trong so phai = 1.0 (tu dong chuan hoa)")

    cols = st.columns(4)
    w_inputs  = []
    default_w = [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10]
    crit_short = ['GRDP/nguoi', 'FDI', 'Digital Index', 'AI Readiness',
                  'LD dao tao', 'R&D/GRDP', 'Internet', 'Gini']
    for i, (label, dw) in enumerate(zip(crit_short, default_w)):
        with cols[i % 4]:
            w_inputs.append(st.number_input(label, 0.01, 0.60, dw, 0.01, key=f'w6_{i}'))

    w_user = np.array(w_inputs)
    w_user = w_user / w_user.sum()

    if st.button("Chay TOPSIS & Phan tich", type="primary"):
        scores_expert  = topsis(X, w_user, IS_BENEFIT)
        w_entropy      = entropy_weights(X)
        scores_entropy = topsis(X, w_entropy, IS_BENEFIT)

        st.subheader("Ket qua TOPSIS")
        ranks_exp = len(region_names) - scores_expert.argsort().argsort()
        ranks_ent = len(region_names) - scores_entropy.argsort().argsort()
        short     = [shorten(n) for n in region_names]

        df_res = pd.DataFrame({
            'Vung':           short,
            'C* (chuyen gia)': np.round(scores_expert, 4),
            'Hang (CG)':      ranks_exp,
            'C* (Entropy)':   np.round(scores_entropy, 4),
            'Hang (Entropy)': ranks_ent,
            'Thay doi hang':  ranks_exp - ranks_ent,
        }).sort_values('Hang (CG)')
        st.dataframe(df_res, use_container_width=True)

        st.subheader("Trong so Entropy (khach quan)")
        st.plotly_chart(plot_entropy_bar(w_entropy, crit_short),
                        use_container_width=True)

        st.subheader("So sanh xep hang: Chuyen gia vs Entropy")
        st.plotly_chart(plot_ranking_bar(scores_expert, scores_entropy, region_names),
                        use_container_width=True)

        st.subheader("Radar chart 6 vung")
        st.plotly_chart(plot_radar(X, w_user, region_names),
                        use_container_width=True)

        st.subheader("Phan tich do nhay: Hang theo w_AI Readiness")
        st.plotly_chart(plot_sensitivity(X, region_names),
                        use_container_width=True)

        top3_exp   = np.argsort(scores_expert)[::-1][:3]
        top3_names = [short[i] for i in top3_exp]
        st.info(f"Top 3 theo trong so chuyen gia: {', '.join(top3_names)}")
