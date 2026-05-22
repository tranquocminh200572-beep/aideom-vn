"""
Bài 6: TOPSIS xếp hạng 6 vùng kinh tế Việt Nam theo mức độ ưu tiên đầu tư AI
Bao gồm: Entropy weights, phân tích độ nhạy trọng số AI Readiness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────
# DỮ LIỆU (load từ file hoặc fallback hardcode)
# ─────────────────────────────────────────────
CRITERIA = [
    'grdp_per_capita_million_VND', 'fdi_registered_billion_USD',
    'digital_index_0_100', 'ai_readiness_0_100',
    'trained_labor_pct', 'rd_intensity_pct',
    'internet_penetration_pct', 'gini_coef'
]
CRITERIA_LABELS = [
    'GRDP/người\n(tr.VND)', 'FDI\n(tỷ USD)', 'Digital\nIndex',
    'AI\nReadiness', 'LĐ đào tạo\n(%)', 'R&D/GRDP\n(%)',
    'Internet\n(%)', 'Gini\n(chi phí↓)'
]
IS_BENEFIT = [True, True, True, True, True, True, True, False]

# Trọng số chuyên gia mặc định
W_EXPERT = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

REGION_NAMES = [
    'Trung du miền núi phía Bắc',
    'Đồng bằng sông Hồng',
    'Bắc Trung Bộ + DH Trung Bộ',
    'Tây Nguyên',
    'Đông Nam Bộ',
    'Đồng bằng sông Cửu Long'
]

# Dữ liệu fallback (từ đề bài, phòng khi file không load được)
DATA_FALLBACK = np.array([
    [57.0,  3.5,  38, 22, 21.5, 0.18, 72, 0.405],
    [152.3, 20.0, 78, 68, 36.8, 0.85, 92, 0.358],
    [87.5,  8.2,  55, 40, 27.5, 0.32, 84, 0.372],
    [68.9,  0.8,  32, 18, 18.2, 0.15, 68, 0.412],
    [158.9, 18.5, 82, 75, 42.5, 0.78, 94, 0.385],
    [80.5,  2.1,  48, 30, 16.8, 0.22, 78, 0.392],
])


def load_data():
    """Load từ xlsx hoặc dùng fallback."""
    for search_path in [
        Path('data/vietnam_regions_2024.xlsx'),
        Path('/mnt/project/vietnam_regions_2024.xlsx'),
    ]:
        if search_path.exists():
            try:
                df = pd.read_excel(search_path)
                X = df[CRITERIA].values.astype(float)
                return X, df['region_name_vi'].tolist()
            except Exception:
                pass
    return DATA_FALLBACK.copy(), REGION_NAMES


def topsis(X, w, is_benefit):
    """
    Tính TOPSIS chuẩn.
    X: (n_alt × n_crit), w: (n_crit,), is_benefit: list bool
    Trả về C_star (n_alt,)
    """
    # Bước 1: Chuẩn hóa vector
    norms = np.sqrt((X ** 2).sum(axis=0))
    norms = np.where(norms == 0, 1e-12, norms)
    R = X / norms
    # Bước 2: Ma trận có trọng số
    V = R * w
    # Bước 3: Lý tưởng dương/âm
    A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg  = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    # Bước 4: Khoảng cách Euclide
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg  = np.sqrt(((V - A_neg)  ** 2).sum(axis=1))
    # Bước 5: Hệ số gần gũi
    denom = S_star + S_neg
    denom = np.where(denom == 0, 1e-12, denom)
    C_star = S_neg / denom
    return C_star


def entropy_weights(X):
    """Tính trọng số khách quan bằng phương pháp Entropy."""
    X_pos = np.abs(X) + 1e-12
    P = X_pos / X_pos.sum(axis=0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    d = np.where(d < 0, 0, d)
    total = d.sum()
    if total == 0:
        return np.ones(len(d)) / len(d)
    return d / total


def plot_ranking_bar(scores_exp, scores_ent, region_names):
    """So sánh điểm TOPSIS hai bộ trọng số."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#1976D2','#388E3C','#F57C00','#7B1FA2','#D32F2F','#0097A7']

    for ax, scores, title, w_label in zip(
        axes,
        [scores_exp, scores_ent],
        ['Trọng số chuyên gia', 'Trọng số Entropy'],
        ['(w_AI=0.20)', '(khách quan)']
    ):
        order = np.argsort(scores)[::-1]
        names_ord = [region_names[i] for i in order]
        scores_ord = scores[order]
        # Rút gọn tên
        short_names = [n.replace('Bắc Trung Bộ + DH Trung Bộ', 'BTB+DHMT')
                        .replace('Trung du miền núi phía Bắc', 'Trung du M.Núi')
                        .replace('Đồng bằng sông Hồng', 'ĐB sông Hồng')
                        .replace('Đồng bằng sông Cửu Long', 'ĐB sông CL') for n in names_ord]
        bars = ax.barh(short_names[::-1], scores_ord[::-1],
                       color=[colors[i] for i in order[::-1]], alpha=0.85)
        ax.set_xlabel('Điểm TOPSIS (C*)')
        ax.set_title(f'{title}\n{w_label}', fontweight='bold', fontsize=11)
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, scores_ord[::-1]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        # Đánh dấu top 3
        for j, (bar, sc) in enumerate(zip(bars[::-1], scores_ord)):
            if j < 3:
                ax.text(0.01, bar.get_y() + bar.get_height()/2,
                        f'#{j+1}', va='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_sensitivity(X, region_names):
    """Heatmap hạng khi thay đổi w_AI từ 0.05 đến 0.40."""
    w_ai_range = np.arange(0.05, 0.45, 0.05)
    n_r = len(region_names)
    rank_matrix = np.zeros((n_r, len(w_ai_range)))

    for j, w_ai in enumerate(w_ai_range):
        w = W_EXPERT.copy()
        delta = w_ai - w[3]  # index 3 = AI Readiness
        # Phân bổ lại phần dư vào các tiêu chí khác theo tỷ lệ
        w[3] = w_ai
        other_idx = [i for i in range(len(w)) if i != 3]
        w_other = w[other_idx]
        w_other = w_other / w_other.sum() * (1 - w_ai)
        w[other_idx] = w_other

        scores = topsis(X, w, IS_BENEFIT)
        ranks = n_r + 1 - scores.argsort().argsort() - 1
        rank_matrix[:, j] = ranks

    fig, ax = plt.subplots(figsize=(10, 5))
    short_names = [n.replace('Bắc Trung Bộ + DH Trung Bộ', 'BTB+DHMT')
                    .replace('Trung du miền núi phía Bắc', 'Trung du M.Núi')
                    .replace('Đồng bằng sông Hồng', 'ĐB sông Hồng')
                    .replace('Đồng bằng sông Cửu Long', 'ĐB sông CL') for n in region_names]
    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=6)
    ax.set_xticks(range(len(w_ai_range)))
    ax.set_xticklabels([f'{w:.2f}' for w in w_ai_range], fontsize=9)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel('Trọng số w_AI (AI Readiness)')
    ax.set_title('Phân tích độ nhạy: Hạng vùng theo w_AI\n(màu xanh = hạng cao)', fontweight='bold')

    for r in range(n_r):
        for c in range(len(w_ai_range)):
            ax.text(c, r, int(rank_matrix[r, c]), ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if rank_matrix[r, c] >= 4 else 'black')

    plt.colorbar(im, ax=ax, label='Hạng (1=tốt nhất)')
    plt.tight_layout()
    return fig


def plot_radar(X, w, region_names):
    """Biểu đồ radar so sánh 6 vùng theo 7 tiêu chí chính."""
    # Chuẩn hóa min-max cho hiển thị
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
    # Đảo Gini (tiêu chí chi phí)
    X_norm[:, 7] = 1 - X_norm[:, 7]

    labels_short = ['GRDP/người', 'FDI', 'Digital', 'AI Ready',
                     'LĐ đào tạo', 'R&D', 'Internet', 'Bình đẳng']
    N = len(labels_short)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ['#1976D2','#388E3C','#F57C00','#7B1FA2','#D32F2F','#0097A7']
    short_names = [n.replace('Bắc Trung Bộ + DH Trung Bộ', 'BTB+DHMT')
                    .replace('Trung du miền núi phía Bắc', 'Trung du\nM.Núi')
                    .replace('Đồng bằng sông Hồng', 'ĐB sông Hồng')
                    .replace('Đồng bằng sông Cửu Long', 'ĐB sông CL') for n in region_names]

    for i, (vals, color, name) in enumerate(zip(X_norm, colors, short_names)):
        v = vals.tolist() + [vals[0]]
        ax.plot(angles, v, 'o-', linewidth=2, color=color, label=name)
        ax.fill(angles, v, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_short, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('So sánh 6 vùng theo 8 tiêu chí\n(chuẩn hóa [0,1])', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# GIAO DIỆN STREAMLIT
# ─────────────────────────────────────────────
def render():
    st.header("🗺️ Bài 6: TOPSIS Xếp hạng 6 Vùng Kinh tế theo Ưu tiên Đầu tư AI")

    with st.expander("📖 Lý thuyết TOPSIS", expanded=False):
        st.markdown("""
        **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution):
        
        1. **Chuẩn hóa vector:** $r_{ij} = x_{ij} / \\sqrt{\\sum_i x_{ij}^2}$
        2. **Ma trận có trọng số:** $v_{ij} = w_j \\cdot r_{ij}$
        3. **Lý tưởng dương** $A^*$ = max/min tốt nhất; **Lý tưởng âm** $A^-$ = tệ nhất
        4. **Khoảng cách Euclide:** $S_i^* = \\sqrt{\\sum_j (v_{ij}-v_j^*)^2}$
        5. **Hệ số gần gũi:** $C_i^* = S_i^- / (S_i^* + S_i^-)$ — càng gần 1 càng tốt
        
        **Gini** là tiêu chí **chi phí** (thấp = tốt hơn về bình đẳng).
        """)

    X, region_names = load_data()

    # ── Trọng số chuyên gia (có thể điều chỉnh) ──
    st.subheader("⚙️ Điều chỉnh trọng số chuyên gia")
    st.caption("Tổng trọng số phải = 1.0 (tự động chuẩn hóa)")

    cols = st.columns(4)
    w_inputs = []
    default_w = [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10]
    crit_short = ['GRDP/người', 'FDI', 'Digital Index', 'AI Readiness',
                  'LĐ đào tạo', 'R&D/GRDP', 'Internet', 'Gini (↓tốt)']
    for i, (label, dw) in enumerate(zip(crit_short, default_w)):
        with cols[i % 4]:
            w_inputs.append(st.number_input(label, 0.01, 0.60, dw, 0.01, key=f'w6_{i}'))

    w_user = np.array(w_inputs)
    w_user = w_user / w_user.sum()  # Chuẩn hóa tổng = 1
    st.caption(f"Trọng số sau chuẩn hóa: {np.round(w_user, 3).tolist()}")

    if st.button("🚀 Chạy TOPSIS & Phân tích", type="primary"):
        # Tính TOPSIS hai bộ trọng số
        scores_expert = topsis(X, w_user, IS_BENEFIT)
        w_entropy = entropy_weights(X)
        scores_entropy = topsis(X, w_entropy, IS_BENEFIT)

        # ── Bảng kết quả ──
        st.subheader("📋 Kết quả TOPSIS")
        ranks_exp = len(region_names) - scores_expert.argsort().argsort()
        ranks_ent = len(region_names) - scores_entropy.argsort().argsort()

        short = [n.replace('Bắc Trung Bộ + DH Trung Bộ', 'BTB+DHMT')
                  .replace('Trung du miền núi phía Bắc', 'Trung du M.Núi')
                  .replace('Đồng bằng sông Hồng', 'ĐB sông Hồng')
                  .replace('Đồng bằng sông Cửu Long', 'ĐB sông CL') for n in region_names]

        df_res = pd.DataFrame({
            'Vùng': short,
            'C* (chuyên gia)': np.round(scores_expert, 4),
            'Hạng (CG)': ranks_exp,
            'C* (Entropy)': np.round(scores_entropy, 4),
            'Hạng (Entropy)': ranks_ent,
            'Thay đổi hạng': ranks_exp - ranks_ent
        }).sort_values('Hạng (CG)')

        st.dataframe(df_res.style.background_gradient(subset=['C* (chuyên gia)', 'C* (Entropy)'],
                                                       cmap='RdYlGn').format({
            'C* (chuyên gia)': '{:.4f}', 'C* (Entropy)': '{:.4f}',
            'Thay đổi hạng': lambda x: f'{"+" if x>0 else ""}{int(x)}'
        }), use_container_width=True)

        # Trọng số Entropy
        st.subheader("📊 Trọng số Entropy (khách quan)")
        df_entropy = pd.DataFrame({'Tiêu chí': crit_short, 'Trọng số Entropy': np.round(w_entropy, 4)})
        fig_ew, ax = plt.subplots(figsize=(8, 3))
        ax.bar(crit_short, w_entropy, color='#1976D2', alpha=0.8)
        ax.axhline(1/8, color='red', linestyle='--', label='Trọng số đều (1/8)')
        ax.set_ylabel('Trọng số')
        ax.set_title('Trọng số Entropy cho 8 tiêu chí', fontweight='bold')
        ax.legend()
        plt.xticks(rotation=15, ha='right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_ew)
        plt.close(fig_ew)

        # ── Biểu đồ xếp hạng ──
        st.subheader("🏆 So sánh xếp hạng: Chuyên gia vs Entropy")
        fig1 = plot_ranking_bar(scores_expert, scores_entropy, region_names)
        st.pyplot(fig1)
        plt.close(fig1)

        # ── Radar chart ──
        st.subheader("🕸️ Radar chart 6 vùng")
        fig2 = plot_radar(X, w_user, region_names)
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Phân tích độ nhạy ──
        st.subheader("🔍 Phân tích độ nhạy: Hạng theo w_AI Readiness")
        fig3 = plot_sensitivity(X, region_names)
        st.pyplot(fig3)
        plt.close(fig3)

        # ── Kết quả top 3 ──
        top3_exp = np.argsort(scores_expert)[::-1][:3]
        top3_ent = np.argsort(scores_entropy)[::-1][:3]
        top3_names_exp = [short[i] for i in top3_exp]
        top3_names_ent = [short[i] for i in top3_ent]

        # ── Thảo luận ──
        st.subheader("💬 Thảo luận chính sách")
        most_changed = df_res['Thay đổi hạng'].abs().idxmax()
        st.info(f"""
        **a) Top 3 theo trọng số chuyên gia:** {', '.join(f'**{n}**' for n in top3_names_exp)}
        → Vùng dẫn đầu ({top3_names_exp[0]}) nên là ưu tiên triển khai trung tâm AI quốc gia đầu tiên,
        có chỉ số AI Readiness, FDI và Digital Index cao nhất.

        **b) Thay đổi lớn nhất khi dùng Entropy:** **{df_res.loc[most_changed, 'Vùng']}**
        (chênh lệch {abs(int(df_res.loc[most_changed, 'Thay đổi hạng']))} bậc).
        Entropy đánh giá cao sự biến thiên thực tế hơn nhận định chủ quan chuyên gia.

        **c) AI Readiness & Internet penetration tương quan cao** → double-counting.
        Giải pháp: dùng PCA để gộp thành 1 chỉ số tổng hợp trước khi chạy TOPSIS,
        hoặc giảm trọng số Internet xuống < 0.05.

        **d) Ba trung tâm AI:** Theo QĐ 127/QĐ-TTg, nên đặt tại
        **{', '.join(f'{n}' for n in top3_names_exp)}** — kết hợp cả năng lực kỹ thuật
        (top 2) và yếu tố địa-chính trị/kết nối vùng (top 3).
        """)
