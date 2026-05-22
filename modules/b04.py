"""
Bài 4: Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng
LP với 24 biến quyết định, ràng buộc công bằng vùng miền
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import streamlit as st
import io

# ─────────────────────────────────────────────
# DỮ LIỆU
# ─────────────────────────────────────────────
REGIONS = ['Trung du\nmiền núi\nphía Bắc', 'Đồng bằng\nsông Hồng',
           'Bắc Trung Bộ\n+DH Trung Bộ', 'Tây Nguyên',
           'Đông Nam Bộ', 'Đồng bằng\nsông Cửu Long']
REGIONS_SHORT = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
ITEMS = ['I (Hạ tầng số)', 'D (CĐS DN)', 'AI', 'H (Nhân lực số)']
ITEMS_SHORT = ['I', 'D', 'AI', 'H']

# Hệ số tác động biên β[vùng][hạng mục]
BETA = np.array([
    [1.15, 0.85, 0.55, 1.30],  # NMM
    [0.95, 1.25, 1.40, 1.05],  # RRD
    [1.05, 0.95, 0.85, 1.15],  # NCC
    [1.20, 0.75, 0.45, 1.35],  # CH
    [0.90, 1.30, 1.55, 1.00],  # SE
    [1.10, 0.85, 0.65, 1.25],  # MD
])

# Chỉ số số hóa ban đầu D0
D0 = np.array([38, 78, 55, 32, 82, 48])


def solve_lp(budget=50000, floor=5000, ceil_r=14000,
             h_floor=10000, gamma=0.002, lam=0.7,
             use_equity=True):
    """
    Giải LP bằng scipy.optimize.linprog (HiGHS).
    Trả về (x_opt [6×4], Z_star, status_msg)
    """
    from scipy.optimize import linprog

    n_r, n_j = 6, 4
    n = n_r * n_j  # 24 biến x[r,j] (flatten: r*4+j)

    # Hàm mục tiêu: max Σβ·x  →  min -Σβ·x
    c = -BETA.flatten()

    A_ub, b_ub = [], []

    # C1: tổng ngân sách ≤ budget
    A_ub.append(np.ones(n))
    b_ub.append(budget)

    # C2: sàn mỗi vùng ≥ floor  →  -Σⱼx[r,j] ≤ -floor
    for r in range(n_r):
        row = np.zeros(n)
        row[r*n_j:(r+1)*n_j] = -1
        A_ub.append(row)
        b_ub.append(-floor)

    # C3: trần mỗi vùng ≤ ceil_r
    for r in range(n_r):
        row = np.zeros(n)
        row[r*n_j:(r+1)*n_j] = 1
        A_ub.append(row)
        b_ub.append(ceil_r)

    # C4: sàn nhân lực Σᵣx[r,H] ≥ h_floor  →  -Σ ≤ -h_floor
    row = np.zeros(n)
    for r in range(n_r):
        row[r*n_j + 3] = -1
    A_ub.append(row)
    b_ub.append(-h_floor)

    # C5: công bằng vùng (linear hóa bằng biến phụ M)
    # Thêm biến M (index 24): Dᵣ + γ·x[r,D] ≤ M  và  Dᵣ + γ·x[r,D] ≥ λ·M
    # Mở rộng vector lên 25 chiều
    if use_equity:
        n_ext = n + 1  # biến M ở index 24
        c_ext = np.append(c, 0.0)  # M không ảnh hưởng mục tiêu

        A_ext, b_ext = [], []
        # Sao chép ràng buộc cũ (thêm cột 0 cho M)
        for row_old, bv in zip(A_ub, b_ub):
            A_ext.append(np.append(row_old, 0.0))
            b_ext.append(bv)

        # Dᵣ + γ·x[r,D] ≤ M  →  γ·x[r,D] - M ≤ -D0[r]
        for r in range(n_r):
            row = np.zeros(n_ext)
            row[r*n_j + 1] = gamma
            row[n] = -1
            A_ext.append(row)
            b_ext.append(-D0[r])

        # Dᵣ + γ·x[r,D] ≥ λ·M  →  -γ·x[r,D] + λ·M ≤ D0[r]
        for r in range(n_r):
            row = np.zeros(n_ext)
            row[r*n_j + 1] = -gamma
            row[n] = lam
            A_ext.append(row)
            b_ext.append(D0[r])

        bounds = [(0, None)] * n + [(0, None)]
        res = linprog(c_ext, A_ub=A_ext, b_ub=b_ext, bounds=bounds, method='highs')
        if res.success:
            x_opt = res.x[:n].reshape(n_r, n_j)
            return x_opt, -res.fun, f"Tối ưu: Z* = {-res.fun:,.0f} tỷ VND GDP gain"
        else:
            return None, None, f"Không tìm được nghiệm: {res.message}"
    else:
        n_ext = n
        c_ext = c
        A_ext = [np.array(row) for row in A_ub]
        b_ext = b_ub
        bounds = [(0, None)] * n
        res = linprog(c_ext, A_ub=A_ext, b_ub=b_ext, bounds=bounds, method='highs')
        if res.success:
            x_opt = res.x.reshape(n_r, n_j)
            return x_opt, -res.fun, f"Tối ưu (không công bằng): Z* = {-res.fun:,.0f} tỷ VND"
        else:
            return None, None, f"Không tìm được nghiệm: {res.message}"


def plot_heatmap(x_opt, title="Phân bổ ngân sách tối ưu (tỷ VND)"):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(x_opt, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Hạ tầng số', 'CĐS DN', 'AI', 'Nhân lực số'], fontsize=10)
    ax.set_yticks(range(6))
    region_labels = ['Trung du\nmiền núi\nphía Bắc', 'Đồng bằng\nsông Hồng',
                     'Bắc Trung Bộ\n+DH Trung Bộ', 'Tây Nguyên',
                     'Đông Nam Bộ', 'Đồng bằng\nsông Cửu Long']
    ax.set_yticklabels(region_labels, fontsize=9)
    for r in range(6):
        for j in range(4):
            ax.text(j, r, f'{x_opt[r,j]:,.0f}', ha='center', va='center',
                    fontsize=9, color='black' if x_opt[r,j] < x_opt.max()*0.7 else 'white')
    plt.colorbar(im, ax=ax, label='Tỷ VND')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_bar_by_region(x_opt, x_no_eq):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in zip(axes, [x_opt, x_no_eq],
                                ['Có ràng buộc công bằng', 'Không có ràng buộc công bằng']):
        totals = data.sum(axis=1)
        colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4']
        ax.barh(range(6), totals, color=colors)
        ax.set_yticks(range(6))
        ax.set_yticklabels(['Trung du', 'ĐB sông Hồng', 'BTB+DHMT',
                            'Tây Nguyên', 'Đông Nam Bộ', 'ĐB sông CL'], fontsize=9)
        ax.set_xlabel('Tỷ VND')
        ax.set_title(title, fontsize=11, fontweight='bold')
        for i, v in enumerate(totals):
            ax.text(v + 100, i, f'{v:,.0f}', va='center', fontsize=9)
    plt.tight_layout()
    return fig


def sensitivity_budget(budgets):
    zs = []
    for b in budgets:
        _, z, _ = solve_lp(budget=b)
        zs.append(z if z else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(budgets, zs, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Ngân sách tổng (nghìn tỷ VND)', fontsize=11)
    ax.set_ylabel('Z* (tỷ VND GDP gain)', fontsize=11)
    ax.set_title('Phân tích độ nhạy: Z* theo ngân sách', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for x, y in zip(budgets, zs):
        ax.annotate(f'{y:,.0f}', (x, y), textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# GIAO DIỆN STREAMLIT
# ─────────────────────────────────────────────
def render():
    st.header("📊 Bài 4: LP Phân bổ Ngân sách Số theo Ngành - Vùng")

    with st.expander("📖 Bối cảnh & Mô hình", expanded=False):
        st.markdown("""
        **Bối cảnh:** Phân bổ **50.000 tỷ VND** ngân sách kinh tế số cho **6 vùng kinh tế xã hội**
        và **4 hạng mục** (Hạ tầng số, CĐS DN, AI, Nhân lực số) nhằm tối đa hóa GDP gain
        đồng thời bảo đảm công bằng vùng miền.

        **Hàm mục tiêu:**
        $$\\max Z = \\sum_r \\sum_j \\beta_{j,r} \\cdot x_{j,r}$$

        **Ràng buộc chính:**
        - C1: Tổng ngân sách ≤ Budget
        - C2: Sàn mỗi vùng ≥ 5.000 tỷ
        - C3: Trần mỗi vùng ≤ 12.000 tỷ
        - C4: Sàn nhân lực ≥ 12.000 tỷ (24% tổng)
        - C5: Công bằng vùng — vùng thấp nhất ≥ 70% vùng cao nhất (về chỉ số số hóa)
        """)

    # ── Tham số ──
    st.subheader("⚙️ Tham số bài toán")
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("Ngân sách tổng (tỷ VND)", 40000, 100000, 50000, 5000)
        h_floor = st.number_input("Sàn nhân lực số (tỷ VND)", 5000, 20000, 10000, 1000)
    with col2:
        floor = st.number_input("Sàn mỗi vùng (tỷ VND)", 2000, 10000, 5000, 500)
        gamma = st.number_input("γ (hiệu quả CĐS DN)", 0.001, 0.010, 0.002, 0.001, format="%.3f")
    with col3:
        ceil_r = st.number_input("Trần mỗi vùng (tỷ VND)", 8000, 20000, 14000, 1000)
        lam = st.slider("λ (ngưỡng công bằng)", 0.5, 0.95, 0.70, 0.05)

    use_equity = st.checkbox("✅ Áp dụng ràng buộc công bằng vùng (C5)", value=True)

    if st.button("🚀 Giải bài toán LP", type="primary"):
        with st.spinner("Đang giải..."):
            x_opt, z_star, msg = solve_lp(budget, floor, ceil_r, h_floor, gamma, lam, use_equity)
            x_no_eq, z_no_eq, _ = solve_lp(budget, floor, ceil_r, h_floor, gamma, lam, False)

        if x_opt is None:
            st.error(msg)
            return

        # Kết quả chính
        st.success(msg)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Z* GDP gain", f"{z_star:,.0f} tỷ VND")
        col_b.metric("Chi phí công bằng", f"{z_no_eq - z_star:,.0f} tỷ VND")
        col_c.metric("% giảm do công bằng", f"{(z_no_eq-z_star)/z_no_eq*100:.2f}%")

        # Bảng phân bổ tối ưu
        st.subheader("📋 Ma trận phân bổ tối ưu x[vùng, hạng mục] (tỷ VND)")
        df_result = pd.DataFrame(x_opt,
                                 index=['Trung du/M.Núi', 'ĐB sông Hồng', 'BTB+DHMT',
                                        'Tây Nguyên', 'Đông Nam Bộ', 'ĐB sông CL'],
                                 columns=['Hạ tầng số', 'CĐS DN', 'AI', 'Nhân lực số'])
        df_result['TỔNG VÙNG'] = df_result.sum(axis=1)
        df_result.loc['TỔNG HM'] = df_result.sum()
        st.dataframe(df_result.style.format("{:,.0f}").background_gradient(cmap='YlOrRd', axis=None))

        # Heatmap
        st.subheader("🗺️ Heatmap phân bổ")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plot_heatmap(x_opt, "Có ràng buộc công bằng")
            st.pyplot(fig1)
            plt.close(fig1)
        with col2:
            fig2 = plot_heatmap(x_no_eq, "Không có ràng buộc công bằng")
            st.pyplot(fig2)
            plt.close(fig2)

        # Bar chart so sánh vùng
        st.subheader("📊 So sánh ngân sách mỗi vùng: có vs không có ràng buộc công bằng")
        fig3 = plot_bar_by_region(x_opt, x_no_eq)
        st.pyplot(fig3)
        plt.close(fig3)

        # Phân tích độ nhạy
        st.subheader("📈 Phân tích độ nhạy: Z* theo ngân sách tổng")
        budgets = list(range(40000, 110000, 10000))
        fig4 = sensitivity_budget(budgets)
        st.pyplot(fig4)
        plt.close(fig4)

        # Thảo luận chính sách
        st.subheader("💬 Thảo luận chính sách")
        diffs = x_no_eq.sum(axis=1) - x_opt.sum(axis=1)
        top_gainer = ['Trung du/M.Núi', 'ĐB sông Hồng', 'BTB+DHMT',
                      'Tây Nguyên', 'Đông Nam Bộ', 'ĐB sông CL'][np.argmax(diffs)]
        st.info(f"""
        **a) Không có ràng buộc công bằng:** Vốn sẽ chảy nhiều về **{top_gainer}**
        (do hệ số β AI cao nhất = 1,55). Dài hạn sẽ tạo khoảng cách phát triển lớn hơn giữa các vùng.

        **b) Chi phí của ràng buộc công bằng:** Giảm Z* khoảng **{(z_no_eq-z_star)/z_no_eq*100:.2f}%**
        ({z_no_eq-z_star:,.0f} tỷ VND). Đây là "chi phí xã hội" để đổi lấy tính công bằng vùng miền.

        **c) Tây Nguyên (hệ số AI = 0,45):** Mô hình ưu tiên đầu tư vào
        **H (nhân lực, β=1,35)** và **I (hạ tầng, β=1,20)** thay vì AI trực tiếp —
        cần xây nền tảng trước khi triển khai AI.
        """)
