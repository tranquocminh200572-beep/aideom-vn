"""
Bài 5: Quy hoạch nguyên hỗn hợp (MIP) lựa chọn dự án chuyển đổi số
15 dự án, biến nhị phân, ràng buộc loại trừ & tiên quyết
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import streamlit as st

# ─────────────────────────────────────────────
# DỮ LIỆU 15 DỰ ÁN
# ─────────────────────────────────────────────
PROJECTS = {
    1:  {"name": "TTDL QG Hòa Lạc",          "field": "Hạ tầng",       "C": 12000, "B": 21500, "C1": 8500,  "p": 0.85},
    2:  {"name": "TTDL QG Phía Nam",           "field": "Hạ tầng",       "C": 11500, "B": 20800, "C1": 7500,  "p": 0.85},
    3:  {"name": "5G phủ sóng toàn quốc",      "field": "Hạ tầng",       "C": 18000, "B": 32500, "C1": 12000, "p": 0.85},
    4:  {"name": "VNeID 2.0",                  "field": "Chính phủ số",   "C": 4500,  "B": 9200,  "C1": 3500,  "p": 0.75},
    5:  {"name": "Cổng DVC QG v3",             "field": "Chính phủ số",   "C": 3200,  "B": 6800,  "C1": 2500,  "p": 0.75},
    6:  {"name": "Y tế số QG",                 "field": "Y tế số",        "C": 5800,  "B": 11400, "C1": 4000,  "p": 0.80},
    7:  {"name": "Giáo dục số K-12",           "field": "Giáo dục",       "C": 6500,  "B": 12200, "C1": 4500,  "p": 0.80},
    8:  {"name": "Trung tâm AI QG",            "field": "AI",             "C": 15000, "B": 28500, "C1": 9000,  "p": 0.65},
    9:  {"name": "Sandbox fintech",             "field": "Tài chính số",   "C": 2500,  "B": 5800,  "C1": 1800,  "p": 0.80},
    10: {"name": "Logistics thông minh",        "field": "Logistics",      "C": 7200,  "B": 13800, "C1": 5000,  "p": 0.80},
    11: {"name": "Nông nghiệp số ĐBSCL",        "field": "Nông nghiệp",    "C": 4800,  "B": 8500,  "C1": 3500,  "p": 0.80},
    12: {"name": "Đào tạo 50k kỹ sư AI",       "field": "Nhân lực",       "C": 8500,  "B": 16200, "C1": 5500,  "p": 0.80},
    13: {"name": "Khu CN bán dẫn",             "field": "Bán dẫn",        "C": 20000, "B": 35000, "C1": 13000, "p": 0.65},
    14: {"name": "An ninh mạng QG (SOC)",       "field": "An ninh",        "C": 3800,  "B": 7500,  "C1": 2800,  "p": 0.80},
    15: {"name": "Open Data QG",               "field": "Dữ liệu",        "C": 1500,  "B": 3800,  "C1": 1200,  "p": 0.80},
}
FIELD_COLORS = {
    'Hạ tầng': '#2196F3', 'Chính phủ số': '#4CAF50', 'Y tế số': '#E91E63',
    'Giáo dục': '#FF9800', 'AI': '#9C27B0', 'Tài chính số': '#00BCD4',
    'Logistics': '#795548', 'Nông nghiệp': '#8BC34A', 'Nhân lực': '#FF5722',
    'Bán dẫn': '#607D8B', 'An ninh': '#F44336', 'Dữ liệu': '#FFC107',
}


def solve_mip(budget_total=80000, budget_y12=40000,
              use_risk=False, force_p1p2=False):
    """
    Giải MIP bằng PuLP (CBC).
    force_p1p2: buộc chọn cả P1 và P2 (câu 5.4.3)
    """
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD, value
    except ImportError:
        return None, None, "Cần cài: pip install pulp"

    P = list(range(1, 16))
    m = LpProblem('VN_Project_Selection', LpMaximize)
    y = LpVariable.dicts('y', P, cat='Binary')

    # Hàm mục tiêu
    if use_risk:
        m += lpSum(PROJECTS[i]['p'] * PROJECTS[i]['B'] * y[i] for i in P)
    else:
        m += lpSum(PROJECTS[i]['B'] * y[i] for i in P)

    # C1: ngân sách tổng
    m += lpSum(PROJECTS[i]['C'] * y[i] for i in P) <= budget_total
    # C2: ngân sách năm 1-2
    m += lpSum(PROJECTS[i]['C1'] * y[i] for i in P) <= budget_y12
    # C3: loại trừ (chỉ 1 trong P1, P2) — trừ khi force_p1p2
    if force_p1p2:
        m += y[1] == 1
        m += y[2] == 1
    else:
        m += y[1] + y[2] <= 1
    # C4: P8 cần P12 (tiên quyết)
    m += y[8] <= y[12]
    # C5: P13 cần P12
    m += y[13] <= y[12]
    # C6: ít nhất 1 chính phủ số; bắt buộc an ninh
    m += y[4] + y[5] >= 1
    m += y[14] == 1
    # C7: số lượng dự án
    m += lpSum(y[i] for i in P) >= 7
    m += lpSum(y[i] for i in P) <= 11

    m.solve(PULP_CBC_CMD(msg=False))

    selected = [i for i in P if y[i].value() and y[i].value() > 0.5]
    z = value(m.objective) if value(m.objective) else None
    total_cost = sum(PROJECTS[i]['C'] for i in selected)

    status = "Tối ưu" if m.status == 1 else "Không tìm được nghiệm"
    return selected, z, f"{status} | Z* = {z:,.0f} tỷ | Chi phí = {total_cost:,.0f} tỷ" if z else status


def plot_project_comparison(selected):
    """Biểu đồ lợi ích vs chi phí của dự án được chọn."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sel_data = [(i, PROJECTS[i]) for i in selected]
    sel_data.sort(key=lambda x: x[1]['B'], reverse=True)

    names = [d['name'] for _, d in sel_data]
    benefits = [d['B'] for _, d in sel_data]
    costs = [d['C'] for _, d in sel_data]
    colors = [FIELD_COLORS.get(d['field'], '#999') for _, d in sel_data]

    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, benefits, w, label='Lợi ích NPV', color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, costs, w, label='Chi phí', color=colors, alpha=0.45, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Tỷ VND')
    ax.set_title('Dự án được chọn: Lợi ích NPV vs Chi phí', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_roi_chart(selected):
    """Tỷ suất B/C của từng dự án được chọn."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sel_sorted = sorted(selected, key=lambda i: PROJECTS[i]['B']/PROJECTS[i]['C'], reverse=True)
    names = [PROJECTS[i]['name'] for i in sel_sorted]
    roi = [PROJECTS[i]['B']/PROJECTS[i]['C'] for i in sel_sorted]
    colors = [FIELD_COLORS.get(PROJECTS[i]['field'], '#999') for i in sel_sorted]

    bars = ax.barh(names, roi, color=colors)
    ax.axvline(x=1.0, color='red', linestyle='--', label='Hòa vốn (B/C=1)')
    ax.set_xlabel('Tỷ suất lợi ích / chi phí (B/C)')
    ax.set_title('Tỷ suất B/C của dự án được chọn', fontweight='bold')
    ax.legend()
    for bar, val in zip(bars, roi):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}x', va='center', fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# GIAO DIỆN STREAMLIT
# ─────────────────────────────────────────────
def render():
    st.header("🏗️ Bài 5: MIP Lựa chọn Dự án Chuyển đổi Số")

    with st.expander("📖 Bối cảnh & Mô hình", expanded=False):
        st.markdown("""
        **Bài toán:** Chọn tập dự án tối ưu từ **15 dự án** ứng cử (tổng ngân sách 80.000 tỷ VND).
        
        **Biến quyết định:** $y_i \\in \\{0,1\\}$ — chọn/không chọn dự án i
        
        **Hàm mục tiêu:** $\\max \\sum_i B_i \\cdot y_i$
        
        **Ràng buộc đặc biệt:**
        - C3: Chỉ chọn 1 trung tâm dữ liệu (P1 hoặc P2)
        - C4/C5: AI quốc gia (P8) và bán dẫn (P13) phải có đào tạo kỹ sư (P12) trước
        - C6: Bắt buộc ít nhất 1 dự án chính phủ số; bắt buộc an ninh mạng (P14)
        - C7: Chọn 7 ≤ số dự án ≤ 11
        """)

    # Hiển thị bảng dự án
    st.subheader("📋 Danh mục 15 dự án ứng cử")
    df_projects = pd.DataFrame([
        {
            'Mã': f'P{i}', 'Tên dự án': d['name'], 'Lĩnh vực': d['field'],
            'Chi phí (tỷ)': d['C'], 'Lợi ích NPV (tỷ)': d['B'],
            'Ngân sách Y1-2 (tỷ)': d['C1'], 'B/C': round(d['B']/d['C'], 2),
            'p thành công': d['p']
        }
        for i, d in PROJECTS.items()
    ])
    st.dataframe(df_projects.style.format({
        'Chi phí (tỷ)': '{:,.0f}', 'Lợi ích NPV (tỷ)': '{:,.0f}',
        'Ngân sách Y1-2 (tỷ)': '{:,.0f}', 'B/C': '{:.2f}', 'p thành công': '{:.0%}'
    }).background_gradient(subset=['B/C'], cmap='RdYlGn'), use_container_width=True)

    # Tham số
    st.subheader("⚙️ Tham số & Kịch bản")
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_total = st.number_input("Ngân sách tổng 5 năm (tỷ VND)", 60000, 150000, 80000, 10000)
    with col2:
        budget_y12 = st.number_input("Ngân sách năm 1-2 (tỷ VND)", 30000, 80000, 40000, 5000)
    with col3:
        scenario = st.selectbox("Kịch bản", [
            "Cơ bản (C3: chỉ 1 TTDL)",
            "Nới ngân sách 100.000 tỷ",
            "Buộc cả P1+P2 (redundancy)",
            "Tối đa hóa lợi ích kỳ vọng (theo rủi ro)"
        ])

    if st.button("🚀 Giải bài toán MIP", type="primary"):
        with st.spinner("CBC đang tính..."):
            use_risk = "rủi ro" in scenario
            force_p1p2 = "P1+P2" in scenario
            b_total = 100000 if "100.000" in scenario else budget_total
            selected, z, msg = solve_mip(b_total, budget_y12, use_risk, force_p1p2)

        if selected is None:
            st.error(msg)
            return

        if "Không tìm" in msg:
            st.error(f"❌ {msg}")
            st.warning("Kịch bản buộc cả P1+P2 vượt quá ràng buộc C3 nên **không khả thi**.")
            return

        st.success(f"✅ {msg}")

        # Metrics
        total_cost = sum(PROJECTS[i]['C'] for i in selected)
        total_ben = sum(PROJECTS[i]['B'] for i in selected)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Dự án được chọn", len(selected))
        col_b.metric("Tổng chi phí", f"{total_cost:,.0f} tỷ")
        col_c.metric("Tổng lợi ích NPV", f"{total_ben:,.0f} tỷ")
        col_d.metric("B/C trung bình", f"{total_ben/total_cost:.2f}x")

        # Danh sách dự án được chọn
        st.subheader("✅ Các dự án được chọn")
        df_sel = pd.DataFrame([
            {'Mã': f'P{i}', 'Tên dự án': PROJECTS[i]['name'],
             'Lĩnh vực': PROJECTS[i]['field'],
             'Chi phí (tỷ)': PROJECTS[i]['C'],
             'Lợi ích NPV (tỷ)': PROJECTS[i]['B'],
             'B/C': round(PROJECTS[i]['B']/PROJECTS[i]['C'], 2)}
            for i in sorted(selected)
        ])
        st.dataframe(df_sel.style.format({
            'Chi phí (tỷ)': '{:,.0f}', 'Lợi ích NPV (tỷ)': '{:,.0f}', 'B/C': '{:.2f}'
        }).background_gradient(subset=['B/C'], cmap='RdYlGn'), use_container_width=True)

        # Biểu đồ
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Lợi ích vs Chi phí")
            fig1 = plot_project_comparison(selected)
            st.pyplot(fig1)
            plt.close(fig1)
        with col2:
            st.subheader("📈 Tỷ suất B/C")
            fig2 = plot_roi_chart(selected)
            st.pyplot(fig2)
            plt.close(fig2)

        # Phân tích dự án không được chọn
        not_selected = [i for i in range(1, 16) if i not in selected]
        st.subheader("❌ Dự án không được chọn")
        if not_selected:
            reasons = []
            for i in not_selected:
                d = PROJECTS[i]
                if i == 1 and 2 in selected:
                    reason = "Bị loại trừ bởi P2 (C3)"
                elif i == 2 and 1 in selected:
                    reason = "Bị loại trừ bởi P1 (C3)"
                elif i in [8, 13] and 12 not in selected:
                    reason = "Cần P12 nhưng P12 không được chọn (C4/C5)"
                else:
                    reason = "Ngân sách không đủ"
                reasons.append({'Mã': f'P{i}', 'Tên': d['name'], 'Chi phí': d['C'],
                                 'B/C': round(d['B']/d['C'], 2), 'Lý do': reason})
            st.dataframe(pd.DataFrame(reasons))

        # Thảo luận
        st.subheader("💬 Thảo luận chính sách")
        p15_roi = PROJECTS[15]['B']/PROJECTS[15]['C']
        st.info(f"""
        **a) P15 (Open Data) bị loại** dù B/C = {p15_roi:.2f}x — vì ngân sách năm 1-2 đã bị chiếm
        bởi các dự án chi phí lớn hơn. Xét về mặt chính sách, đây là điểm đáng lo ngại vì
        dữ liệu mở là nền tảng cho toàn hệ sinh thái số.

        **b) P14 (An ninh mạng) bắt buộc** làm giảm tính linh hoạt nhưng hoàn toàn hợp lý —
        không thể triển khai AI và dữ liệu số mà không có nền tảng bảo mật.

        **c) Hiệu ứng cộng hưởng P8+P13:** Thực tế, P8 (AI QG) và P13 (bán dẫn) tạo ra
        lợi ích cộng hưởng lớn hơn tổng lợi ích độc lập. Có thể mô hình hóa bằng biến $y_{{8,13}}$
        với lợi ích bonus khi cả hai được chọn.
        """)
