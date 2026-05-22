"""
AIDEOM-VN Webapp — Streamlit
Bài tập cuối kỳ: Các mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI
"""

import streamlit as st
import importlib, sys, os, base64
from pathlib import Path
import pandas as pd
import numpy as np

# ── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIDEOM-VN | Mô hình Kinh tế Việt Nam AI",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f0f2f6; border-radius:8px; padding:12px 16px; margin:4px 0;
}
.bai-header {
    background: linear-gradient(135deg,#d32f2f,#b71c1c);
    color:white; padding:10px 20px; border-radius:8px; margin-bottom:12px;
}
.status-done   { color:#27ae60; font-weight:bold; }
.status-wip    { color:#e67e22; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

THIS_DIR = Path(__file__).parent
def _import(name):
    path = THIS_DIR / "modules" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def show_img(b64: str):
    st.image(base64.b64decode(b64))

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🇻🇳 AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI")
    st.divider()

    BAI_LIST = {
        "🏠 Tổng quan": "home",
        "─── CẤP ĐỘ DỄ ───": None,
        "Bài 1 · Cobb-Douglas": "b01",
        "Bài 2 · LP ngân sách": "b02",
        "Bài 3 · Chỉ số ưu tiên ngành": "b03",
        "─── TRUNG BÌNH ───": None,
        "Bài 4 · LP ngành-vùng": "b04",
        "Bài 5 · MIP lựa chọn dự án": "b05",
        "Bài 6 · TOPSIS xếp hạng vùng": "b06",
        "─── KHÁ KHÓ ───": None,
        "Bài 7 · Pareto NSGA-II": "b07",
        "Bài 8 · Tối ưu động 2026-2035": "b08",
        "Bài 9 · Lao động & AI": "b09",
        "─── KHÓ ───": None,
        "Bài 10 · Stochastic LP": "b10",
        "Bài 11 · Q-learning RL": "b11",
        "Bài 12 · Đồ án tích hợp": "b12",
    }

    selected = None
    for label, key in BAI_LIST.items():
        if key is None:
            st.markdown(f"**{label}**")
        elif key == "home":
            if st.button("🏠 Tổng quan", use_container_width=True):
                st.session_state["page"] = "home"
        else:
            if st.button(label, use_container_width=True):
                st.session_state["page"] = key

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    st.divider()
    st.caption("📊 Dữ liệu: GSO/NSO, MoST, MPI 2020-2025")

page = st.session_state["page"]

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown('<div class="bai-header"><h2>🇻🇳 AIDEOM-VN Dashboard</h2>'
                '<p>Bài tập cuối kỳ — Các mô hình ra quyết định</p></div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP 2024", "11.511 nghìn tỷ VND", "+7.09%")
    c2.metric("GDP/người 2024", "4.700 USD", "↑ vs 2023")
    c3.metric("Kinh tế số/GDP", "18.3%", "+1.8 pp")
    c4.metric("FDI giải ngân", "25.35 tỷ USD", "Kỷ lục")

    st.divider()
    st.subheader("📋 Tiến độ 12 bài tập")
    progress = {
        "Bài 1 — Cobb-Douglas": "✅ Hoàn thành",
        "Bài 2 — LP ngân sách": "✅ Hoàn thành",
        "Bài 3 — Chỉ số ưu tiên ngành": "✅ Hoàn thành",
        "Bài 4 — LP ngành-vùng": "✅ Hoàn thành",
        "Bài 5 — MIP lựa chọn dự án": "✅ Hoàn thành",
        "Bài 6 — TOPSIS": "✅ Hoàn thành",
        "Bài 7 — Pareto NSGA-II": "✅ Hoàn thành",
        "Bài 8 — Tối ưu động": "✅ Hoàn thành",
        "Bài 9 — Lao động & AI": "✅ Hoàn thành",
        "Bài 10 — Stochastic LP": "✅ Hoàn thành",
        "Bài 11 — Q-learning": "✅ Hoàn thành",
        "Bài 12 — Đồ án tích hợp": "🔄 Tổng hợp từ B01-B11",
    }
    cols = st.columns(3)
    for i, (k, v) in enumerate(progress.items()):
        cols[i%3].markdown(f"**{k}**<br>{v}", unsafe_allow_html=True)
        cols[i%3].divider()

elif page in ("b01","b02","b03","b04","b05","b06"):
    try:
        mod = _import(page)
        mod.render()
    except Exception as e:
        st.error(f"Lỗi load {page}: {e}")
        st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# BÀI 7 — NSGA-II với đầy đủ tham số điều chỉnh
# ══════════════════════════════════════════════════════════════════════════════
elif page == "b07":
    st.markdown('<div class="bai-header"><h3>Bài 7 · Tối ưu đa mục tiêu Pareto với NSGA-II</h3></div>',
                unsafe_allow_html=True)
    st.markdown("""
    **4 mục tiêu:** (1) Tối đa GDP gain | (2) Giảm bất bình đẳng vùng | 
    (3) Giảm phát thải CO₂ | (4) Giảm rủi ro an ninh dữ liệu  
    **24 biến:** 6 vùng × 4 hạng mục
    """)

    st.markdown("### 🎛️ Tham số điều chỉnh")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**⚙️ Tham số NSGA-II**")
        pop_size = st.slider("Pop size", 20, 200, 80, 10, key="b7_pop")
        n_gen    = st.slider("Số generation", 50, 300, 150, 10, key="b7_gen")
        seed_b7  = st.number_input("Seed", 0, 999, 42, 1, key="b7_seed")
    with col_b:
        st.markdown("**💰 Ràng buộc ngân sách (tỷ VND)**")
        budget_b7 = st.slider("Ngân sách tổng", 30000, 80000, 50000, 5000, key="b7_budget")
        min_r_b7  = st.slider("Sàn mỗi vùng", 2000, 8000, 5000, 500, key="b7_minr")
        max_r_b7  = st.slider("Trần mỗi vùng", 8000, 20000, 14000, 1000, key="b7_maxr")
        min_h_b7  = st.slider("Sàn nhân lực tổng", 5000, 20000, 12000, 1000, key="b7_minh")
    with col_c:
        st.markdown("**⚖️ Trọng số TOPSIS (tổng = 1)**")
        w_gdp  = st.slider("w₁ GDP gain",        0.10, 0.70, 0.40, 0.05, key="b7_w1")
        w_eq   = st.slider("w₂ Bao trùm",        0.10, 0.50, 0.25, 0.05, key="b7_w2")
        w_env  = st.slider("w₃ Môi trường",       0.05, 0.40, 0.20, 0.05, key="b7_w3")
        w_sec  = st.slider("w₄ An ninh dữ liệu",  0.05, 0.30, 0.15, 0.05, key="b7_w4")
        total_w = w_gdp + w_eq + w_env + w_sec
        # Chuẩn hóa
        w_gdp_n = w_gdp/total_w; w_eq_n = w_eq/total_w
        w_env_n = w_env/total_w; w_sec_n = w_sec/total_w
        color_w = "green" if abs(total_w-1.0)<0.02 else "orange"
        st.markdown(f"Tổng: :{color_w}[{total_w:.2f}] → chuẩn hóa về 1.0")
        st.markdown(f"→ w=({w_gdp_n:.2f}, {w_eq_n:.2f}, {w_env_n:.2f}, {w_sec_n:.2f})")

    if st.button("🚀 Chạy NSGA-II / Random Search", type="primary", key="b7_run"):
        with st.spinner(f"Đang tìm tập Pareto (pop={pop_size}, gen={n_gen})... ~30-60 giây"):
            import importlib.util as _ilu7
            spec7 = _ilu7.spec_from_file_location("b07", THIS_DIR/"modules"/"b07.py")
            m7 = _ilu7.module_from_spec(spec7); spec7.loader.exec_module(m7)
            res = m7.run_b07(
                pop_size=pop_size, n_gen=n_gen, seed=int(seed_b7),
                budget=budget_b7, min_r=min_r_b7, max_r=max_r_b7, min_h=min_h_b7,
                w_gdp=w_gdp_n, w_equity=w_eq_n, w_env=w_env_n, w_security=w_sec_n,
            )

        img_path = os.path.join(THIS_DIR, 'modules', 'b12_aideom_vn', 'b07_enhanced.png')
        if os.path.exists(img_path):
            st.markdown('---')
            st.image(img_path, use_container_width=True)

        if "error" in res:
            st.error(res["error"])
        else:
            st.success(f"✅ Tìm được **{res['pareto_size']}** nghiệm Pareto")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GDP gain (thỏa hiệp)", f"{-res['best_F'][0]:,.0f} tỷ")
            c2.metric("Bất bình đẳng", f"{res['best_F'][1]:.1f}")
            c3.metric("Phát thải CO₂", f"{res['best_F'][2]:.1f}")
            c4.metric("Rủi ro AN dữ liệu", f"{res['best_F'][3]:.2f}")

            tab1, tab2, tab3 = st.tabs(["📊 Đường biên Pareto 3D", "📈 Parallel Coords", "📋 Phân bổ tối ưu"])
            with tab1:
                show_img(res["img_3d"])
            with tab2:
                show_img(res["img_parallel"])
            with tab3:
                st.dataframe(res["best_alloc"].style.format("{:.0f}"), use_container_width=True)
                st.subheader("📊 Phân tích đánh đổi (Trade-off)")
                st.table(pd.DataFrame(res["trade_off"].items(), columns=["Chỉ tiêu","Giá trị"]))

    with st.expander("💡 Câu hỏi thảo luận chính sách"):
        st.markdown("""
        1. **Đánh đổi tăng trưởng vs bao trùm**: Khi ưu tiên GDP, vốn tập trung về ĐNB và ĐBSH — bất bình đẳng tăng.
        2. **Trọng số chính sách** (0.40;0.25;0.20;0.15) phản ánh Đại hội XIII ưu tiên tăng trưởng trước; COP26 đòi hỏi tăng trọng w₃.
        3. **NSGA-II ≠ quyết định chính trị**: tập Pareto cung cấp *danh mục lựa chọn*, không thay thế đối thoại xã hội.
        4. **Pop size & generation**: Pop lớn hơn → đa dạng hơn nhưng chậm hơn; gen nhiều hơn → hội tụ tốt hơn.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# BÀI 8
# ══════════════════════════════════════════════════════════════════════════════
elif page == "b08":
    st.markdown('<div class="bai-header"><h3>Bài 8 · Tối ưu động phân bổ liên thời gian 2026–2035</h3></div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Mô hình:** Cobb-Douglas mở rộng + tích lũy vốn 4 loại + TFP nội sinh  
    **Mục tiêu:** Tối đa phúc lợi ∑ρᵗ·ln(Cₜ) | **ρ=0.97** | **Giải:** SLSQP
    """)

    st.markdown("### 🎛️ Tham số điều chỉnh")
    col8a, col8b, col8c = st.columns(3)
    with col8a:
        st.markdown("**⚙️ Tham số chung**")
        show_shock = st.toggle("Mô phỏng cú sốc 2028 (-8% GDP)", value=False)
        rho_val = st.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01, key="b8_rho")
        maxiter_b8 = st.slider("Số vòng lặp tối đa", 100, 500, 300, 50, key="b8_iter")
    with col8b:
        st.markdown("**📊 Trần tỷ lệ đầu tư (%GDP)**")
        max_sK  = st.slider("Trần đầu tư K (%)", 30, 80, 80, 5, key="b8_maxK")
        max_sD  = st.slider("Trần đầu tư D (%)", 10, 50, 30, 5, key="b8_maxD")
        max_sAI = st.slider("Trần đầu tư AI (%)", 5, 40, 25, 5, key="b8_maxAI")
        max_sH  = st.slider("Trần đầu tư H (%)", 5, 35, 20, 5, key="b8_maxH")
    with col8c:
        st.markdown("**📉 Tỷ lệ khấu hao (%/năm)**")
        delta_K_val  = st.slider("δK vốn vật chất", 3, 10, 5, 1, key="b8_dK") / 100
        delta_D_val  = st.slider("δD hạ tầng số", 5, 20, 12, 1, key="b8_dD") / 100
        delta_AI_val = st.slider("δAI công nghệ AI", 10, 25, 15, 1, key="b8_dAI") / 100

    if st.button("▶️ Chạy tối ưu động", type="primary"):
        with st.spinner("Đang tối ưu (SLSQP)..."):
            try:
                import b08 as m8
            except:
                import importlib.util
                spec = importlib.util.spec_from_file_location("b08", THIS_DIR/"b08.py")
                m8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m8)
            m8.RHO_D = rho_val
            res = m8.run_b08(scenario_shock=show_shock)

        if res.get("converged"):
            st.success(f"✅ Hội tụ | GDP 2035: **{res['GDP_2035']} nghìn tỷ VND** | Welfare: {res['welfare']:.3f}")
        else:
            st.warning("⚠️ Chưa hội tụ hoàn toàn — kết quả gần đúng")

        show_img(res["img"])

        st.subheader("📋 Quỹ đạo tối ưu")
        st.dataframe(res["trajectory_display"], use_container_width=True)

    with st.expander("💡 Câu hỏi thảo luận"):
        st.markdown("""
        1. **Front-loaded hay back-loaded?** Với ρ=0.97 (quan tâm dài hạn), mô hình thường đề xuất đầu tư sớm vào D và H để tích lũy TFP.
        2. **Tỷ lệ IAI/IH**: Nhân lực phải đi trước hoặc đồng thời AI — ràng buộc H hấp thụ AI.
        3. **ρ=0.90 vs 0.97**: Chiết khấu ngắn hạn hơn giải thích vì sao chính phủ "dưới đầu tư" vào R&D.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# BÀI 9 — PHIÊN BẢN MỚI DÙNG PLOTLY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "b09":
    import plotly.graph_objects as go

    st.markdown('<div class="bai-header"><h3>Bài 9 · Tác động AI tới thị trường lao động Việt Nam</h3></div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Mục tiêu:** Tối đa NetJob ròng (Σ việc làm mới + nâng cấp − dịch chuyển)  
    **Ngân sách:** 30.000 tỷ VND | **8 ngành** | **Ràng buộc:** Displaced ≤ RetrainCapacity
    """)

    st.markdown("### 🎛️ Tham số điều chỉnh")
    col9a, col9b, col9c = st.columns(3)
    with col9a:
        st.markdown("**💰 Ngân sách**")
        budget_b9 = st.slider("Ngân sách tổng (tỷ VND)", 10000, 60000, 30000, 5000, key="b9_budget")
    with col9b:
        st.markdown("**📊 Sàn đầu tư mỗi ngành**")
        min_xH_b9  = st.slider("Sàn x_H mỗi ngành (tỷ)", 100, 2000, 500, 100, key="b9_minh")
        min_xAI_b9 = st.slider("Sàn x_AI mỗi ngành (tỷ)", 50, 1000, 200, 50, key="b9_minai")
    with col9c:
        st.markdown("**📈 Trần đầu tư mỗi ngành**")
        max_xH_b9  = st.slider("Trần x_H mỗi ngành (tỷ)", 2000, 15000, 5000, 500, key="b9_maxh")
        max_xAI_b9 = st.slider("Trần x_AI mỗi ngành (tỷ)", 1000, 10000, 3000, 500, key="b9_maxai")

    if st.button("▶️ Giải bài toán phân bổ lao động", type="primary"):
        with st.spinner("Đang tối ưu linprog..."):
            import importlib.util
            spec = importlib.util.spec_from_file_location("b09", THIS_DIR/"modules"/"b09.py")
            m9 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m9)
            m9.BUDGET = budget_b9
            res = m9.run_b09_scipy()

        if "error" in res:
            st.error(res["error"])
        else:
            df9 = res["optimal"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng NetJob ròng", f"{res['total_netjob']:,.0f} nghìn việc")
            c2.metric("Ngân sách sử dụng", f"{res['total_budget_used']:,.0f} tỷ")
            c3.metric("Ngưỡng x_H tối thiểu CN chế biến", f"{res['threshold_xH_sector2']:,.0f} tỷ")

            sectors9 = df9["Nganh"].tolist()

            tab1, tab2, tab3 = st.tabs(["📊 Phân bổ ngân sách", "📈 NetJob & Luồng lao động", "📋 Bảng kết quả"])

            with tab1:
                fig9a = go.Figure()
                fig9a.add_trace(go.Bar(
                    name="x_AI (tỷ VND)", x=sectors9, y=df9["x_AI (ty VND)"],
                    marker_color="#e74c3c"))
                fig9a.add_trace(go.Bar(
                    name="x_H (tỷ VND)", x=sectors9, y=df9["x_H (ty VND)"],
                    marker_color="#3498db"))
                fig9a.update_layout(
                    barmode="group",
                    title="Phân bổ ngân sách AI vs Nhân lực theo ngành",
                    xaxis_tickangle=-30, height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig9a, use_container_width=True)

            with tab2:
                fig9b = go.Figure(go.Bar(
                    x=sectors9,
                    y=df9["NetJob rong (nghin)"],
                    marker_color=["#27ae60" if v >= 0 else "#e74c3c"
                                  for v in df9["NetJob rong (nghin)"]],
                    text=[f"{v:,.0f}" for v in df9["NetJob rong (nghin)"]],
                    textposition="outside",
                ))
                fig9b.update_layout(
                    title="NetJob ròng theo ngành (nghìn việc làm)",
                    xaxis_tickangle=-30, height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig9b, use_container_width=True)

                # Luồng dịch chuyển
                fig9c = go.Figure()
                fig9c.add_trace(go.Bar(name="Nâng cấp (H)", x=sectors9,
                    y=df9["Nang cap (nghin)"], marker_color="#2ecc71"))
                fig9c.add_trace(go.Bar(name="Việc làm mới (AI)", x=sectors9,
                    y=df9["Viec lam moi (nghin)"], marker_color="#3498db"))
                fig9c.add_trace(go.Bar(name="Dịch chuyển (TĐH)", x=sectors9,
                    y=-df9["Dich chuyen (nghin)"], marker_color="#e74c3c"))
                fig9c.update_layout(
                    barmode="group",
                    title="Luồng dịch chuyển lao động theo ngành",
                    xaxis_tickangle=-30, height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig9c, use_container_width=True)

            with tab3:
                st.dataframe(df9, use_container_width=True, hide_index=True)

    with st.expander("💡 Câu hỏi thảo luận"):
        st.markdown("""
        1. **CN chế biến chế tạo** (risk 42%) cần đầu tư đào tạo lại nhiều nhất — phù hợp thực tế Việt Nam.
        2. **Tài chính-Ngân hàng** (risk 52%, a₁=45.8): AI tạo việc làm mới nhanh, nhưng cần đào tạo lại song song.
        3. **Nông-Lâm-Thủy sản**: Ít đầu tư AI (hệ số tạo việc thấp), ưu tiên H để nâng cấp kỹ năng lao động phổ thông.
        4. Ràng buộc **Displaced ≤ RetrainCap** = "tốc độ tự động hóa không vượt năng lực đào tạo lại".
        """)

# ══════════════════════════════════════════════════════════════════════════════
# BÀI 10
# ══════════════════════════════════════════════════════════════════════════════
elif page == "b10":
    st.markdown('<div class="bai-header"><h3>Bài 10 · Quy hoạch ngẫu nhiên hai giai đoạn</h3></div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Cấu trúc:** First-stage (65.000 tỷ) + Second-stage theo 4 kịch bản kinh tế thế giới  
    **Xác suất:** Lạc quan 30% | Cơ sở 45% | Bi quan 20% | Khủng hoảng 5%  
    **Tính:** VSS (Value of Stochastic Solution) & EVPI (Expected Value of Perfect Information)
    """)

    st.markdown("### 🎛️ Tham số điều chỉnh")
    col10a, col10b = st.columns(2)
    with col10a:
        st.markdown("**💰 Ngân sách (tỷ VND)**")
        budget1_b10 = st.slider("Ngân sách Stage 1", 40000, 75000, 65000, 5000, key="b10_b1")
        budget2_b10 = st.slider("Ngân sách Stage 2 (dự phòng)", 5000, 30000, 15000, 5000, key="b10_b2")
        st.info(f"Tổng: {budget1_b10+budget2_b10:,} tỷ VND")
    with col10b:
        st.markdown("**🎲 Xác suất kịch bản**")
        p1 = st.slider("p₁ Lạc quan", 0.05, 0.60, 0.30, 0.05, key="b10_p1")
        p2 = st.slider("p₂ Cơ sở", 0.10, 0.70, 0.45, 0.05, key="b10_p2")
        p3 = st.slider("p₃ Bi quan", 0.05, 0.40, 0.20, 0.05, key="b10_p3")
        p4 = max(0.01, round(1-p1-p2-p3, 2))
        st.markdown(f"p₄ Khủng hoảng = **{p4}**")

    if st.button("▶️ Giải bài toán Stochastic LP", type="primary"):
        with st.spinner("Đang giải 3 mô hình (SP, EV, WS)..."):
            try:
                import b10 as m10
            except:
                import importlib.util
                spec = importlib.util.spec_from_file_location("b10", THIS_DIR/"b10.py")
                m10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m10)
            res = m10.run_b10()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("SP Objective", f"{res['SP_obj']:,.0f}")
        c2.metric("EV Objective", f"{res['EV_obj']:,.0f}")
        c3.metric("WS (Perfect Info)", f"{res['WS_obj']:,.0f}")
        c4.metric("VSS = SP−EV", f"{res['VSS']:,.1f}", help="Lợi ích của tư duy xác suất")
        c5.metric("EVPI = WS−SP", f"{res['EVPI']:,.1f}", help="Giá trị thông tin hoàn hảo")

        vss_val = res['VSS']
        if vss_val < 0:
            vss_color = "#fff3cd"
            vss_text  = f"""**VSS = {vss_val:,.1f} (âm):** SP bị ràng buộc ngân sách first-stage chặt hơn EV, tạo ra bất lợi số học. Tuy nhiên SP duy trì 15.000 tỷ dự phòng để ứng phó kịch bản xấu. EVPI = {res['EVPI']:,.0f} nghìn tỷ dương mới là chỉ số quan trọng hơn."""
        else:
            vss_color = "#d4edda"
            vss_text  = f"""**VSS = {vss_val:,.1f} (dương):** Tư duy xác suất mang lại lợi ích thực sự. EVPI = {res['EVPI']:,.0f} nghìn tỷ là trần chi phí thu thập thêm thông tin dự báo."""

        st.markdown(f'<div style="background:{vss_color};border-left:4px solid #856404;padding:14px;border-radius:6px;margin:12px 0">{vss_text}</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Biểu đồ phân bổ", "📋 First-stage", "📋 Second-stage"])
        with tab1:
            show_img(res["img"])
        with tab2:
            st.dataframe(res["first_stage_df"].style.format("{:.0f}", subset=['SP first-stage (tỷ VND)','EV first-stage (tỷ VND)']), use_container_width=True)
            st.markdown("**Lợi ích GDP kỳ vọng theo kịch bản:**")
            st.json(res["scenario_obj"])
        with tab3:
            st.dataframe(res["second_stage_df"].style.format("{:.0f}"), use_container_width=True)

    with st.expander("💡 Câu hỏi thảo luận"):
        st.markdown("""
        1. **SP vs EV**: SP đầu tư H nhiều hơn vì H hoạt động tốt cả kịch bản xấu.
        2. **VSS > 0**: Tư duy xác suất mang lại GDP tăng thêm so với dùng kịch bản trung bình.
        3. **COVID-2020 & bão Yagi-2024**: Việt Nam "dưới đầu tư" nhân lực số như hàng hóa bảo hiểm.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# BÀI 11 — Q-learning với đầy đủ tham số
# ══════════════════════════════════════════════════════════════════════════════
elif page == "b11":
    st.markdown('<div class="bai-header"><h3>Bài 11 · Học tăng cường Q-learning cho chính sách kinh tế</h3></div>',
                unsafe_allow_html=True)
    st.markdown("""
    **MDP:** trạng thái = (GDP_growth, D, AI, Unemploy_risk) × 3 mức = **81 trạng thái**.  
    **5 hành động:** a0 truyền thống · a1 cân bằng · a2 số hóa nhanh · a3 AI dẫn dắt · a4 bao trùm.  
    **Reward:** $R = w_1\\Delta GDP - w_2\\Delta U - w_3 CyberRisk - w_4 Emission$, $w = (0.40, 0.25, 0.20, 0.15)$.
    """)

    st.markdown("### 🎛️ Tham số huấn luyện")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_ep = st.slider("Số episode", 1000, 15000, 3000, 500, key="b11_ep")
    with col2:
        alpha_val = st.slider("α (learning rate)", 0.01, 0.50, 0.10, 0.01, key="b11_alpha")
    with col3:
        gamma_val = st.slider("γ (discount)", 0.80, 0.99, 0.95, 0.01, key="b11_gamma")
    with col4:
        seed_val = st.number_input("Seed", min_value=0, max_value=999, value=42, step=1, key="b11_seed")

    if st.button("🤖 Train Q-learning", type="primary", key="b11_train"):
        with st.spinner(f"Đang huấn luyện {n_ep:,} episodes (α={alpha_val}, γ={gamma_val}, seed={seed_val})..."):
            import importlib.util as _ilu11
            spec11 = _ilu11.spec_from_file_location("b11", THIS_DIR/"modules"/"b11.py")
            m11 = _ilu11.module_from_spec(spec11)
            spec11.loader.exec_module(m11)

            Q, rewards = m11.train_qlearning(
                n_episodes=n_ep, alpha=alpha_val,
                gamma=gamma_val, seed=int(seed_val)
            )
            policy_df  = m11.extract_policy(Q)
            comparison = m11.compare_policies(Q, n_eval=50)
            img        = m11.plot_results(rewards, policy_df, comparison, Q=Q)

            vn_state = (1,1,0,1)
            ACTION_NAMES = ['Truyền thống','Cân bằng','Số hóa nhanh','AI dẫn dắt','Bao trùm']
            best_a = int(np.argmax(Q[vn_state]))
            special_states = {
                'GDP thấp, D thấp, AI thấp, U cao': (0,0,0,2),
                'GDP cao, AI cao, U thấp':           (2,1,2,0),
                'GDP TB, D cao, AI TB, U TB':        (1,2,1,1),
                'GDP thấp, D TB, AI cao, U cao':     (0,1,2,2),
            }
            special_df = pd.DataFrame([
                {'Trạng thái': k,
                 'Hành động tối ưu': ACTION_NAMES[int(np.argmax(Q[v]))],
                 'Q max': round(float(Q[v].max()), 3)}
                for k,v in special_states.items()
            ])

        st.success(f"✅ Done. Mean reward 100 ep cuối: {np.mean(rewards[-100:]):.2f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("🇻🇳 Hành động tối ưu VN 2026", ACTION_NAMES[best_a])
        c2.metric("Reward TB 500 ep cuối", f"{np.mean(rewards[-500:]):.4f}")
        c3.metric("Episodes đã train", f"{n_ep:,}")

        st.markdown("### 📈 Learning curve")
        show_img(img)

        st.markdown("### 📊 So sánh chính sách")
        comp_df = pd.DataFrame(comparison.items(), columns=["Chính sách","Reward TB"])
        comp_df = comp_df.sort_values("Reward TB", ascending=False)
        st.dataframe(comp_df.style.highlight_max(subset=["Reward TB"], color="#d4edda"),
                     use_container_width=True, hide_index=True)

        st.markdown("### 🗺️ Chính sách tối ưu tại trạng thái đặc biệt")
        st.dataframe(special_df, use_container_width=True, hide_index=True)

        img_path = os.path.join(THIS_DIR, 'modules', 'b12_aideom_vn', 'b11_enhanced.png')
        if os.path.exists(img_path):
            st.markdown("---")
            st.markdown("### 🔬 Nâng cấp: Q-table vs DQN · Policy Heatmap")
            st.image(img_path, use_container_width=True)

    with st.expander("💡 Câu hỏi thảo luận"):
        st.markdown("""
        1. **GDP thấp, D thấp, U cao** → thường chọn *Bao trùm* (a4) hoặc *Số hóa* (a2) — "quick win".
        2. **GDP cao, AI cao, U thấp** → chọn *AI dẫn dắt* (a3) — đẩy mạnh khi nền tảng đã vững.
        3. **Tích hợp π* vào chính sách**: Dùng như công cụ tư vấn, không phải autopilot.
        4. **α và γ ảnh hưởng thế nào?** α cao → học nhanh nhưng dễ dao động; γ cao → quan tâm dài hạn hơn.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# BÀI 12
# ══════════════════════════════════════════════════════════════════════════════
elif page == "b12":
    import importlib.util as _ilu, os as _os, sys as _sys
    _b12_dir = str(THIS_DIR / "modules" / "b12_aideom_vn")
    if _b12_dir not in _sys.path:
        _sys.path.insert(0, _b12_dir)
    try:
        _spec = _ilu.spec_from_file_location("b12_dashboard", _os.path.join(_b12_dir, "b12_dashboard.py"))
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
    except FileNotFoundError:
        st.error("❌ Không tìm thấy thư mục **modules/b12_aideom_vn/**\n\nHãy giải nén file `b12_aideom_vn.zip` vào `modules/` rồi thử lại.")
    except Exception as _e:
        st.error(f"❌ Lỗi load Bài 12: {_e}")
        st.exception(_e)

elif page == "nangcap":
    import os, numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    st.markdown("<h2 style='color:#C8102E'>🔬 Kỹ thuật nâng cao</h2>", unsafe_allow_html=True)

    tab_b01, tab_b07, tab_b11, tab_whatif, tab_map = st.tabs([
        "📐 Bài 1 OLS+Ridge","🧬 Bài 7 Hypervolume","🤖 Bài 11 Q vs DQN",
        "🎛️ What-if Simulator","🗺️ Bản đồ vùng",
    ])

    with tab_b01:
        img = os.path.join(THIS_DIR, "modules", "b12_aideom_vn", "b01_enhanced.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.warning("Chua co anh - chay b01_enhanced.py truoc")

    with tab_b07:
        img = os.path.join(THIS_DIR, "modules", "b12_aideom_vn", "b07_enhanced.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.warning("Chua co anh - chay b07_enhanced.py truoc")

    with tab_b11:
        img = os.path.join(THIS_DIR, "modules", "b12_aideom_vn", "b11_enhanced.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.warning("Chua co anh - chay b11_enhanced.py truoc")

    with tab_whatif:
        st.markdown("#### GDP What-if Simulator")
        col1, col2 = st.columns(2)
        with col1:
            w_K   = st.slider("Tang truong von K (%/nam)", 3, 12, 6)
            w_D   = st.slider("Kinh te so/GDP 2030 (%)", 20, 40, 30)
            w_AI  = st.slider("DN cong nghe so 2030 (nghin)", 80, 150, 100)
            w_H   = st.slider("LD qua dao tao 2030 (%)", 28, 45, 35)
            w_TFP = st.slider("TFP growth (%/nam)", 0.5, 3.0, 1.2, 0.1)
        with col2:
            import numpy as np
            Y0 = 12847.6; K0, L0 = 25900.0, 53.4
            alpha, beta_l, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
            A0 = 0.09
            years = list(range(2025, 2031))

            # Baseline (không nhiễu)
            gdp_base = []; K_t = K0
            for i in range(6):
                K_t = K_t*(1+w_K/100); L_t=L0*(1.01)**i
                D_t=19.5+(w_D-19.5)*i/5; AI_t=80+(w_AI-80)*i/5
                H_t=29.2+(w_H-29.2)*i/5; A_t=A0*(1+w_TFP/100)**i
                Y_t=A_t*K_t**alpha*L_t**beta_l*D_t**gamma*AI_t**delta*H_t**theta
                gdp_base.append(round(Y_t,0))

            # Kịch bản có nhiễu ngẫu nhiên + cú sốc 2028
            np.random.seed(42)
            gdp_shock = []; K_t2 = K0
            for i in range(6):
                K_t2 = K_t2*(1+w_K/100); L_t=L0*(1.01)**i
                D_t=19.5+(w_D-19.5)*i/5; AI_t=80+(w_AI-80)*i/5
                H_t=29.2+(w_H-29.2)*i/5; A_t=A0*(1+w_TFP/100)**i
                Y_t=A_t*K_t2**alpha*L_t**beta_l*D_t**gamma*AI_t**delta*H_t**theta
                # Nhiễu ngẫu nhiên ±1.5%
                noise = np.random.uniform(-0.015, 0.015)
                # Cú sốc 2028 (năm thứ 3, i=3) giảm 5%
                shock = -0.05 if i == 3 else 0
                Y_t = Y_t * (1 + noise + shock)
                gdp_shock.append(round(Y_t, 0))

            fig_wif = go.Figure()
            # Đường baseline
            fig_wif.add_trace(go.Scatter(
                x=years, y=gdp_base, mode="lines+markers",
                name="Baseline (không cú sốc)",
                line=dict(color="#C8102E", width=3),
                marker=dict(size=8)
            ))
            # Đường có nhiễu + cú sốc
            fig_wif.add_trace(go.Scatter(
                x=years, y=gdp_shock, mode="lines+markers",
                name="Có nhiễu + cú sốc 2028",
                line=dict(color="#FF8C00", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond")
            ))
            fig_wif.add_hline(y=Y0, line_dash="dot", line_color="gray",
                              annotation_text="GDP 2025")
            # Đánh dấu cú sốc 2028
            fig_wif.add_vrect(x0=2027.7, x1=2028.3,
                              fillcolor="orange", opacity=0.15,
                              annotation_text="Cú sốc<br>2028", annotation_position="top left")
            fig_wif.update_layout(
                title=f"GDP 2030: {gdp_base[-1]:,.0f} ngh.tỷ (baseline) | CAGR={((gdp_base[-1]/Y0)**(1/5)-1)*100:.1f}%",
                xaxis_title="Năm", yaxis_title="GDP (nghìn tỷ VND)",
                height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_wif, use_container_width=True)
            cagr=((gdp_base[-1]/Y0)**(1/5)-1)*100
            ca,cb,cc,cd=st.columns(4)
            ca.metric("GDP 2030 (baseline)",f"{gdp_base[-1]:,.0f}")
            cb.metric("CAGR",f"{cagr:.1f}%")
            cc.metric("Tăng thêm",f"+{gdp_base[-1]-Y0:,.0f}")
            cd.metric("GDP 2030 (có cú sốc)",f"{gdp_shock[-1]:,.0f}",
                      delta=f"{gdp_shock[-1]-gdp_base[-1]:,.0f}", delta_color="inverse")

    with tab_map:
        st.markdown("#### Ban do 6 vung kinh te xa hoi Viet Nam")
        df_map = pd.DataFrame({
            "Vung":["Trung du mien nui phia Bac","Dong bang song Hong","Bac Trung Bo DH Trung Bo","Tay Nguyen","Dong Nam Bo","Dong bang song Cuu Long"],
            "Digital Index":[38,78,55,32,82,48],"AI Readiness":[22,68,40,18,75,30],
            "GRDP nguoi":[57,152,88,69,159,81],"FDI ty USD":[3.5,20.0,8.2,0.8,18.5,2.1],
            "Lat":[21.8,21.0,16.5,13.0,10.8,10.0],"Lon":[104.5,106.0,107.5,108.0,106.7,105.5],
        })
        metric=st.selectbox("Chon chi so",["Digital Index","AI Readiness","GRDP nguoi","FDI ty USD"])
        fig_map=px.scatter_mapbox(df_map,lat="Lat",lon="Lon",size=metric,color=metric,hover_name="Vung",color_continuous_scale="RdYlGn",size_max=50,zoom=4.5,mapbox_style="carto-positron",height=450)
        fig_map.update_layout(margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_map,use_container_width=True)
        fig_bar=px.bar(df_map,x="Vung",y=metric,color=metric,color_continuous_scale="Blues",height=280)
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
        st.plotly_chart(fig_bar,use_container_width=True)
