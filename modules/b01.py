import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DATA = {
    "year": [2020, 2021, 2022, 2023, 2024, 2025],
    "Y":    [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
    "K":    [16500, 17800, 19600, 21300, 23500, 25900],
    "L":    [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
    "D":    [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
    "AI":   [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
    "H":    [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
    "growth": [2.91, 2.58, 8.02, 5.05, 7.09, 8.02],
}

def render():
    st.subheader("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa")
    st.markdown("**Mô hình:** $Y_t = A_t \\cdot K_t^\\alpha \\cdot L_t^\\beta \\cdot D_t^\\gamma \\cdot AI_t^\\delta \\cdot H_t^\\theta$")

    col_side, col_main = st.columns([1, 3])

    with col_side:
        st.markdown("#### Hệ số Cobb-Douglas")
        alpha = st.slider("α – Vốn vật chất (K)", 0.10, 0.55, 0.33, 0.01)
        beta  = st.slider("β – Lao động (L)",      0.10, 0.60, 0.42, 0.01)
        gamma = st.slider("γ – Số hóa (D)",         0.01, 0.25, 0.10, 0.01)
        delta = st.slider("δ – Năng lực AI",         0.01, 0.20, 0.08, 0.01)
        theta = st.slider("θ – Nhân lực số (H)",    0.01, 0.20, 0.07, 0.01)

        total = alpha + beta + gamma + delta + theta
        color = "green" if abs(total - 1.0) < 0.01 else "red"
        st.markdown(f"**Tổng hệ số:** :{color}[{total:.2f}]")

        st.markdown("---")
        st.markdown("#### Kịch bản dự báo 2030")
        sc_D   = st.number_input("Kinh tế số / GDP (%)", 15, 50, 30)
        sc_AI  = st.number_input("DN công nghệ (nghìn)", 80, 200, 100)
        sc_H   = st.number_input("Lao động qua ĐT (%)", 25, 50, 35)
        sc_KL  = st.number_input("K & L tăng (%/năm)",  3.0, 12.0, 6.0, 0.5)
        sc_TFP = st.number_input("TFP tăng (%/năm)",    0.0, 3.0,  1.2, 0.1)

    with col_main:
        Y  = np.array(DATA["Y"])
        K  = np.array(DATA["K"])
        L  = np.array(DATA["L"])
        D  = np.array(DATA["D"])
        AI = np.array(DATA["AI"])
        H  = np.array(DATA["H"])
        years = DATA["year"]
        N = len(years)

        A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
        Amean = A.mean()
        Yhat  = Amean * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
        mape  = float(np.mean(np.abs((Y - Yhat) / Y)) * 100)
        cagr  = float((Y[-1]/Y[0])**(1/(N-1)) - 1) * 100

        n_fore = 2030 - 2025
        K30 = K[-1] * (1 + sc_KL/100)**n_fore
        L30 = L[-1] * (1 + sc_KL/100)**n_fore
        A30 = Amean * (1 + sc_TFP/100)**n_fore
        Y30 = A30 * K30**alpha * L30**beta * sc_D**gamma * sc_AI**delta * sc_H**theta
        cagr30 = float((Y30/Y[-1])**(1/n_fore) - 1) * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAPE dự báo", f"{mape:.2f}%")
        k2.metric("TFP trung bình Ā", f"{Amean:.4f}")
        k3.metric("CAGR 2020–2025", f"{cagr:.2f}%")
        k4.metric("GDP dự báo 2030", f"{Y30:,.0f} nghìn tỷ")

        # Chart 1: GDP thực vs dự báo
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=years, y=Y,    name="Thực tế",
                                  line=dict(color="#c0392b", width=3),
                                  mode="lines+markers"))
        fig1.add_trace(go.Scatter(x=years, y=Yhat, name="Dự báo (Ā)",
                                  line=dict(color="#d4a017", width=2, dash="dash"),
                                  mode="lines+markers"))
        fig1.update_layout(title="GDP thực tế vs. Dự báo (nghìn tỷ VND)",
                           height=280, margin=dict(t=40,b=20,l=20,r=20),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 & 3 cạnh nhau
        c1, c2 = st.columns(2)
        with c1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=years, y=A, fill="tozeroy",
                                      line=dict(color="#1a2744", width=2),
                                      name="TFP Aₜ"))
            fig2.update_layout(title="Năng suất nhân tố tổng hợp Aₜ",
                               height=250, margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            contribs = []
            labels   = ["Vốn K (α)", "Lao động L (β)", "Số hóa D (γ)",
                        "AI (δ)", "Nhân lực H (θ)", "TFP (dư)"]
            colors   = ["#c0392b","#2563eb","#d4a017","#0d9488","#7c3aed","#16a34a"]
            for i in range(1, N):
                dlnY  = np.log(Y[i]/Y[i-1])
                parts = [
                    alpha * np.log(K[i]/K[i-1]),
                    beta  * np.log(L[i]/L[i-1]),
                    gamma * np.log(D[i]/D[i-1]),
                    delta * np.log(AI[i]/AI[i-1]),
                    theta * np.log(H[i]/H[i-1]),
                ]
                tfp = dlnY - sum(parts)
                contribs.append(parts + [tfp])

            fig3 = go.Figure()
            contribs_arr = np.array(contribs) * 100
            yr_labels = [str(y) for y in years[1:]]
            for j, (lab, col) in enumerate(zip(labels, colors)):
                fig3.add_trace(go.Bar(name=lab, x=yr_labels,
                                      y=contribs_arr[:, j],
                                      marker_color=col))
            fig3.update_layout(barmode="stack",
                               title="Phân rã tăng trưởng (điểm %)",
                               height=250, margin=dict(t=40,b=20,l=20,r=20),
                               legend=dict(font=dict(size=9)))
            st.plotly_chart(fig3, use_container_width=True)

        # Bảng phân rã
        rows = []
        for i in range(1, N):
            dlnY  = np.log(Y[i]/Y[i-1])
            parts = [
                alpha * np.log(K[i]/K[i-1]),
                beta  * np.log(L[i]/L[i-1]),
                gamma * np.log(D[i]/D[i-1]),
                delta * np.log(AI[i]/AI[i-1]),
                theta * np.log(H[i]/H[i-1]),
            ]
            tfp = dlnY - sum(parts)
            rows.append({
                "Năm": years[i],
                "GDP thực (%)": f"{DATA['growth'][i]:.2f}",
                "K (α)": f"{parts[0]*100:.2f}",
                "L (β)": f"{parts[1]*100:.2f}",
                "D (γ)": f"{parts[2]*100:.2f}",
                "AI (δ)": f"{parts[3]*100:.2f}",
                "H (θ)": f"{parts[4]*100:.2f}",
                "TFP": f"{tfp*100:.2f}",
            })
        st.markdown("**Bảng phân rã tăng trưởng 2020–2025 (điểm %)**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Dự báo 2030
        st.success(f"🔮 **GDP dự báo 2030:** {Y30:,.0f} nghìn tỷ VND "
                   f"| CAGR {cagr30:.2f}%/năm so với 2025 "
                   f"| Tăng {(Y30/Y[-1]-1)*100:.1f}% so với 2025")

        # Thảo luận chính sách
        with st.expander("📋 Thảo luận chính sách"):
            trend = "tăng" if A[-1] > A[0] else "giảm"
            st.markdown(f"""
**a) Xu hướng TFP:** TFP có xu hướng **{trend}** giai đoạn 2020–2025
({A[0]:.4f} → {A[-1]:.4f}).
{"Tăng trưởng đang dần chuyển sang chiều sâu, dựa vào năng suất." if A[-1] > A[0] else "Tăng trưởng chủ yếu dựa vào đầu vào (vốn, lao động), chất lượng cần cải thiện."}

**b) Yếu tố đóng góp lớn nhất (D, AI, H):**
Với hệ số γ={gamma}, δ={delta}, θ={theta} — yếu tố có hệ số lớn nhất là
{"Số hóa D" if gamma >= delta and gamma >= theta else "AI" if delta >= theta else "Nhân lực H"}.

**c) Mục tiêu 30% kinh tế số/GDP 2030:**
Kịch bản hiện tại D={sc_D}% — {"✅ Khả thi" if sc_D >= 28 else "⚠️ Cần nỗ lực lớn"}.
Cần tăng bình quân {((sc_D/19.5)**(1/5)-1)*100:.1f}%/năm từ mức 19.5% (2025).
Nghị quyết 57-NQ/TW và QĐ 749/QĐ-TTg tạo nền tảng pháp lý quan trọng.
""")
# ── NÂNG CẤP KỸ THUẬT ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Nâng cấp: OLS-CRS · Ridge (LOO-CV) · Bootstrap CI")
    
    import os
    img_path = os.path.join(os.path.dirname(__file__), 
                            "b12_aideom_vn", "b01_enhanced.png")
    if os.path.exists(img_path):
        st.image(img_path, 
                 caption="OLS-CRS · Ridge LOO-CV · Bootstrap CI (B=2000) · Growth Accounting · Dự báo 2030",
                 use_container_width=True)
    else:
        if st.button("▶ Tạo biểu đồ nâng cấp (OLS + Ridge + Bootstrap)"):
            import subprocess, sys
            script = os.path.join(os.path.dirname(__file__), 
                                  "b12_aideom_vn", "b01_enhanced.py")
            subprocess.run([sys.executable, script], 
                          cwd=os.path.join(os.path.dirname(__file__), "b12_aideom_vn"))
            st.rerun()