import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

SECTORS = {
    "Tên ngành": [
        "Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng","Khai khoáng",
        "Bán buôn-bán lẻ","Tài chính-Ngân hàng","Logistics-Vận tải",
        "CNTT-Truyền thông","Giáo dục-Đào tạo","Y tế"
    ],
    "Tăng trưởng (%)": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
    "Năng suất (tr.VND/LĐ)": [103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1],
    "Lan tỏa (0-1)": [0.35,0.78,0.42,0.30,0.55,0.85,0.72,0.92,0.65,0.60],
    "XK (tỷ USD)": [40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0.0,0.0],
    "Việc làm (tr.LĐ)": [13.20,11.50,4.80,0.30,7.80,0.55,1.95,0.62,2.15,0.75],
    "AI Readiness (0-100)": [15,55,20,30,48,72,42,88,38,45],
    "Rủi ro TĐH (%)": [18,42,25,55,38,52,35,28,22,18],
}

def render():
    st.subheader("Bài 3 — Chỉ số ưu tiên ngành cho 10 ngành kinh tế Việt Nam")
    st.markdown("**Công thức:** Priority$_i$ = $a_1$·Growth + $a_2$·Productivity + $a_3$·Spillover + $a_4$·Export + $a_5$·Employment + $a_6$·AIReadiness − $a_7$·Risk")

    col_side, col_main = st.columns([1, 3])
    df = pd.DataFrame(SECTORS)

    with col_side:
        st.markdown("#### Bộ trọng số")
        preset = st.radio("Chọn nhanh:", ["Mặc định","Tăng trưởng","Bao trùm","Tùy chỉnh"])

        if preset == "Mặc định":
            defaults = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.05]
        elif preset == "Tăng trưởng":
            defaults = [0.25, 0.20, 0.10, 0.20, 0.05, 0.15, 0.05]
        elif preset == "Bao trùm":
            defaults = [0.10, 0.05, 0.20, 0.05, 0.25, 0.15, 0.20]
        else:
            defaults = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.05]

        labels_w = ["a₁ Tăng trưởng","a₂ Năng suất","a₃ Lan tỏa",
                    "a₄ Xuất khẩu","a₅ Việc làm","a₆ AI Readiness","a₇ Rủi ro TĐH"]
        weights_raw = []
        for i, (lab, d) in enumerate(zip(labels_w, defaults)):
            w = st.slider(lab, 0.0, 0.5, d, 0.01, key=f"w{i}")
            weights_raw.append(w)

        total_w = sum(weights_raw)
        # Tự động chuẩn hóa về tổng = 1.0
        if total_w > 0:
            weights = [w / total_w for w in weights_raw]
        else:
            weights = [1/7] * 7

        color_w = "green" if abs(total_w - 1.0) < 0.02 else "orange"
        st.markdown(f"**Tổng trọng số gốc:** :{color_w}[{total_w:.2f}]")
        st.markdown(f"**Sau chuẩn hóa:** :green[{sum(weights):.2f}]")
        if abs(total_w - 1.0) >= 0.02:
            st.info("⚡ Trọng số đã được tự động chuẩn hóa về tổng = 1.0")

        # Hiển thị trọng số đã chuẩn hóa
        st.markdown("**Trọng số thực dùng:**")
        for lab, w in zip(labels_w, weights):
            st.markdown(f"- {lab}: **{w:.3f}**")

    with col_main:
        def norm_good(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
        def norm_bad(s):  return (s.max() - s) / (s.max() - s.min() + 1e-9)

        cols_good = ["Tăng trưởng (%)","Năng suất (tr.VND/LĐ)","Lan tỏa (0-1)",
                     "XK (tỷ USD)","Việc làm (tr.LĐ)","AI Readiness (0-100)"]
        col_bad   = "Rủi ro TĐH (%)"

        nw = np.array(weights)  # đã chuẩn hóa, tổng = 1
        Xg = np.column_stack([norm_good(df[c]) for c in cols_good])
        Xb = norm_bad(df[col_bad]).values
        priority = Xg @ nw[:6] - nw[6] * Xb

        df["Priority"] = priority
        df_sorted = df.sort_values("Priority", ascending=False).reset_index(drop=True)
        df_sorted["Hạng"] = range(1, len(df_sorted)+1)

        k1, k2, k3 = st.columns(3)
        k1.metric("🥇 Ưu tiên cao nhất", df_sorted.iloc[0]["Tên ngành"])
        k2.metric("🥈 Ưu tiên thứ 2",    df_sorted.iloc[1]["Tên ngành"])
        k3.metric("🥉 Ưu tiên thứ 3",    df_sorted.iloc[2]["Tên ngành"])

        colors_bar = ["#c0392b" if i < 3 else "#94a3b8" for i in range(len(df_sorted))]
        fig = go.Figure(go.Bar(
            x=df_sorted["Priority"],
            y=df_sorted["Tên ngành"],
            orientation="h",
            marker_color=colors_bar,
            text=[f"{v:.3f}" for v in df_sorted["Priority"]],
            textposition="outside"
        ))
        fig.update_layout(title=f"Xếp hạng ưu tiên ngành — bộ trọng số '{preset}'",
                          height=380, margin=dict(t=40,b=20,l=20,r=60),
                          xaxis_title="Chỉ số Priority",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap ma trận chuẩn hóa
        norm_matrix = pd.DataFrame(Xg, columns=cols_good)
        norm_matrix["Rủi ro TĐH (đảo)"] = Xb
        norm_matrix.index = df["Tên ngành"]

        fig2 = px.imshow(norm_matrix.T, aspect="auto",
                         color_continuous_scale="RdYlGn",
                         title="Heatmap ma trận chuẩn hóa min-max")
        fig2.update_layout(height=300, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig2, use_container_width=True)

        # ── Phân tích độ nhạy a₆ (AI Readiness) ──────────────────
        st.markdown("---")
        st.markdown("### 📊 Câu 3.4.3 – Phân tích độ nhạy trọng số a₆ (AI Readiness)")
        ai_range = np.arange(0.05, 0.45, 0.05)
        top3_by_ai = []
        for a6 in ai_range:
            # Chuẩn hóa lại: giữ tỷ lệ các trọng số khác, thay a6
            remaining = 1.0 - a6
            other_weights = [w for w in weights_raw[:5]] + [weights_raw[6]]
            sum_other = sum(other_weights) + 1e-9
            nw_sens = np.array([w/sum_other*remaining for w in other_weights[:5]] + [a6] + [other_weights[5]/sum_other*remaining])
            p_sens = Xg @ nw_sens[:6] - nw_sens[6] * Xb
            top3 = df["Tên ngành"].iloc[np.argsort(-p_sens)[:3]].tolist()
            top3_by_ai.append({"a₆": round(a6, 2), "Top1": top3[0], "Top2": top3[1], "Top3": top3[2]})

        df_sens = pd.DataFrame(top3_by_ai)
        st.dataframe(df_sens, use_container_width=True, hide_index=True)

        # Bảng kết quả đầy đủ
        st.markdown("**Bảng xếp hạng đầy đủ:**")
        df_show = df_sorted[["Hạng","Tên ngành","Priority",
                              "Tăng trưởng (%)","AI Readiness (0-100)","Rủi ro TĐH (%)"]].copy()
        df_show["Priority"] = df_show["Priority"].round(4)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # ── So sánh 2 bộ trọng số ─────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚖️ Câu 3.4.4 – So sánh Tăng trưởng vs Bao trùm")
        w_growth = np.array([0.25, 0.20, 0.10, 0.20, 0.05, 0.15, 0.05])
        w_inclusive = np.array([0.10, 0.05, 0.20, 0.05, 0.25, 0.15, 0.20])
        p_growth   = Xg @ w_growth[:6]   - w_growth[6]   * Xb
        p_inclusive= Xg @ w_inclusive[:6] - w_inclusive[6] * Xb
        top3_g = df["Tên ngành"].iloc[np.argsort(-p_growth)[:3]].tolist()
        top3_i = df["Tên ngành"].iloc[np.argsort(-p_inclusive)[:3]].tolist()

        cg, ci = st.columns(2)
        cg.success(f"**Tăng trưởng Top-3:**\n1. {top3_g[0]}\n2. {top3_g[1]}\n3. {top3_g[2]}")
        ci.info(f"**Bao trùm Top-3:**\n1. {top3_i[0]}\n2. {top3_i[1]}\n3. {top3_i[2]}")

        with st.expander("📋 Thảo luận chính sách"):
            top3 = df_sorted.head(3)["Tên ngành"].tolist()
            st.markdown(f"""
**a) Ba ngành ưu tiên hàng đầu:** {', '.join(top3)}
Kết quả này {"phù hợp" if "CNTT-Truyền thông" in top3 else "cần đối chiếu"} với Nghị quyết 57-NQ/TW về ưu tiên ngành công nghệ số.

**b) Tại sao Khai khoáng không được ưu tiên?** Dù năng suất rất cao (1.290 tr.VND/LĐ), ngành này có rủi ro tự động hóa cao nhất (55%), lan tỏa thấp (0.30) và tăng trưởng âm (-1.20%) năm 2024.

**c) Ai quyết định bộ trọng số?** Cần quy trình đối thoại công khai giữa chuyên gia kỹ thuật, hội đồng chính sách và đại diện xã hội để bảo đảm tính chính danh.
""")
