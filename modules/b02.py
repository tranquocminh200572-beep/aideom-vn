import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import linprog

def render():
    st.subheader("Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách số")
    st.markdown("**Mô hình:** max Z = 0.85x₁ + 1.20x₂ + 0.95x₃ + 1.35x₄")

    col_side, col_main = st.columns([1, 3])

    with col_side:
        st.markdown("#### Tham số bài toán")
        budget = st.slider("Ngân sách tổng (nghìn tỷ)", 80, 200, 100, 10)
        min_I  = st.slider("Sàn hạ tầng số x₁ (≥)", 10, 40, 25, 5)
        min_AI = st.slider("Sàn AI & dữ liệu x₂ (≥)", 5, 30, 15, 5)
        min_H  = st.slider("Sàn nhân lực số x₃ (≥)", 10, 40, 20, 5)
        min_RD = st.slider("Sàn R&D x₄ (≥)", 5, 20, 10, 5)
        tech_ratio = st.slider("Tỷ trọng công nghệ chiến lược (x₂+x₄) ≥ (%)", 25, 50, 35, 5)

        st.markdown("#### Hệ số tác động GDP")
        c1_val = st.number_input("c₁ – Hạ tầng số", 0.5, 2.0, 0.85, 0.05)
        c2_val = st.number_input("c₂ – AI & dữ liệu", 0.5, 2.0, 1.20, 0.05)
        c3_val = st.number_input("c₃ – Nhân lực số", 0.5, 2.0, 0.95, 0.05)
        c4_val = st.number_input("c₄ – R&D công nghệ", 0.5, 2.0, 1.35, 0.05)

    with col_main:
        c = [-c1_val, -c2_val, -c3_val, -c4_val]
        tr = tech_ratio / 100
        A_ub = [
            [1, 1, 1, 1],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [tr-1, tr, tr-1, tr],
        ]
        b_ub = [budget, -min_I, -min_AI, -min_H, -min_RD, 0]
        bounds = [(0, None)] * 4

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.success:
            x = res.x
            Z = -res.fun
            labels = ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D"]
            colors = ["#c0392b", "#2563eb", "#d4a017", "#16a34a"]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("GDP tăng kỳ vọng Z*", f"{Z:.1f} nghìn tỷ")
            k2.metric("Hiệu suất Z*/Ngân sách", f"{Z/budget:.3f}")
            k3.metric("Tổng phân bổ", f"{sum(x):.1f} nghìn tỷ")
            k4.metric("Còn dư ngân sách", f"{budget - sum(x):.1f} nghìn tỷ")

            c_left, c_right = st.columns(2)
            with c_left:
                fig = go.Figure(go.Bar(
                    x=[f"{v:.1f}" for v in x],
                    y=labels, orientation="h",
                    marker_color=colors,
                    text=[f"{v:.1f} nghìn tỷ" for v in x],
                    textposition="outside"
                ))
                fig.update_layout(title="Phân bổ ngân sách tối ưu",
                                  height=280, margin=dict(t=40,b=20,l=20,r=60),
                                  xaxis_title="Nghìn tỷ VND")
                st.plotly_chart(fig, use_container_width=True)

            with c_right:
                fig2 = go.Figure(go.Pie(
                    labels=labels, values=x,
                    marker_colors=colors,
                    hole=0.4
                ))
                fig2.update_layout(title="Tỷ trọng phân bổ (%)",
                                   height=280, margin=dict(t=40,b=20,l=20,r=20))
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("**Kết quả phân bổ tối ưu:**")
            df_res = pd.DataFrame({
                "Hạng mục": labels,
                "Phân bổ (nghìn tỷ)": [f"{v:.2f}" for v in x],
                "Tỷ trọng (%)": [f"{v/sum(x)*100:.1f}" for v in x],
                "Hệ số tác động": [c1_val, c2_val, c3_val, c4_val],
                "GDP kỳ vọng": [f"{v*ci:.2f}" for v, ci in zip(x, [c1_val,c2_val,c3_val,c4_val])],
            })
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            # ══════════════════════════════════════════════════════
            # CÂU 2.4.2 – SHADOW PRICE (GIÁ ĐỐI NGẪU)
            # ══════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 💰 Câu 2.4.2 – Giá đối ngẫu (Shadow Price / Dual Values)")

            constraint_names = [
                "Ngân sách tổng (≤ B)",
                "Sàn hạ tầng x₁ (≥ min_I)",
                "Sàn AI x₂ (≥ min_AI)",
                "Sàn nhân lực x₃ (≥ min_H)",
                "Sàn R&D x₄ (≥ min_RD)",
                "Tỷ lệ công nghệ chiến lược",
            ]
            # Shadow price từ HiGHS dual solution
            if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                duals = list(res.ineqlin.marginals)
            else:
                # Tính xấp xỉ shadow price bằng perturbation
                duals = []
                eps = 0.1
                for k in range(len(b_ub)):
                    b2 = b_ub.copy()
                    b2[k] += eps
                    r2 = linprog(c, A_ub=A_ub, b_ub=b2, bounds=bounds, method="highs")
                    if r2.success:
                        duals.append(round((-r2.fun - Z) / eps, 4))
                    else:
                        duals.append(0.0)

            # Đảo dấu (linprog minimize, shadow price của maximize ngược dấu)
            duals_display = [-d for d in duals]

            df_shadow = pd.DataFrame({
                "Ràng buộc": constraint_names,
                "RHS (giá trị ràng buộc)": [budget, -min_I, -min_AI, -min_H, -min_RD, 0],
                "Trạng thái": ["Ràng buộc bất đẳng thức"] * 6,
                "Shadow Price (nghìn tỷ GDP / nghìn tỷ ngân sách)": [round(d, 4) for d in duals_display],
                "Ý nghĩa": [
                    f"Tăng 1 nghìn tỷ ngân sách → GDP kỳ vọng tăng {abs(duals_display[0]):.4f} nghìn tỷ",
                    f"Nới sàn x₁ thêm 1 nghìn tỷ → GDP {'tăng' if duals_display[1]>0 else 'giảm'} {abs(duals_display[1]):.4f}",
                    f"Nới sàn x₂ thêm 1 nghìn tỷ → GDP {'tăng' if duals_display[2]>0 else 'giảm'} {abs(duals_display[2]):.4f}",
                    f"Nới sàn x₃ thêm 1 nghìn tỷ → GDP {'tăng' if duals_display[3]>0 else 'giảm'} {abs(duals_display[3]):.4f}",
                    f"Nới sàn x₄ thêm 1 nghìn tỷ → GDP {'tăng' if duals_display[4]>0 else 'giảm'} {abs(duals_display[4]):.4f}",
                    f"Ràng buộc công nghệ chiến lược: shadow price = {duals_display[5]:.4f}",
                ],
            })
            st.dataframe(df_shadow, use_container_width=True, hide_index=True)

            # Biểu đồ shadow price
            fig_sp = go.Figure(go.Bar(
                x=constraint_names,
                y=[round(d, 4) for d in duals_display],
                marker_color=["#c0392b" if d > 0 else "#94a3b8" for d in duals_display],
                text=[f"{d:.4f}" for d in duals_display],
                textposition="outside",
            ))
            fig_sp.update_layout(
                title="Shadow Price của từng ràng buộc",
                xaxis_title="Ràng buộc",
                yaxis_title="Shadow Price (ngh.tỷ GDP / ngh.tỷ đầu tư)",
                height=300, margin=dict(t=40,b=80,l=20,r=20),
                xaxis_tickangle=-20,
            )
            st.plotly_chart(fig_sp, use_container_width=True)

            st.info(f"""
**📌 Giải thích Shadow Price ngân sách tổng:**  
Shadow price = **{abs(duals_display[0]):.4f}** nghìn tỷ GDP / nghìn tỷ đầu tư.  
→ Mỗi 1 nghìn tỷ VND tăng thêm vào ngân sách sẽ tạo ra thêm **{abs(duals_display[0]):.4f} nghìn tỷ VND** GDP kỳ vọng.  
→ Đây là **cận trên hợp lý** của chi phí cơ hội vốn công — chính phủ chỉ nên vay thêm nếu lãi suất < {abs(duals_display[0])*100:.1f}%.
""")

            # ══════════════════════════════════════════════════════
            # CÂU 2.4.3 – ĐỘ NHẠY NGÂN SÁCH
            # ══════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 📈 Câu 2.4.3 – Phân tích độ nhạy Z*(B)")
            budgets = list(range(80, 201, 10))
            zs = []
            for b in budgets:
                b_ub2 = [b, -min_I, -min_AI, -min_H, -min_RD, 0]
                r2 = linprog(c, A_ub=A_ub, b_ub=b_ub2, bounds=bounds, method="highs")
                zs.append(-r2.fun if r2.success else None)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=budgets, y=zs, mode="lines+markers",
                                      line=dict(color="#c0392b", width=2),
                                      name="Z*(B)"))
            fig3.add_vline(x=budget, line_dash="dash", line_color="#d4a017",
                           annotation_text=f"Hiện tại: {budget}")
            fig3.update_layout(title="Phân tích độ nhạy: Z* theo ngân sách B",
                               xaxis_title="Ngân sách (nghìn tỷ)",
                               yaxis_title="GDP tăng kỳ vọng Z*",
                               height=280, margin=dict(t=40,b=30,l=20,r=20))
            st.plotly_chart(fig3, use_container_width=True)

            # ══════════════════════════════════════════════════════
            # CÂU 2.4.4 – RÀNG BUỘC x₃ ≥ 30
            # ══════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("### 🔒 Câu 2.4.4 – Ưu tiên nhân lực số: x₃ ≥ 30")
            b_ub_new = [budget, -min_I, -min_AI, -30, -min_RD, 0]
            res_new = linprog(c, A_ub=A_ub, b_ub=b_ub_new, bounds=bounds, method="highs")
            if res_new.success:
                Z_new = -res_new.fun
                delta_Z = Z_new - Z
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Z* (x₃ ≥ 30)", f"{Z_new:.2f} ngh.tỷ")
                col_b.metric("Z* gốc (x₃ ≥ 20)", f"{Z:.2f} ngh.tỷ")
                col_c.metric("Thay đổi ΔZ*", f"{delta_Z:.2f} ngh.tỷ",
                             delta_color="inverse" if delta_Z < 0 else "normal")
                st.success(f"✅ Bài toán **còn khả thi**. Ưu tiên nhân lực số làm Z* {'giảm' if delta_Z<0 else 'tăng'} {abs(delta_Z):.2f} nghìn tỷ ({delta_Z/Z*100:.1f}%).")
            else:
                st.error("❌ Bài toán **không khả thi** khi x₃ ≥ 30 với ngân sách và sàn hiện tại!")

            with st.expander("📋 Thảo luận chính sách"):
                st.markdown(f"""
**a) Shadow price ngân sách:** Mỗi 1 nghìn tỷ tăng thêm → GDP kỳ vọng tăng ~**{abs(duals_display[0]):.4f} nghìn tỷ**.
Đây là cận trên hợp lý của chi phí cơ hội vốn công.

**b) Vì sao R&D có hệ số cao nhất nhưng sàn thấp nhất?** R&D có tác động lan tỏa dài hạn (hệ số {c4_val}) nhưng kết quả chậm → nhà nước đặt sàn thấp, phần còn lại để thị trường quyết định.

**c) Tỷ lệ {tech_ratio}% công nghệ chiến lược:** Theo QĐ 749/QĐ-TTg, tỷ lệ này {"khả thi" if tech_ratio <= 40 else "đòi hỏi tái cơ cấu ngân sách lớn"} trong bối cảnh ngân sách 2025.
""")
        else:
            st.error("❌ Bài toán không có nghiệm khả thi! Kiểm tra lại các ràng buộc sàn.")
