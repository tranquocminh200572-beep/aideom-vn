content = open("app.py", encoding="utf-8").read()

old = '    if st.button("▶️ Giải bài toán Stochastic LP", type="primary"):\n        with st.spinner("Đang giải 3 mô hình (SP, EV, WS)..."):'

new = '''    st.markdown("### 🎛️ Tham số điều chỉnh")
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
        with st.spinner("Đang giải 3 mô hình (SP, EV, WS)..."):'''

content = content.replace(old, new)
open("app.py", "w", encoding="utf-8").write(content)
print("Done:", "b10_b1" in content)
