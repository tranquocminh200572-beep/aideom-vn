content = open("app.py", encoding="utf-8").read()

old = """    col_s, col_r = st.columns([2,1])
    with col_s:
        show_shock = st.toggle("Mô phỏng cú sốc 2028 (-8% GDP)", value=False)
        rho_val = st.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01)"""

new = """    st.markdown("### 🎛️ Tham số điều chỉnh")
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
        delta_AI_val = st.slider("δAI công nghệ AI", 10, 25, 15, 1, key="b8_dAI") / 100"""

content = content.replace(old, new)
open("app.py", "w", encoding="utf-8").write(content)
print("Done:", "b8_rho" in content)
