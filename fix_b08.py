content = open("app.py", encoding="utf-8").read()
old = '    col_s, col_r = st.columns([2,1])\n    with col_s:\n        show_shock = st.toggle("Mo phong cu soc 2028 (-8% GDP)", value=False)\n        rho_val = st.slider("He so chiet khau p", 0.85, 0.99, 0.97, 0.01)'
print("Found old:", old[:30] in content)
