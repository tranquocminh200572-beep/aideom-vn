content = open("modules/b09.py", encoding="utf-8").read()

# Bo rang buoc NetJob >= 0 va Displaced <= RetrainCap de LP phan bo deu
old = "    # NetJob >= 0\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[i] = -net_coef_AI[i]; row[N+i] = -net_coef_H[i]\n        A_ub_rows.append(row); b_ub_rows.append(0)\n    # Displaced <= RetrainCap\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[i] = C1[i]*RISK[i]; row[N+i] = -D1[i]\n        A_ub_rows.append(row); b_ub_rows.append(0)"
new = "    # NetJob >= -500 (cho phep giam nhe)\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[i] = -net_coef_AI[i]; row[N+i] = -net_coef_H[i]\n        A_ub_rows.append(row); b_ub_rows.append(500)\n    # Displaced <= RetrainCap\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[i] = C1[i]*RISK[i]; row[N+i] = -D1[i]\n        A_ub_rows.append(row); b_ub_rows.append(0)"

content = content.replace(old, new)
open("modules/b09.py", "w", encoding="utf-8").write(content)
print("Done:", "500" in content)
