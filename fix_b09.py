content = open("modules/b09.py", encoding="utf-8").read()

# Sửa sàn tối thiểu hợp lý hơn để LP phân bổ đều các ngành
old = "    # C2: Mỗi ngành x_H >= 200 tỷ (sàn đào tạo cơ bản)\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[N+i] = -1\n        A_ub_rows.append(row)\n        b_ub_rows.append(-200)\n\n    # C3: Mỗi ngành x_AI >= 100 tỷ\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[i] = -1\n        A_ub_rows.append(row)\n        b_ub_rows.append(-100)"

new = "    # C2: Mỗi ngành x_H >= 500 tỷ\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[N+i] = -1\n        A_ub_rows.append(row)\n        b_ub_rows.append(-500)\n\n    # C3: Mỗi ngành x_AI >= 200 tỷ\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[i] = -1\n        A_ub_rows.append(row)\n        b_ub_rows.append(-200)\n\n    # C6: Trần x_H mỗi ngành <= 8000 tỷ để tránh tập trung\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[N+i] = 1\n        A_ub_rows.append(row)\n        b_ub_rows.append(8000)\n\n    # C7: Trần x_AI mỗi ngành <= 5000 tỷ\n    for i in range(N):\n        row = np.zeros(2*N)\n        row[i] = 1\n        A_ub_rows.append(row)\n        b_ub_rows.append(5000)"

content = content.replace(old, new)
open("modules/b09.py", "w", encoding="utf-8").write(content)
print("Done:", "8000" in content)
