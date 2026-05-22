content = open("app.py", encoding="utf-8").read()

# Thay toàn bộ đoạn dùng trajectory_display thành trajectory
old = 'st.dataframe(res["trajectory_display"].style.format({'
new = 'st.dataframe(res["trajectory"].style.format({'

content = content.replace(old, new)

# Sửa format keys cho đúng với tên cột mới trong b08.py
old2 = "'GDP (nghìn tỷ VND)': '{:.2f}',"
new2 = "'GDP_nghin_ty': '{:.2f}',"
content = content.replace(old2, new2)

old3 = "'Tiêu dùng': '{:.0f}',"
new3 = "'Tieu_dung_pct': '{:.0f}',"
content = content.replace(old3, new3)

old4 = "'Đầu tư AI (IAI)': '{:.0f}',"
new4 = "'Dau_tu_AI_pct': '{:.0f}',"
content = content.replace(old4, new4)

old5 = "'Đầu tư nhân lực (IH)': '{:.0f}',"
new5 = "'Dau_tu_H_pct': '{:.0f}',"
content = content.replace(old5, new5)

old6 = "'TFP A': '{:.3f}',"
new6 = "'TFP_A': '{:.3f}',"
content = content.replace(old6, new6)

open("app.py", "w", encoding="utf-8").write(content)
print("Done")
