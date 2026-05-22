content = open("app.py", encoding="utf-8").read()
old = 'st.dataframe(res["trajectory"][cols_show].style.format({'
new = 'st.dataframe(res["trajectory_display"].style.format({'
content = content.replace(old, new)
open("app.py", "w", encoding="utf-8").write(content)
print("Done:", content.count("trajectory_display"))
