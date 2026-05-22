content = open("app.py", encoding="utf-8").read()

old = """            with tab1:
                show_img(res["img_bar"])
                st.divider()
                show_img(res["img_sankey"])
            with tab2:
                st.dataframe(res["optimal"].style.format({
                    'x_AI (ty VND)': '{:.0f}',
                    'x_H (ty VND)': '{:.0f}',
                    'NetJob rong (nghin)': '{:+.0f}',
                }), use_container_width=True)"""

new = """            with tab1:
                import plotly.graph_objects as go
                df9 = res["optimal"]
                sectors9 = df9["Nganh"].tolist()
                fig9a = go.Figure()
                fig9a.add_trace(go.Bar(name="x_AI", x=sectors9, y=df9["x_AI (ty VND)"], marker_color="#e74c3c"))
                fig9a.add_trace(go.Bar(name="x_H", x=sectors9, y=df9["x_H (ty VND)"], marker_color="#3498db"))
                fig9a.update_layout(barmode="group", title="Phan bo ngan sach AI vs Nhan luc", height=320, xaxis_tickangle=-30)
                st.plotly_chart(fig9a, use_container_width=True)
                fig9b = go.Figure(go.Bar(x=sectors9, y=df9["NetJob rong (nghin)"],
                    marker_color=["green" if v>=0 else "red" for v in df9["NetJob rong (nghin)"]]))
                fig9b.update_layout(title="NetJob rong theo nganh", height=300, xaxis_tickangle=-30)
                st.plotly_chart(fig9b, use_container_width=True)
            with tab2:
                st.dataframe(res["optimal"], use_container_width=True)"""

content = content.replace(old, new)
open("app.py", "w", encoding="utf-8").write(content)
print("Done:", "fig9a" in content)
