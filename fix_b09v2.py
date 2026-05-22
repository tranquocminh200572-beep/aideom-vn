# Ghi đè toàn bộ phần run_b09_scipy trong b09.py
content = open("modules/b09.py", encoding="utf-8").read()

new_func = """
def run_b09_scipy():
    from scipy.optimize import linprog
    net_coef_AI = A1 - C1 * RISK
    net_coef_H  = B1
    c = np.concatenate([-net_coef_AI, -net_coef_H])
    A_ub_rows, b_ub_rows = [], []
    # Tong ngan sach
    A_ub_rows.append(np.ones(2*N)); b_ub_rows.append(BUDGET)
    # San x_H moi nganh >= 500
    for i in range(N):
        row = np.zeros(2*N); row[N+i] = -1
        A_ub_rows.append(row); b_ub_rows.append(-500)
    # San x_AI moi nganh >= 200
    for i in range(N):
        row = np.zeros(2*N); row[i] = -1
        A_ub_rows.append(row); b_ub_rows.append(-200)
    # Tran x_H moi nganh <= 5000
    for i in range(N):
        row = np.zeros(2*N); row[N+i] = 1
        A_ub_rows.append(row); b_ub_rows.append(5000)
    # Tran x_AI moi nganh <= 3000
    for i in range(N):
        row = np.zeros(2*N); row[i] = 1
        A_ub_rows.append(row); b_ub_rows.append(3000)
    # NetJob >= 0
    for i in range(N):
        row = np.zeros(2*N)
        row[i] = -net_coef_AI[i]; row[N+i] = -net_coef_H[i]
        A_ub_rows.append(row); b_ub_rows.append(0)
    # Displaced <= RetrainCap
    for i in range(N):
        row = np.zeros(2*N)
        row[i] = C1[i]*RISK[i]; row[N+i] = -D1[i]
        A_ub_rows.append(row); b_ub_rows.append(0)
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    bounds = [(0, None)] * (2*N)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        return {"error": f"LP khong hoi tu: {res.message}"}
    x_AI = res.x[:N]
    x_H  = res.x[N:]
    NewJob       = A1 * x_AI
    UpgradeJob   = B1 * x_H
    DisplacedJob = C1 * RISK * x_AI
    NetJob       = NewJob + UpgradeJob - DisplacedJob
    coef2 = net_coef_AI[1]
    min_xH2 = max(0, -coef2 * BUDGET / B1[1])
    df = pd.DataFrame({
        "Nganh": SECTORS,
        "Lao dong (tr.)": LABOR,
        "Rui ro TDH (%)": (RISK*100).round(0),
        "x_AI (ty VND)": np.round(x_AI,1),
        "x_H (ty VND)": np.round(x_H,1),
        "Viec lam moi (nghin)": np.round(NewJob,1),
        "Nang cap (nghin)": np.round(UpgradeJob,1),
        "Dich chuyen (nghin)": np.round(DisplacedJob,1),
        "NetJob rong (nghin)": np.round(NetJob,1),
    })
    img_bar    = _plot_netjob(df, x_AI, x_H, NetJob)
    img_sankey = _plot_sankey(df, NewJob, UpgradeJob, DisplacedJob)
    return {
        "optimal": df,
        "total_netjob": round(NetJob.sum(), 0),
        "total_budget_used": round((x_AI+x_H).sum(), 0),
        "threshold_xH_sector2": round(min_xH2, 1),
        "img_bar": img_bar,
        "img_sankey": img_sankey,
        "net_coef_AI": net_coef_AI,
    }
"""

# Tim va thay the ham run_b09_scipy
import re
pattern = r"def run_b09_scipy\(\):.*?(?=\ndef |\Z)"
content_new = re.sub(pattern, new_func.strip(), content, flags=re.DOTALL)
open("modules/b09.py", "w", encoding="utf-8").write(content_new)
print("Done:", "x_AI (ty VND)" in content_new)
