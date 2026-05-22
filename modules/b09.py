"""
Bài 9: Tác động AI tới thị trường lao động Việt Nam
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

SECTORS = [
    'Nông-Lâm-Thủy sản',
    'CN chế biến chế tạo',
    'Xây dựng',
    'Bán buôn-bán lẻ',
    'Tài chính-Ngân hàng',
    'Logistics-Vận tải',
    'CNTT-Truyền thông',
    'Giáo dục-Đào tạo',
]

LABOR  = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
RISK   = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
A1     = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
A2     = np.array([12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0])
B1     = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
C1     = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1     = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])

BUDGET = 30_000
N = len(SECTORS)


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
    # NetJob >= -500 (cho phep giam nhe)
    for i in range(N):
        row = np.zeros(2*N)
        row[i] = -net_coef_AI[i]; row[N+i] = -net_coef_H[i]
        A_ub_rows.append(row); b_ub_rows.append(500)
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
def _plot_netjob(df, x_AI, x_H, NetJob):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(N)
    ax = axes[0]
    ax.bar(x, x_AI, label='x_AI (tỷ VND)', color='#e74c3c')
    ax.bar(x, x_H, bottom=x_AI, label='x_H (tỷ VND)', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(SECTORS, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Tỷ VND')
    ax.set_title('Phân bổ ngân sách AI vs Nhân lực theo ngành')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    colors = ['green' if v >= 0 else 'red' for v in NetJob]
    ax2.bar(x, NetJob, color=colors)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SECTORS, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Nghìn việc làm ròng')
    ax2.set_title('NetJob ròng theo ngành')
    ax2.grid(axis='y', alpha=0.3)

    for i, v in enumerate(NetJob):
        ax2.text(i, v + (50 if v >= 0 else -100), f'{v:.0f}', ha='center', fontsize=7)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


def _plot_sankey(df, NewJob, UpgradeJob, DisplacedJob):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(N)
    w = 0.25
    ax.bar(x - w, UpgradeJob,   width=w, label='Việc làm nâng cấp (H)', color='#2ecc71')
    ax.bar(x,     NewJob,        width=w, label='Việc làm mới (AI)',      color='#3498db')
    ax.bar(x + w, -DisplacedJob, width=w, label='Việc làm mất (TĐH)',     color='#e74c3c')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(SECTORS, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Nghìn việc làm')
    ax.set_title('Luồng dịch chuyển lao động theo ngành — Bài 9')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == '__main__':
    res = run_b09_scipy()
    if 'error' not in res:
        print(f"Tổng NetJob ròng: {res['total_netjob']:,.0f} nghìn việc làm")
        print(f"Ngân sách dùng: {res['total_budget_used']:,.0f} tỷ VND / {BUDGET:,}")
        print(f"Ngưỡng x_H tối thiểu ngành 2: {res['threshold_xH_sector2']:,.1f} tỷ VND")
        print(res['optimal'][['Ngành','x_AI (tỷ VND)','x_H (tỷ VND)','NetJob ròng (nghìn)']].to_string(index=False))
    else:
        print(res['error'])
