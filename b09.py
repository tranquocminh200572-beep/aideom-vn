"""
Bài 9: Tác động AI tới thị trường lao động Việt Nam
Tối ưu phân bổ ngân sách 30.000 tỷ cho x_AI và x_H để maximize NetJob
NetJob_i = NewJob + UpgradeJob - DisplacedJob >= 0
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

LABOR  = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])  # triệu LĐ
RISK   = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
A1     = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
A2     = np.array([12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0])
B1     = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
C1     = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1     = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])

BUDGET = 30_000
N = len(SECTORS)


def run_b09_scipy():
    """Giải bằng scipy linprog (LP tuyến tính)"""
    from scipy.optimize import linprog

    # Biến: [x_AI(8), x_H(8)] = 16 biến
    # NetJob_i = (a1_i - c1_i*risk_i)*x_AI_i + b1_i*x_H_i >= 0
    # Displaced_i <= RetrainCap_i => c1_i*risk_i*x_AI_i <= d1_i*x_H_i
    # Tối đa Σ NetJob_i => minimize -Σ NetJob_i

    net_coef_AI = A1 - C1 * RISK  # hệ số x_AI trong NetJob
    net_coef_H  = B1               # hệ số x_H  trong NetJob

    # c vector (minimize -NetJob)
    c = np.concatenate([-net_coef_AI, -net_coef_H])

    # Ràng buộc bất đẳng thức (A_ub @ x <= b_ub)
    A_ub_rows, b_ub_rows = [], []

    # C1: Σ(x_AI + x_H) <= 30000
    row = np.ones(2*N)
    A_ub_rows.append(row); b_ub_rows.append(BUDGET)

    # C_min: mỗi ngành tối thiểu x_H >= 100 tỷ (đảm bảo có đào tạo cơ bản)
    for i in range(N):
        row = np.zeros(2*N)
        row[N+i] = -1  # -x_H_i <= -100
        A_ub_rows.append(row); b_ub_rows.append(-100)

    # C_min_ai: mỗi ngành tối thiểu x_AI >= 50 tỷ để có ứng dụng AI cơ bản
    for i in range(N):
        row = np.zeros(2*N)
        row[i] = -1
        A_ub_rows.append(row); b_ub_rows.append(-50)

    # NetJob_i >= 0 → -(net_AI_i*x_AI_i + b1_i*x_H_i) <= 0
    for i in range(N):
        row = np.zeros(2*N)
        row[i]   = -net_coef_AI[i]
        row[N+i] = -net_coef_H[i]
        A_ub_rows.append(row); b_ub_rows.append(0)

    # Displaced <= RetrainCap: c1*risk*x_AI - d1*x_H <= 0
    for i in range(N):
        row = np.zeros(2*N)
        row[i]   =  C1[i] * RISK[i]
        row[N+i] = -D1[i]
        A_ub_rows.append(row); b_ub_rows.append(0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    bounds = [(0, None)] * (2*N)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.status != 0:
        return {'error': f'Không tìm được nghiệm: {res.message}'}

    x_AI = res.x[:N]
    x_H  = res.x[N:]

    NewJob      = A1 * x_AI + A2 * 0   # a2 × x_D (x_D=0 ở đây)
    UpgradeJob  = B1 * x_H
    DisplacedJob= C1 * RISK * x_AI
    NetJob      = NewJob + UpgradeJob - DisplacedJob

    # Câu 9.4.2: Ngưỡng x_H tối thiểu ngành 2 khi x_AI tối đa
    max_xAI2 = BUDGET  # cực đoan
    min_xH2  = (C1[1]*RISK[1] - (A1[1]-0)*1) * max_xAI2 / B1[1]  # xấp xỉ
    # Công thức chính xác: NetJob2>=0 <=> (a1-c1*risk)*x_AI2 + b1*x_H2 >= 0
    # => x_H2 >= max(0, -(a1-c1*risk)*x_AI2 / b1)
    coef2 = net_coef_AI[1]
    min_xH2_formula = max(0, -coef2 * BUDGET / B1[1])

    df = pd.DataFrame({
        'Ngành': SECTORS,
        'x_AI (tỷ VND)': np.round(x_AI, 1),
        'x_H (tỷ VND)':  np.round(x_H, 1),
        'NewJob (nghìn)': np.round(NewJob, 1),
        'UpgradeJob (nghìn)': np.round(UpgradeJob, 1),
        'DisplacedJob (nghìn)': np.round(DisplacedJob, 1),
        'NetJob ròng (nghìn)': np.round(NetJob, 1),
        'Lao động hiện tại (triệu)': LABOR,
    })

    img_bar = _plot_netjob(df)
    img_sankey = _plot_sankey(df)

    return {
        'optimal': df,
        'total_netjob': round(NetJob.sum(), 0),
        'total_budget_used': round((x_AI + x_H).sum(), 0),
        'threshold_xH_sector2': round(min_xH2_formula, 1),
        'img_bar': img_bar,
        'img_sankey': img_sankey,
    }


def _plot_netjob(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar: phân bổ ngân sách
    ax = axes[0]
    x = np.arange(N)
    ax.bar(x, df['x_AI (tỷ VND)'], label='x_AI', color='#e74c3c')
    ax.bar(x, df['x_H (tỷ VND)'], bottom=df['x_AI (tỷ VND)'], label='x_H', color='#3498db')
    ax.set_xticks(x); ax.set_xticklabels(SECTORS, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Tỷ VND')
    ax.set_title('Phân bổ ngân sách AI vs Nhân lực theo ngành')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # NetJob bar
    ax2 = axes[1]
    colors = ['green' if v >= 0 else 'red' for v in df['NetJob ròng (nghìn)']]
    ax2.bar(x, df['NetJob ròng (nghìn)'], color=colors)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels(SECTORS, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Nghìn việc làm ròng')
    ax2.set_title('NetJob ròng theo ngành')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


def _plot_sankey(df: pd.DataFrame):
    """Đơn giản hóa: waterfall chart thay Sankey"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N)
    ax.bar(x - 0.25, df['UpgradeJob (nghìn)'],  width=0.25, label='Việc làm nâng cấp', color='#2ecc71')
    ax.bar(x,        df['NewJob (nghìn)'],        width=0.25, label='Việc làm mới (AI)', color='#3498db')
    ax.bar(x + 0.25,-df['DisplacedJob (nghìn)'], width=0.25, label='Việc làm mất (tự động hóa)', color='#e74c3c')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(SECTORS, rotation=45, ha='right', fontsize=8)
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
        print(f"Ngưỡng x_H tối thiểu ngành 2 (CN chế biến): {res['threshold_xH_sector2']:,.1f} tỷ VND")
        print(res['optimal'][['Ngành','x_AI (tỷ VND)','x_H (tỷ VND)','NetJob ròng (nghìn)']].to_string(index=False))
