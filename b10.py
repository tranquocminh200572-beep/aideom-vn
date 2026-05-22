"""
Bài 10: Quy hoạch ngẫu nhiên hai giai đoạn (Two-Stage Stochastic Programming)
Giải bằng scipy/pulp (thay vì Pyomo) để đơn giản hóa dependency.
Tính VSS và EVPI.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

ITEMS = ['I', 'D', 'AI', 'H']
SCENARIOS = ['s1', 's2', 's3', 's4']

PROB = {'s1': 0.30, 's2': 0.45, 's3': 0.20, 's4': 0.05}
BETA_BASE = {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95}

BETA_S = {
    ('s1','I'):1.25, ('s1','D'):1.35, ('s1','AI'):1.55, ('s1','H'):1.05,
    ('s2','I'):1.00, ('s2','D'):1.10, ('s2','AI'):1.25, ('s2','H'):0.95,
    ('s3','I'):0.75, ('s3','D'):0.85, ('s3','AI'):0.90, ('s3','H'):1.00,
    ('s4','I'):0.40, ('s4','D'):0.50, ('s4','AI'):0.55, ('s4','H'):1.10,
}

BUDGET1 = 65_000
BUDGET2 = 15_000
TOTAL   = BUDGET1 + BUDGET2


def solve_stochastic():
    """Giải bài toán SP dạng đơn giản hóa bằng linprog."""
    from scipy.optimize import linprog

    # Biến: x(4) first-stage + y_s(4) second-stage × 4 kịch bản = 4+16=20
    # Thứ tự: [xI, xD, xAI, xH, y_s1_I, y_s1_D, y_s1_AI, y_s1_H, ..., y_s4_H]
    n_vars = 4 + 4*4

    # Hàm mục tiêu: maximize Σβ_j*x_j + Σ p_s Σ β_s_j * y_s_j
    c = np.zeros(n_vars)
    for j, item in enumerate(ITEMS):
        c[j] = -BETA_BASE[item]
    for s_idx, s in enumerate(SCENARIOS):
        for j, item in enumerate(ITEMS):
            idx = 4 + s_idx*4 + j
            c[idx] = -PROB[s] * BETA_S[(s, item)]

    A_ub, b_ub = [], []

    # C1: Σ x_j <= 65000
    row = np.zeros(n_vars); row[:4] = 1
    A_ub.append(row); b_ub.append(BUDGET1)

    # C2: Σ y_s_j <= 15000 for each s
    for s_idx in range(4):
        row = np.zeros(n_vars)
        row[4+s_idx*4:4+s_idx*4+4] = 1
        A_ub.append(row); b_ub.append(BUDGET2)

    # C3: y_s_AI <= 0.5 * x_H (x_H is index 3)
    for s_idx in range(4):
        row = np.zeros(n_vars)
        row[3] = -0.5          # -0.5 x_H
        row[4 + s_idx*4 + 2] = 1  # y_s_AI
        A_ub.append(row); b_ub.append(0)

    A_ub = np.array(A_ub); b_ub = np.array(b_ub)
    bounds = [(0, None)] * n_vars

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    obj = -res.fun

    x_opt = res.x[:4]
    y_opt = res.x[4:].reshape(4, 4)

    return obj, x_opt, y_opt, res.status == 0


def solve_deterministic_ev():
    """EV: dùng kịch bản trung bình (expected value)"""
    from scipy.optimize import linprog

    # Beta kỳ vọng theo xác suất
    beta_ev = {}
    for j, item in enumerate(ITEMS):
        beta_ev[item] = sum(PROB[s]*BETA_S[(s,item)] for s in SCENARIOS)

    # Chỉ có first-stage, tổng ngân sách = 80000
    c = [-beta_ev[item] for item in ITEMS]
    A_ub = [np.ones(4)]; b_ub = [TOTAL]
    bounds = [(0, None)]*4
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return -res.fun, res.x


def solve_perfect_info():
    """WS (Wait-and-See): giải tối ưu từng kịch bản riêng"""
    from scipy.optimize import linprog

    total_ws = 0.0
    for s in SCENARIOS:
        beta_s = [BETA_S[(s, item)] for item in ITEMS]
        c = [-b for b in beta_s]
        A_ub = [np.ones(4)]; b_ub = [TOTAL]
        bounds = [(0, None)]*4
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        total_ws += PROB[s] * (-res.fun)
    return total_ws


def run_b10():
    sp_obj, x_opt, y_opt, converged = solve_stochastic()
    ev_obj, x_ev = solve_deterministic_ev()
    ws_obj = solve_perfect_info()

    VSS  = round(sp_obj - ev_obj, 2)
    EVPI = round(ws_obj - sp_obj, 2)

    x_df = pd.DataFrame({
        'Hạng mục': ITEMS,
        'SP first-stage (tỷ VND)': np.round(x_opt, 1),
        'EV first-stage (tỷ VND)': np.round(x_ev, 1),
    })

    y_df = pd.DataFrame(
        np.round(y_opt, 1),
        index=SCENARIOS,
        columns=ITEMS
    )
    y_df.index.name = 'Kịch bản'

    scenario_obj = {}
    for s_idx, s in enumerate(SCENARIOS):
        obj_s = sum(BETA_BASE[j]*x_opt[i] for i,j in enumerate(ITEMS)) + \
                sum(BETA_S[(s,j)]*y_opt[s_idx,i] for i,j in enumerate(ITEMS))
        scenario_obj[s] = round(obj_s, 1)

    img = _plot_b10(x_opt, x_ev, y_opt)

    return {
        'converged':     converged,
        'SP_obj':        round(sp_obj, 1),
        'EV_obj':        round(ev_obj, 1),
        'WS_obj':        round(ws_obj, 1),
        'VSS':           VSS,
        'EVPI':          EVPI,
        'first_stage_df': x_df,
        'second_stage_df': y_df,
        'scenario_obj':  scenario_obj,
        'img':           img,
    }


def _plot_b10(x_opt, x_ev, y_opt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = np.arange(4)
    w = 0.35
    bars1 = ax.bar(x - w/2, x_opt/1000, w, label='SP (Ngẫu nhiên)', color='#3498db')
    bars2 = ax.bar(x + w/2, x_ev/1000, w, label='EV (Xác định)', color='#e67e22')
    ax.set_xticks(x); ax.set_xticklabels(ITEMS)
    ax.set_ylabel('Nghìn tỷ VND')
    ax.set_title('So sánh phân bổ First-Stage: SP vs EV')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    colors = ['#2ecc71','#3498db','#e74c3c','#9b59b6']
    for s_idx, s in enumerate(SCENARIOS):
        ax2.bar(x + s_idx*0.2 - 0.3, y_opt[s_idx]/1000, 0.2,
                label=f'{s} (p={PROB[s]})', color=colors[s_idx])
    ax2.set_xticks(x); ax2.set_xticklabels(ITEMS)
    ax2.set_ylabel('Nghìn tỷ VND')
    ax2.set_title('Phân bổ Second-Stage theo kịch bản')
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == '__main__':
    res = run_b10()
    print(f"SP Objective (kỳ vọng GDP gain): {res['SP_obj']:,.1f} nghìn tỷ")
    print(f"EV Objective:                    {res['EV_obj']:,.1f} nghìn tỷ")
    print(f"WS Objective (thông tin hoàn hảo): {res['WS_obj']:,.1f} nghìn tỷ")
    print(f"\nVSS  = SP - EV = {res['VSS']:,.2f} nghìn tỷ")
    print(f"EVPI = WS - SP = {res['EVPI']:,.2f} nghìn tỷ")
    print("\nPhân bổ First-Stage:")
    print(res['first_stage_df'].to_string(index=False))
    print("\nPhân bổ Second-Stage theo kịch bản:")
    print(res['second_stage_df'].to_string())
