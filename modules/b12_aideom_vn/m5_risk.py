"""
Module M5 - Đánh giá rủi ro (Đa mục tiêu + Stochastic LP)
AIDEOM-VN Bài 12
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ── Tham số vùng (Bài 7) ───────────────────────────────────────
NR = 6
BETA_MATRIX = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
])
E_EMISSION = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
RHO_RISK   = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
SIGMA_RISK = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

# ── Kịch bản ngẫu nhiên (Bài 10) ──────────────────────────────
SCENARIOS = {
    "s1_optimistic":   {"p": 0.30, "mult": [1.25, 1.35, 1.55, 1.05]},
    "s2_baseline":     {"p": 0.45, "mult": [1.00, 1.10, 1.25, 0.95]},
    "s3_pessimistic":  {"p": 0.20, "mult": [0.75, 0.85, 0.90, 1.00]},
    "s4_crisis":       {"p": 0.05, "mult": [0.40, 0.50, 0.55, 1.10]},
}

SCENARIO_LABELS = {
    "s1_optimistic": "Lạc quan",
    "s2_baseline": "Cơ sở",
    "s3_pessimistic": "Bi quan",
    "s4_crisis": "Khủng hoảng",
}


def multi_objective_tradeoff(budget: float = 50000) -> dict:
    """
    Tính frontier Pareto đơn giản: tối đa f1 GDP tại các mức ràng buộc
    về phát thải f3 khác nhau. Trả về điểm tradeoff.
    """
    points = []
    emission_caps = np.linspace(1500, 5000, 15)

    for cap in emission_caps:
        # Biến: x flat 24 (NR x NJ: I,D,AI,H)
        n = NR * 4
        c = -BETA_MATRIX.flatten()  # minimize -GDP gain

        A_ub, b_ub = [], []

        # Budget
        A_ub.append(np.ones(n)); b_ub.append(budget)

        # Sàn vùng
        for r in range(NR):
            row = np.zeros(n); row[r*4:(r+1)*4] = -1
            A_ub.append(row); b_ub.append(-5000)

        # Trần vùng
        for r in range(NR):
            row = np.zeros(n); row[r*4:(r+1)*4] = 1
            A_ub.append(row); b_ub.append(12000)

        # Phát thải cap: sum_r e_r*(x_I,r + x_AI,r) <= cap
        row_em = np.zeros(n)
        for r in range(NR):
            row_em[r*4]   = E_EMISSION[r]   # I
            row_em[r*4+2] = E_EMISSION[r]   # AI
        A_ub.append(row_em); b_ub.append(cap)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(0, None)]*n, method="highs")
        if res.status == 0:
            x = res.x.reshape(NR, 4)
            emission = float((E_EMISSION * (x[:,0] + x[:,2])).sum())
            # Rủi ro ròng
            risk_net = float((RHO_RISK * x[:,2]).sum() - (SIGMA_RISK * x[:,3]).sum())
            points.append({
                "emission_cap": round(cap, 0),
                "GDP_gain": round(-res.fun, 1),
                "actual_emission": round(emission, 1),
                "risk_net": round(risk_net, 2),
            })

    return {"frontier": pd.DataFrame(points)}


def stochastic_lp(budget1: float = 65000,
                   budget2: float = 15000) -> dict:
    """
    Quy hoạch ngẫu nhiên hai giai đoạn đơn giản hóa.
    Stage 1: x (4 biến); Stage 2: y_s (4 biến cho mỗi kịch bản).
    """
    beta1 = np.array([1.00, 1.10, 1.25, 0.95])   # hệ số stage 1
    n1 = 4
    n2 = 4 * len(SCENARIOS)
    n_total = n1 + n2

    # Objective: -beta1'x - sum_s p_s * beta_s' y_s
    c = np.zeros(n_total)
    c[:n1] = -beta1

    offset = n1
    for s_key, s_val in SCENARIOS.items():
        beta_s = np.array(beta1) * np.array(s_val["mult"])
        c[offset:offset+4] = -s_val["p"] * beta_s
        offset += 4

    A_ub, b_ub = [], []

    # Stage 1 budget
    row = np.zeros(n_total); row[:n1] = 1
    A_ub.append(row); b_ub.append(budget1)

    # Stage 2 budget per scenario
    offset = n1
    for _ in SCENARIOS:
        row = np.zeros(n_total); row[offset:offset+4] = 1
        A_ub.append(row); b_ub.append(budget2)
        offset += 4

    # y_AI <= 0.5 * x_H
    offset = n1
    for _ in SCENARIOS:
        row = np.zeros(n_total)
        row[3] = -0.5          # -0.5 x_H
        row[offset + 2] = 1    # y_AI
        A_ub.append(row); b_ub.append(0)
        offset += 4

    # Stage 1 minimums
    for j in range(n1):
        row = np.zeros(n_total); row[j] = -1
        A_ub.append(row); b_ub.append(-5000)

    bounds = [(0, None)] * n_total
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.status != 0:
        return {"status": "infeasible"}

    x1 = res.x[:n1]
    items = ["Hạ tầng số", "Chuyển đổi số", "AI", "Nhân lực số"]
    stage1_df = pd.DataFrame({"Hạng mục": items,
                               "Phân bổ Stage 1 (tỷ)": x1.round(0)})

    offset = n1
    stage2_rows = []
    for s_key in SCENARIOS:
        ys = res.x[offset:offset+4]
        for j, item in enumerate(items):
            stage2_rows.append({
                "Kịch bản": SCENARIO_LABELS[s_key],
                "Hạng mục": item,
                "Điều chỉnh (tỷ)": round(ys[j], 0),
            })
        offset += 4

    # Tính VSS đơn giản: so sánh với EV solution
    ev_beta = np.array([1.00, 1.10, 1.25, 0.95])  # kịch bản cơ sở
    c_det = np.zeros(n1)
    c_det[:n1] = -ev_beta
    row_b = np.ones(n1)
    A_det = [row_b] + [[-1 if j==k else 0 for k in range(n1)] for j in range(n1)]
    b_det = [budget1] + [-5000]*n1
    res_det = linprog(c_det, A_ub=A_det, b_ub=b_det,
                      bounds=[(0,None)]*n1, method="highs")
    ev_obj = float(-res_det.fun) if res_det.status==0 else 0
    sp_obj = float(-res.fun)
    vss = sp_obj - ev_obj

    return {
        "status": "optimal",
        "sp_objective": round(sp_obj, 1),
        "ev_objective": round(ev_obj, 1),
        "vss": round(vss, 1),
        "stage1": stage1_df,
        "stage2": pd.DataFrame(stage2_rows),
        "x1": x1,
    }


def risk_dashboard() -> dict:
    """Tổng hợp đánh giá rủi ro cho dashboard."""
    frontier = multi_objective_tradeoff()
    stoch = stochastic_lp()
    return {"frontier": frontier, "stochastic": stoch}


if __name__ == "__main__":
    r = stochastic_lp()
    print("SP Objective:", r["sp_objective"])
    print("VSS:", r["vss"])
    print(r["stage1"])
