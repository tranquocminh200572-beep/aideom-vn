"""
Module M4 - Mô phỏng lao động AI & NetJob
AIDEOM-VN Bài 12
FIX v4: Đặt tran riêng cho x_AI và x_H (không tran chung theo ngành)
        + sàn tuyệt đối đủ lớn => mọi ngành đều hiển thị rõ.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

SECTORS = [
    "Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng",
    "Bán buôn-bán lẻ",   "Tài chính-Ngân hàng",
    "Logistics-Vận tải",  "CNTT-Truyền thông",  "Giáo dục-Đào tạo",
]
N = 8

LABOR    = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
RISK_PCT = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
A1 = np.array([ 8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
B1 = np.array([45,   28,   35,   32,   22,   30,   20,   55  ])
C1 = np.array([ 5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1 = np.array([50,   32,   42,   38,   26,   36,   24,   62  ])


def optimize_netjob(budget: float = 30000) -> dict:
    """
    Tối đa hóa tổng NetJob ròng.
    NetJob_i = (a1_i - c1_i*risk_i)*xAI_i + b1_i*xH_i

    Ràng buộc:
      C1 - Tổng ngân sách <= budget
      C2 - Displaced_i <= RetrainCap_i  (an toàn xã hội)
      C3 - x_AI_i <= 20% budget (tránh CNTT chiếm toàn bộ AI)
      C4 - x_H_i  <= 15% budget (tránh GD chiếm toàn bộ H)
      C5 - Sàn tuyệt đối >= 500 tỷ cho mỗi biến (mọi ngành đều hiển thị)
    """
    net_ai = A1 - C1 * RISK_PCT

    # Tran riêng cho từng loại đầu tư
    cap_ai = 0.20 * budget   # tối đa 20% ngân sách cho x_AI mỗi ngành
    cap_h  = 0.15 * budget   # tối đa 15% ngân sách cho x_H  mỗi ngành
    floor  = max(500, budget * 0.015)  # sàn tối thiểu ~1.5% ngân sách

    c = np.concatenate([-net_ai, -B1])
    A_ub, b_ub = [], []

    # C1: tổng ngân sách
    A_ub.append(np.ones(2 * N))
    b_ub.append(budget)

    # C2: Displaced_i <= RetrainCap_i
    for i in range(N):
        row = np.zeros(2 * N)
        row[i]     =  C1[i] * RISK_PCT[i]
        row[N + i] = -D1[i]
        A_ub.append(row)
        b_ub.append(0)

    # C3 & C4: tran riêng x_AI và x_H (bounds)
    bounds = ([(floor, cap_ai) for _ in range(N)] +
              [(floor, cap_h)  for _ in range(N)])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.status != 0:
        return {"status": "infeasible"}

    x_AI = res.x[:N]
    x_H  = res.x[N:]

    new_job   = A1 * x_AI
    upgrade   = B1 * x_H
    displaced = C1 * RISK_PCT * x_AI
    netjob    = new_job + upgrade - displaced

    df = pd.DataFrame({
        "Ngành":               SECTORS,
        "LĐ hiện tại (triệu)": LABOR,
        "Rủi ro TĐH (%)":      (RISK_PCT * 100).round(0).astype(int),
        "x_AI (tỷ)":           x_AI.round(0).astype(int),
        "x_H (tỷ)":            x_H.round(0).astype(int),
        "Việc làm mới":        new_job.round(0).astype(int),
        "Nâng cấp":            upgrade.round(0).astype(int),
        "Dịch chuyển":         displaced.round(0).astype(int),
        "NetJob":              netjob.round(0).astype(int),
    })

    return {
        "status":            "optimal",
        "total_netjob":      int(netjob.sum()),
        "total_budget_used": round((x_AI + x_H).sum(), 0),
        "detail":            df,
    }


def scenario_compare_labor(scenarios: dict) -> pd.DataFrame:
    rows = []
    for name, budget in scenarios.items():
        r = optimize_netjob(budget)
        rows.append({
            "Kịch bản":       name,
            "Ngân sách (tỷ)": budget,
            "NetJob tổng":    r.get("total_netjob", 0),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = optimize_netjob(30000)
    print(f"Tổng NetJob: {r['total_netjob']:,}")
    print(r["detail"][["Ngành", "x_AI (tỷ)", "x_H (tỷ)", "NetJob"]])
