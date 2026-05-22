"""
Module M3 - Tối ưu phân bổ ngân sách số (LP ngành-vùng)
AIDEOM-VN Bài 12
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = [
    "Trung du MN phía Bắc", "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DHMT", "Tây Nguyên",
    "Đông Nam Bộ", "Đồng bằng SCL",
]
ITEMS  = ["I", "D", "AI", "H"]
ITEM_NAMES = ["Hạ tầng số", "Chuyển đổi số DN", "Năng lực AI", "Nhân lực số"]
NR, NJ = 6, 4

# Hệ số tác động biên β_{j,r}  shape (NR, NJ): [I, D, AI, H]
BETA = np.array([
    [1.15, 0.85, 0.55, 1.30],   # NMM
    [0.95, 1.25, 1.40, 1.05],   # RRD
    [1.05, 0.95, 0.85, 1.15],   # NCC
    [1.20, 0.75, 0.45, 1.35],   # CH
    [0.90, 1.30, 1.55, 1.00],   # SE
    [1.10, 0.85, 0.65, 1.25],   # MD
])

D0 = np.array([38, 78, 55, 32, 82, 48])   # Chỉ số số hóa ban đầu
GAMMA_EQ, LAMBDA_EQ = 0.002, 0.55        # lambda nới để bài toán khả thi


def _build_lp(budget: float = 50000, fairness: bool = True):
    """
    Xây dựng bài toán LP scipy.
    Biến: x flat (NR*NJ,) rồi M (1 biến phụ cho max Dmax).
    Tổng: NR*NJ + 1 = 25 biến.
    """
    n = NR * NJ  # 24
    # Hàm mục tiêu (minimize -Z)
    c = np.zeros(n + 1)
    c[:n] = -BETA.flatten()          # tối đa hóa GDP gain

    A_ub, b_ub = [], []

    # C1: tổng ngân sách
    row = np.zeros(n + 1); row[:n] = 1; A_ub.append(row); b_ub.append(budget)

    # C2: sàn mỗi vùng (>= 5000) → -sum >= -5000
    for r in range(NR):
        row = np.zeros(n + 1)
        row[r*NJ:(r+1)*NJ] = -1
        A_ub.append(row); b_ub.append(-5000)

    # C3: trần mỗi vùng (<= 12000)
    for r in range(NR):
        row = np.zeros(n + 1)
        row[r*NJ:(r+1)*NJ] = 1
        A_ub.append(row); b_ub.append(12000)

    # C4: sàn nhân lực số >= 0.24*budget
    row = np.zeros(n + 1)
    for r in range(NR):
        row[r*NJ + 3] = -1   # H = index 3
    A_ub.append(row); b_ub.append(-0.24 * budget)

    if fairness:
        # C5: D_r + gamma*x_{D,r} <= M  (M là biến số 24)
        for r in range(NR):
            row = np.zeros(n + 1)
            row[r*NJ + 1] = GAMMA_EQ    # x_{D,r}
            row[n] = -1                  # -M
            A_ub.append(row); b_ub.append(-D0[r])

        # C5b: D_r + gamma*x_{D,r} >= lambda*M
        for r in range(NR):
            row = np.zeros(n + 1)
            row[r*NJ + 1] = -GAMMA_EQ
            row[n] = LAMBDA_EQ
            A_ub.append(row); b_ub.append(D0[r])

    bounds = [(0, None)] * n + [(0, None)]   # M >= 0
    return c, A_ub, b_ub, bounds


def optimize_allocation(budget: float = 50000,
                         fairness: bool = True) -> dict:
    """Giải LP phân bổ ngân sách. Trả về dict kết quả."""
    c, A_ub, b_ub, bounds = _build_lp(budget, fairness)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.status != 0:
        return {"status": "infeasible", "message": res.message}

    x_opt = res.x[:NR*NJ].reshape(NR, NJ)
    gdp_gain = float(-res.fun)

    df_alloc = pd.DataFrame(
        x_opt.round(1),
        index=REGION_NAMES,
        columns=ITEM_NAMES,
    )
    df_alloc["Tổng"] = df_alloc.sum(axis=1).round(1)

    return {
        "status": "optimal",
        "gdp_gain_tỷVND": round(gdp_gain, 1),
        "allocation": df_alloc,
        "budget": budget,
        "fairness": fairness,
    }


def sensitivity_budget(budgets=None) -> pd.DataFrame:
    """Phân tích độ nhạy theo ngân sách."""
    if budgets is None:
        budgets = [40000, 50000, 60000, 70000]
    rows = []
    for b in budgets:
        r = optimize_allocation(b, fairness=True)
        rows.append({"Ngân sách(tỷ)": b,
                     "GDP Gain(tỷ)": r.get("gdp_gain_tỷVND", 0)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = optimize_allocation(50000, fairness=True)
    print("GDP Gain:", res["gdp_gain_tỷVND"], "tỷ VND")
    print(res["allocation"])
