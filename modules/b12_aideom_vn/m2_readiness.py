"""
Module M2 - Đánh giá sẵn sàng số (TOPSIS + Entropy)
AIDEOM-VN Bài 12
"""
import numpy as np
import pandas as pd

REGIONS = [
    "Trung du MN phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DHMT",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng SCL",
]

# Ma trận dữ liệu 6 vùng (từ vietnam_regions_2024.xlsx)
DATA_REGIONS = {
    "GRDP/người(tr.VND)": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
    "FDI(tỷUSD)":         [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
    "DigitalIndex":       [38, 78, 55, 32, 82, 48],
    "AIReadiness":        [22, 68, 40, 18, 75, 30],
    "LĐĐàoTạo(%)":        [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
    "R&D/GRDP(%)":        [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
    "Internet(%)":        [72, 92, 84, 68, 94, 78],
    "Gini":               [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
}

IS_BENEFIT = [True, True, True, True, True, True, True, False]  # Gini là cost

EXPERT_WEIGHTS = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])


def entropy_weights(X: np.ndarray) -> np.ndarray:
    """Tính trọng số khách quan bằng phương pháp Entropy."""
    X_pos = np.abs(X) + 1e-12
    P = X_pos / X_pos.sum(axis=0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()


def run_topsis(weights: np.ndarray = None) -> pd.DataFrame:
    """Chạy TOPSIS, trả về DataFrame có cột TOPSIS_score và rank."""
    df = pd.DataFrame(DATA_REGIONS, index=REGIONS)
    X = df.values.astype(float)

    if weights is None:
        weights = EXPERT_WEIGHTS

    # Bước 1: chuẩn hóa vector
    R = X / np.sqrt((X**2).sum(axis=0))

    # Bước 2: ma trận có trọng số
    V = R * weights

    # Bước 3: ideal+ và ideal-
    A_star = np.where(IS_BENEFIT, V.max(axis=0), V.min(axis=0))
    A_neg  = np.where(IS_BENEFIT, V.min(axis=0), V.max(axis=0))

    # Bước 4: khoảng cách
    S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
    S_neg  = np.sqrt(((V - A_neg )**2).sum(axis=1))

    # Bước 5: hệ số gần gũi
    C_star = S_neg / (S_star + S_neg)

    result = df.copy()
    result["TOPSIS_score"] = C_star.round(4)
    result["Rank"] = result["TOPSIS_score"].rank(ascending=False).astype(int)
    return result.sort_values("TOPSIS_score", ascending=False)


def run_entropy_topsis() -> pd.DataFrame:
    """TOPSIS với trọng số Entropy."""
    df = pd.DataFrame(DATA_REGIONS, index=REGIONS)
    X = df.values.astype(float)
    w_ent = entropy_weights(X)
    return run_topsis(w_ent), w_ent


def digital_readiness_summary() -> dict:
    """Tóm tắt sẵn sàng số theo trọng số chuyên gia và entropy."""
    expert_result = run_topsis(EXPERT_WEIGHTS)
    entropy_result, w_ent = run_entropy_topsis()
    return {
        "expert": expert_result,
        "entropy": entropy_result,
        "entropy_weights": dict(zip(DATA_REGIONS.keys(), w_ent.round(4))),
        "top3_expert": expert_result.index[:3].tolist(),
        "top3_entropy": entropy_result.index[:3].tolist(),
    }


if __name__ == "__main__":
    res = digital_readiness_summary()
    print("=== TOP 3 (Chuyên gia) ===")
    print(res["expert"][["TOPSIS_score","Rank"]])
    print("\n=== TOP 3 (Entropy) ===")
    print(res["entropy"][["TOPSIS_score","Rank"]])
