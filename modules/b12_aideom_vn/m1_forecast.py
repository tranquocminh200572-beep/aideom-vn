"""
Module M1 - Dự báo kinh tế (Cobb-Douglas mở rộng)
AIDEOM-VN Bài 12
"""
import numpy as np
import pandas as pd

# ── Tham số mô hình ──────────────────────────────────────────────
ALPHA, BETA, GAMMA, DELTA, THETA = 0.33, 0.42, 0.10, 0.08, 0.07

DATA_MACRO = {
    "year": [2020, 2021, 2022, 2023, 2024, 2025],
    "Y":    [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
    "K":    [16500, 17800, 19600, 21300, 23500, 25900],
    "L":    [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
    "D":    [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
    "AI":   [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
    "H":    [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
}

def estimate_tfp(df=None):
    """Ước lượng TFP theo năm từ dữ liệu lịch sử."""
    if df is None:
        df = pd.DataFrame(DATA_MACRO)
    K = df["K"].values
    L = df["L"].values
    D = df["D"].values
    AI = df["AI"].values
    H = df["H"].values
    Y = df["Y"].values
    A = Y / (K**ALPHA * L**BETA * D**GAMMA * AI**DELTA * H**THETA)
    return df["year"].values, A


def forecast_gdp(scenario: dict) -> dict:
    """
    Dự báo GDP 2026-2030 theo kịch bản.
    
    scenario keys:
        D_2030, AI_2030, H_2030  - giá trị mục tiêu 2030
        K_growth, L_growth       - tăng trưởng hàng năm (%)
        tfp_growth               - tăng trưởng TFP/năm (%)
    """
    _, A_hist = estimate_tfp()
    A0 = float(np.mean(A_hist))          # TFP cơ sở

    years = list(range(2026, 2031))
    n = len(years)

    K0, L0 = 25900 * 1.06, 53.4         # 2026 baseline
    D0, AI0, H0 = 19.5, 80.1, 29.2

    kg = 1 + scenario.get("K_growth", 6) / 100
    lg = 1 + scenario.get("L_growth", 1) / 100
    tg = 1 + scenario.get("tfp_growth", 1.2) / 100

    # Nội suy tuyến tính D, AI, H về mục tiêu 2030
    D_end  = scenario.get("D_2030", 30.0)
    AI_end = scenario.get("AI_2030", 100.0)
    H_end  = scenario.get("H_2030", 35.0)

    results = []
    for i, yr in enumerate(years):
        t = i + 1
        frac = t / n
        K  = K0 * kg**t
        L  = L0 * lg**t
        D  = D0 + (D_end - D0) * frac
        AI = AI0 + (AI_end - AI0) * frac
        H  = H0 + (H_end - H0) * frac
        A  = A0 * tg**t
        Y  = A * K**ALPHA * L**BETA * D**GAMMA * AI**DELTA * H**THETA
        results.append({
            "year": yr, "GDP_nghintỷ": round(Y, 1),
            "K": round(K, 0), "L": round(L, 2),
            "D": round(D, 1), "AI": round(AI, 1),
            "H": round(H, 1), "TFP": round(A, 4),
        })

    return {
        "forecast": pd.DataFrame(results),
        "tfp_hist": dict(zip(DATA_MACRO["year"], A_hist.round(4))),
        "A0": round(A0, 4),
    }


def growth_decomposition() -> pd.DataFrame:
    """Phân rã tăng trưởng GDP 2020-2025 theo từng yếu tố."""
    df = pd.DataFrame(DATA_MACRO)
    rows = []
    for i in range(1, len(df)):
        yr = df["year"].iloc[i]
        dY  = np.log(df["Y"].iloc[i]) - np.log(df["Y"].iloc[i-1])
        dK  = np.log(df["K"].iloc[i]) - np.log(df["K"].iloc[i-1])
        dL  = np.log(df["L"].iloc[i]) - np.log(df["L"].iloc[i-1])
        dD  = np.log(df["D"].iloc[i]) - np.log(df["D"].iloc[i-1])
        dAI = np.log(df["AI"].iloc[i]) - np.log(df["AI"].iloc[i-1])
        dH  = np.log(df["H"].iloc[i]) - np.log(df["H"].iloc[i-1])
        contrib_K  = ALPHA * dK
        contrib_L  = BETA  * dL
        contrib_D  = GAMMA * dD
        contrib_AI = DELTA * dAI
        contrib_H  = THETA * dH
        contrib_TFP = dY - (contrib_K + contrib_L + contrib_D + contrib_AI + contrib_H)
        rows.append({
            "Năm": yr, "TăngTrưởngGDP%": round(dY*100, 2),
            "Vốn(K)%": round(contrib_K*100, 2),
            "LaođộngL%": round(contrib_L*100, 2),
            "SốhóaD%": round(contrib_D*100, 2),
            "AI%": round(contrib_AI*100, 2),
            "NhânlựcH%": round(contrib_H*100, 2),
            "TFP%": round(contrib_TFP*100, 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = forecast_gdp({"D_2030": 30, "AI_2030": 100, "H_2030": 35,
                       "K_growth": 6, "L_growth": 1, "tfp_growth": 1.2})
    print(r["forecast"])
    print("\nPhân rã tăng trưởng:")
    print(growth_decomposition())
