"""
Bài 8: Tối ưu động phân bổ liên thời gian 2026–2035
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

T       = 10
RHO_D   = 0.97
DELTA_K = 0.05
DELTA_D = 0.12
DELTA_AI= 0.15
THETA_H = 0.80
MU      = 0.02
PHI1, PHI2, PHI3 = 0.003, 0.002, 0.004

ALPHA, BETA_CD, GAMMA_CD, DELTA_CD, THETA_CD = 0.33, 0.42, 0.10, 0.08, 0.07
L_FIXED = 53.9

K0   = 1.0
D0   = 1.0
AI0  = 1.0
H0   = 1.0
A0   = 12_848.0


def production(K, D, AI, H, A):
    return A * (max(K,1e-6) ** ALPHA) * (L_FIXED/53.9 ** BETA_CD) * \
           (max(D,1e-6) ** GAMMA_CD) * (max(AI,1e-6) ** DELTA_CD) * \
           (max(H,1e-6) ** THETA_CD)


def unpack(z):
    sK  = z[0*T:1*T]
    sD  = z[1*T:2*T]
    sAI = z[2*T:3*T]
    sH  = z[3*T:4*T]
    sC  = z[4*T:5*T]
    return sK, sD, sAI, sH, sC


def simulate(z):
    sK, sD, sAI, sH, sC = unpack(z)
    Ks, Ds, AIs, Hs, As, Ys = [K0],[D0],[AI0],[H0],[A0],[]
    for t in range(T):
        K  = Ks[-1]; D  = Ds[-1]; AI = AIs[-1]; H  = Hs[-1]; A = As[-1]
        Y  = production(K, D, AI, H, A)
        Ys.append(Y)
        IK  = max(sK[t],  0) * Y
        ID  = max(sD[t],  0) * Y
        IAI = max(sAI[t], 0) * Y
        IH  = max(sH[t],  0) * Y
        Ks.append((1-DELTA_K)  * K  + IK  / (A0))
        Ds.append((1-DELTA_D)  * D  + ID  / (A0 * 0.1))
        AIs.append((1-DELTA_AI) * AI + IAI / (A0 * 0.08))
        Hs.append(max(0.01, H + THETA_H * IH/(A0*0.07) - MU*H))
        tfp_growth = min(PHI1*D + PHI2*AI + PHI3*H, 0.03)
        As.append(A * (1 + tfp_growth))
    return Ks, Ds, AIs, Hs, As, Ys


def welfare(z):
    _, _, _, _, _, Ys = simulate(z)
    sK, sD, sAI, sH, sC = unpack(z)
    W = 0.0
    for t in range(T):
        Ct = max(sC[t] * Ys[t], 1.0)
        W += (RHO_D ** t) * np.log(Ct)
    return -W


def run_b08(scenario_shock=False):
    sK0  = np.full(T, 0.20)
    sD0  = np.full(T, 0.05)
    sAI0 = np.full(T, 0.03)
    sH0  = np.full(T, 0.07)
    sC0  = np.full(T, 0.60)
    z0   = np.concatenate([sK0, sD0, sAI0, sH0, sC0])

    cons = []
    for t in range(T):
        def make_con(t_):
            return lambda z: 1.0 - (z[t_] + z[T+t_] + z[2*T+t_] + z[3*T+t_] + z[4*T+t_])
        cons.append({'type':'ineq', 'fun': make_con(t)})

    bounds = [(0.01, 0.80)] * T
    bounds += [(0.01, 0.30)] * T
    bounds += [(0.01, 0.25)] * T
    bounds += [(0.01, 0.20)] * T
    bounds += [(0.20, 0.80)] * T

    result = minimize(welfare, z0, method='SLSQP',
                      bounds=bounds, constraints=cons,
                      options={'maxiter': 300, 'ftol': 1e-5, 'disp': False})

    z_opt = result.x
    sK, sD, sAI, sH, sC = unpack(z_opt)
    Ks, Ds, AIs, Hs, As, Ys = simulate(z_opt)

    years = list(range(2026, 2026+T))

    # FIX: Tên cột nhất quán, không dùng ký tự đặc biệt gây KeyError
    traj = pd.DataFrame({
        'Nam': years,
        'GDP_nghin_ty': [round(y/1000, 2) for y in Ys],
        'Tieu_dung_pct': np.round(sC*100, 1),
        'Dau_tu_K_pct': np.round(sK*100, 1),
        'Dau_tu_D_pct': np.round(sD*100, 1),
        'Dau_tu_AI_pct': np.round(sAI*100, 1),
        'Dau_tu_H_pct': np.round(sH*100, 1),
        'TFP_A': np.round(As[:T], 1),
    })

    # Bảng hiển thị đẹp với tên tiếng Việt
    traj_display = pd.DataFrame({
        'Năm': years,
        'GDP (nghìn tỷ VND)': traj['GDP_nghin_ty'],
        'Tiêu dùng C (%Y)': traj['Tieu_dung_pct'],
        'Đầu tư K (%Y)': traj['Dau_tu_K_pct'],
        'Đầu tư D (%Y)': traj['Dau_tu_D_pct'],
        'Đầu tư AI (%Y)': traj['Dau_tu_AI_pct'],
        'Đầu tư H (%Y)': traj['Dau_tu_H_pct'],
        'TFP A': traj['TFP_A'],
    })

    img = _plot_trajectory(traj)
    welfare_val = -welfare(z_opt)

    traj_display = pd.DataFrame({
        'Năm': years,
        'GDP (nghìn tỷ VND)': traj['GDP_nghin_ty'],
        'Tiêu dùng C (%Y)':   traj['Tieu_dung_pct'],
        'Đầu tư K (%Y)':      traj['Dau_tu_K_pct'],
        'Đầu tư D (%Y)':      traj['Dau_tu_D_pct'],
        'Đầu tư AI (%Y)':     traj['Dau_tu_AI_pct'],
        'Đầu tư H (%Y)':      traj['Dau_tu_H_pct'],
        'TFP A':               traj['TFP_A'],
    })

    return {
        'converged':          result.success,
        'welfare':            round(welfare_val, 4),
        'trajectory':         traj,
        'trajectory_display': traj_display,
        'img':                img,
        'GDP_2035':           round(Ys[-1]/1000, 2),
    }


def _plot_trajectory(traj):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.flatten()

    ax[0].plot(traj['Nam'], traj['GDP_nghin_ty'], 'b-o', label='GDP cơ sở')
    ax[0].set_title('Quỹ đạo GDP 2026-2035')
    ax[0].set_ylabel('Nghìn tỷ VND')
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    ax[1].plot(traj['Nam'], traj['Dau_tu_K_pct'],  label='IK (%Y)', marker='o')
    ax[1].plot(traj['Nam'], traj['Dau_tu_D_pct'],  label='ID (%Y)', marker='s')
    ax[1].plot(traj['Nam'], traj['Dau_tu_AI_pct'], label='IAI (%Y)', marker='^')
    ax[1].plot(traj['Nam'], traj['Dau_tu_H_pct'],  label='IH (%Y)', marker='D')
    ax[1].set_title('Phân bổ đầu tư (% GDP)')
    ax[1].set_ylabel('%')
    ax[1].legend(); ax[1].grid(True, alpha=0.3)

    ax[2].plot(traj['Nam'], traj['Tieu_dung_pct'], 'g-o')
    ax[2].set_title('Tiêu dùng C_t (% GDP)')
    ax[2].set_ylabel('%')
    ax[2].grid(True, alpha=0.3)

    ax[3].plot(traj['Nam'], traj['TFP_A'], 'm-s')
    ax[3].set_title('TFP A_t (nội sinh)')
    ax[3].set_ylabel('TFP A')
    ax[3].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == '__main__':
    res = run_b08()
    print(f"Hội tụ: {res['converged']}")
    print(f"Welfare: {res['welfare']:.4f}")
    print(res['trajectory_display'].to_string(index=False))
