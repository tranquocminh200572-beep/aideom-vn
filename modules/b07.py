"""
Bài 7: Tối ưu đa mục tiêu Pareto với NSGA-II
4 mục tiêu: tối đa GDP gain, giảm bất bình đẳng, giảm phát thải, giảm rủi ro an ninh dữ liệu
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

REGIONS = ['TDMNPB', 'ĐBSH', 'BTB+DHMT', 'TN', 'ĐNB', 'ĐBSCL']
ITEMS   = ['I', 'D', 'AI', 'H']

BETA = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
])

E   = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
RHO = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
SIG = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

BUDGET    = 50_000
MIN_R     = 5_000
MAX_R     = 14_000
MIN_H_TOT = 12_000
GAMMA = 0.002
LAM   = 0.7
D0    = np.array([38, 78, 55, 32, 82, 48])


def decode(x):
    return x.reshape(6, 4)


def eval_objectives(x, weights=(0.40, 0.25, 0.20, 0.15)):
    X = decode(x)
    f1 = -(BETA * X).sum()
    sums = X.sum(axis=1)
    f2 = np.abs(sums - sums.mean()).mean()
    f3 = (E * (X[:, 0] + X[:, 2])).sum()
    f4 = (RHO * X[:, 2]).sum() - (SIG * X[:, 3]).sum()
    return np.array([f1, f2, f3, f4])


def eval_constraints(x, budget=None, min_r=None, max_r=None, min_h=None):
    budget = budget or BUDGET
    min_r  = min_r  or MIN_R
    max_r  = max_r  or MAX_R
    min_h  = min_h  or MIN_H_TOT
    X = decode(x)
    g = []
    g.append(X.sum() - budget)
    for r in range(6):
        g.append(min_r - X[r].sum())
    for r in range(6):
        g.append(X[r].sum() - max_r)
    g.append(min_h - X[:, 3].sum())
    D_new = D0 + GAMMA * X[:, 1]
    Dmax  = D_new.max()
    for r in range(6):
        g.append(LAM * Dmax - D_new[r])
    return np.array(g)


def run_nsga2(pop_size=80, n_gen=150, seed=42, budget=None, min_r=None, max_r=None, min_h=None, weights=(0.40,0.25,0.20,0.15)):
    budget = budget or BUDGET
    min_r  = min_r  or MIN_R
    max_r  = max_r  or MAX_R
    min_h  = min_h  or MIN_H_TOT

    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as pymoo_min
        from pymoo.termination import get_termination

        class VNProblem(ElementwiseProblem):
            def __init__(self):
                n_c = 1 + 6 + 6 + 1 + 6
                super().__init__(n_var=24, n_obj=4, n_ieq_constr=n_c,
                                 xl=np.zeros(24), xu=np.ones(24) * max_r)

            def _evaluate(self, x, out, *args, **kwargs):
                out['F'] = eval_objectives(x, weights)
                out['G'] = eval_constraints(x, budget, min_r, max_r, min_h)

        res = pymoo_min(VNProblem(), NSGA2(pop_size=pop_size),
                        get_termination('n_gen', n_gen),
                        seed=seed, verbose=False)
        return res.F, res.X
    except ImportError:
        return _fallback_random(seed, budget, min_r, max_r, min_h)


def _fallback_random(seed=42, budget=None, min_r=None, max_r=None, min_h=None):
    budget = budget or BUDGET
    min_r  = min_r  or MIN_R
    max_r  = max_r  or MAX_R
    min_h  = min_h  or MIN_H_TOT
    rng = np.random.default_rng(seed)
    F_list, X_list = [], []
    attempts = 0
    while len(X_list) < 300 and attempts < 500_000:
        attempts += 1
        X = np.zeros((6, 4))
        for r in range(6):
            total_r = rng.uniform(min_r, max_r)
            shares = rng.dirichlet(np.ones(4))
            X[r] = shares * total_r
        if X.sum() > budget:
            continue
        if X[:, 3].sum() < min_h:
            deficit = min_h - X[:, 3].sum()
            X[:, 3] += deficit / 6
        if X.sum() > budget:
            continue
        x = X.flatten()
        F_list.append(eval_objectives(x))
        X_list.append(x)

    if len(X_list) == 0:
        X = np.zeros((6, 4))
        for r in range(6):
            X[r] = [min_r*0.3, min_r*0.2, min_r*0.1, min_r*0.4]
        F_list.append(eval_objectives(X.flatten()))
        X_list.append(X.flatten())

    return np.array(F_list), np.array(X_list)


def topsis_on_pareto(F, weights=(0.40, 0.25, 0.20, 0.15)):
    w = np.array(weights)
    norms = np.sqrt((F ** 2).sum(axis=0))
    R = F / (norms + 1e-12)
    V = R * w
    A_star = V.min(axis=0)
    A_neg  = V.max(axis=0)
    S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
    S_neg  = np.sqrt(((V - A_neg)  ** 2).sum(axis=1))
    C_star = S_neg / (S_star + S_neg + 1e-12)
    best_idx = np.argmax(C_star)
    return best_idx, C_star


def plot_pareto_3d(F, best_idx):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(-F[:, 0], F[:, 1], F[:, 2],
                    c=F[:, 3], cmap='viridis', s=20, alpha=0.6)
    ax.scatter(-F[best_idx, 0], F[best_idx, 1], F[best_idx, 2],
               c='red', s=150, marker='*', label='Nghiệm thỏa hiệp')
    ax.set_xlabel('GDP gain (tỷ VND)')
    ax.set_ylabel('Bất bình đẳng (MAD)')
    ax.set_zlabel('Phát thải CO₂')
    ax.set_title('Đường biên Pareto 3D — Bài 7')
    plt.colorbar(sc, ax=ax, label='Rủi ro an ninh dữ liệu')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


def plot_parallel_coords(F, best_idx):
    labels = ['GDP gain', 'Bất bình đẳng', 'Phát thải', 'Rủi ro AN']
    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-12)
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = range(4)
    for i, row in enumerate(F_norm):
        color = 'red' if i == best_idx else 'steelblue'
        lw    = 2.5  if i == best_idx else 0.3
        alpha = 1.0  if i == best_idx else 0.2
        ax.plot(xs, row, color=color, lw=lw, alpha=alpha)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Giá trị chuẩn hóa [0,1]')
    ax.set_title('Parallel Coordinates — Tập Pareto (đỏ = nghiệm thỏa hiệp)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


def run_b07(pop_size=80, n_gen=150, seed=42, budget=50000,
            min_r=5000, max_r=14000, min_h=12000,
            w_gdp=0.40, w_equity=0.25, w_env=0.20, w_security=0.15):
    weights = (w_gdp, w_equity, w_env, w_security)
    F, X = run_nsga2(pop_size=pop_size, n_gen=n_gen, seed=seed,
                     budget=budget, min_r=min_r, max_r=max_r, min_h=min_h,
                     weights=weights)
    if F is None or len(F) == 0:
        return {"error": "Không tìm được nghiệm khả thi"}

    best_idx, C_star = topsis_on_pareto(F, weights)
    best_X = decode(X[best_idx])
    best_F = F[best_idx]

    growth_best_idx = np.argmin(F[:, 0])
    growth_F = F[growth_best_idx]

    alloc_df = pd.DataFrame(best_X, index=REGIONS, columns=ITEMS)
    alloc_df['Tổng'] = alloc_df.sum(axis=1)

    trade_off = {
        'GDP tối đa (tỷ VND)':      f"{-growth_F[0]:,.0f}",
        'GDP nghiệm thỏa hiệp':      f"{-best_F[0]:,.0f}",
        'Bất bình đẳng (tối đa GDP)':f"{growth_F[1]:.1f}",
        'Bất bình đẳng (thỏa hiệp)': f"{best_F[1]:.1f}",
        'Phát thải (tối đa GDP)':     f"{growth_F[2]:.1f}",
        'Phát thải (thỏa hiệp)':      f"{best_F[2]:.1f}",
    }

    img_3d = plot_pareto_3d(F, best_idx)
    img_pc = plot_parallel_coords(F, best_idx)

    return {
        "pareto_size":  len(F),
        "best_idx":     int(best_idx),
        "best_F":       best_F.tolist(),
        "best_alloc":   alloc_df,
        "trade_off":    trade_off,
        "img_3d":       img_3d,
        "img_parallel": img_pc,
    }


if __name__ == '__main__':
    result = run_b07()
    if 'error' not in result:
        print(f"Tập Pareto: {result['pareto_size']} nghiệm")
        print(f"GDP gain: {-result['best_F'][0]:,.0f} tỷ VND")
        print(result['best_alloc'].to_string())
