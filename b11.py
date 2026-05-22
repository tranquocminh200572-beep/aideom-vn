"""
Bài 11: Học tăng cường (Q-learning tabular) cho chính sách kinh tế thích nghi
MDP: state = (GDP_growth, Digital, AI_capacity, Unemployment) × 3 mức
5 hành động tương ứng 5 chiến lược phân bổ ngân sách
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

# ── Định nghĩa MDP ───────────────────────────────────────────────────────────
N_LEVELS = 3   # low/medium/high cho mỗi chiều
STATE_DIMS = (3, 3, 3, 3)   # GDP_growth, Digital, AI_cap, Unemp_risk
N_ACTIONS = 5

ALLOCATION = np.array([
    [0.70, 0.10, 0.10, 0.10],   # a0: Truyền thống
    [0.40, 0.25, 0.15, 0.20],   # a1: Cân bằng
    [0.25, 0.45, 0.15, 0.15],   # a2: Số hóa nhanh
    [0.20, 0.20, 0.45, 0.15],   # a3: AI dẫn dắt
    [0.30, 0.20, 0.10, 0.40],   # a4: Bao trùm
])   # K, D, AI, H

ANNUAL_BUDGET = 1_000  # nghìn tỷ / năm
T_EPISODE    = 10
W = np.array([0.40, 0.25, 0.20, 0.15])  # trọng số welfare

# Hàm sản xuất Cobb-Douglas
ALPHA_CD, BETA_CD, GAMMA_CD, DELTA_CD, THETA_CD = 0.33, 0.42, 0.10, 0.08, 0.07
L_FIXED = 54.0


class VietnamEconEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Trạng thái VN 2026: GDP medium, D medium, AI low, U medium
        self.gdp_idx   = 1  # medium
        self.d_idx     = 1
        self.ai_idx    = 0  # low
        self.u_idx     = 1  # medium
        self.K  = 27_500.0; self.D_val  = 20.3
        self.AI = 86.0;      self.H_val  = 30.0
        self.A  = 1.35
        self.t  = 0
        return self._state()

    def _state(self):
        return (self.gdp_idx, self.d_idx, self.ai_idx, self.u_idx)

    def _discretize(self, val, lo, hi):
        if val < lo: return 0
        if val > hi: return 2
        return 1

    def step(self, action):
        a = ALLOCATION[action]
        IK  = a[0] * ANNUAL_BUDGET
        ID  = a[1] * ANNUAL_BUDGET
        IAI = a[2] * ANNUAL_BUDGET
        IH  = a[3] * ANNUAL_BUDGET

        Y_prev = self.A * (self.K**ALPHA_CD) * (L_FIXED**BETA_CD) * \
                 (self.D_val**GAMMA_CD) * (self.AI**DELTA_CD) * (self.H_val**THETA_CD)

        # Cập nhật trạng thái vốn
        self.K    = 0.95 * self.K + IK
        self.D_val = 0.88 * self.D_val + ID/100
        self.AI   = 0.85 * self.AI + IAI/20
        self.H_val = self.H_val + 0.8 * IH/200 - 0.02 * self.H_val
        self.A    = self.A * (1 + 0.003*self.D_val + 0.002*self.AI + 0.004*self.H_val)

        Y_new = self.A * (self.K**ALPHA_CD) * (L_FIXED**BETA_CD) * \
                (self.D_val**GAMMA_CD) * (self.AI**DELTA_CD) * (self.H_val**THETA_CD)

        gdp_growth_pct = (Y_new - Y_prev) / (Y_prev + 1e-6) * 100

        # Rủi ro thất nghiệp tăng khi AI tăng nhanh, giảm khi H tăng
        unemp_risk = max(0, 30 + 0.5*(IAI - IH))  # simplified proxy

        # Rủi ro an ninh mạng ~ tỷ lệ AI
        cyber_risk = 0.3 * IAI / ANNUAL_BUDGET

        # Phát thải ~ IK + IAI
        emission = 0.5 * (IK + IAI) / ANNUAL_BUDGET

        # Reward
        reward = (W[0] * gdp_growth_pct / 10
                - W[1] * unemp_risk / 50
                - W[2] * cyber_risk
                - W[3] * emission)

        # Cập nhật chỉ số rời rạc
        self.gdp_idx = self._discretize(gdp_growth_pct, 5, 9)
        self.d_idx   = self._discretize(self.D_val,     15, 25)
        self.ai_idx  = self._discretize(self.AI,         60, 100)
        self.u_idx   = self._discretize(unemp_risk,      15, 35)

        self.t += 1
        done = self.t >= T_EPISODE
        return self._state(), reward, done


def train_qlearning(n_episodes=10_000, alpha=0.1, gamma=0.95, seed=42):
    np.random.seed(seed)
    Q = np.zeros((*STATE_DIMS, N_ACTIONS))
    env = VietnamEconEnv()
    rewards_per_ep = []

    for ep in range(n_episodes):
        s = env.reset()
        ep_reward = 0
        eps = max(0.05, 1.0 - ep / (n_episodes * 0.6))
        while True:
            if np.random.rand() < eps:
                a = np.random.randint(N_ACTIONS)
            else:
                a = int(np.argmax(Q[s]))
            s2, r, done = env.step(a)
            Q[s + (a,)] += alpha * (r + gamma * Q[s2].max() - Q[s + (a,)])
            s = s2
            ep_reward += r
            if done: break
        rewards_per_ep.append(ep_reward)
    return Q, rewards_per_ep


def extract_policy(Q: np.ndarray):
    ACTION_NAMES = ['Truyền thống', 'Cân bằng', 'Số hóa nhanh', 'AI dẫn dắt', 'Bao trùm']
    LEVEL_NAMES  = ['Thấp', 'TB', 'Cao']

    rows = []
    for g in range(3):
        for d in range(3):
            for ai in range(3):
                for u in range(3):
                    best_a = int(np.argmax(Q[g, d, ai, u]))
                    rows.append({
                        'GDP growth': LEVEL_NAMES[g],
                        'Digital':    LEVEL_NAMES[d],
                        'AI cap':     LEVEL_NAMES[ai],
                        'Unemp':      LEVEL_NAMES[u],
                        'Hành động':  ACTION_NAMES[best_a],
                        'Q value':    round(float(Q[g,d,ai,u].max()), 3),
                    })
    return pd.DataFrame(rows)


def compare_policies(Q: np.ndarray, n_eval=50, seed=99):
    """So sánh π* vs a1, a3, random"""
    np.random.seed(seed)
    env = VietnamEconEnv()
    results = {}
    policies = {
        'π* (Q-learning)': lambda s: int(np.argmax(Q[s])),
        'Luôn a1 (Cân bằng)': lambda s: 1,
        'Luôn a3 (AI dẫn dắt)': lambda s: 3,
        'Ngẫu nhiên': lambda s: np.random.randint(N_ACTIONS),
    }
    for name, pol in policies.items():
        totals = []
        for _ in range(n_eval):
            s = env.reset()
            total = 0
            while True:
                a = pol(s)
                s, r, done = env.step(a)
                total += r
                if done: break
            totals.append(total)
        results[name] = np.mean(totals)
    return results


def plot_results(rewards_per_ep, policy_df, comparison, Q=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Learning curve (moving average)
    window = 200
    ma = pd.Series(rewards_per_ep).rolling(window).mean()
    axes[0].plot(rewards_per_ep, alpha=0.2, color='steelblue')
    axes[0].plot(ma, color='red', lw=2, label=f'MA-{window}')
    axes[0].set_title('Learning Curve Q-learning'); axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward tích lũy'); axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Policy comparison bar
    names = list(comparison.keys())
    vals  = list(comparison.values())
    colors = ['green' if i==0 else 'steelblue' for i in range(len(names))]
    axes[1].barh(names, vals, color=colors)
    axes[1].set_title('So sánh chính sách: Reward trung bình')
    axes[1].set_xlabel('Reward trung bình / episode')
    axes[1].grid(axis='x', alpha=0.3)

    # Heatmap
    ACTION_SHORT = ['Truyền thống','Cân bằng','Số hóa','AI','Bao trùm']
    if Q is not None:
        data = np.zeros((3,3))
        for g in range(3):
            for ai in range(3):
                data[g, ai] = int(np.argmax(Q[g,1,ai,1]))
        axes[2].imshow(data, cmap='tab10', vmin=0, vmax=4, aspect='auto')
        axes[2].set_xticks([0,1,2]); axes[2].set_xticklabels(['AI thấp','AI TB','AI cao'])
        axes[2].set_yticks([0,1,2]); axes[2].set_yticklabels(['GDP thấp','GDP TB','GDP cao'])
        axes[2].set_title('Policy π*(g,ai) khi D=TB, U=TB')
        for g in range(3):
            for ai in range(3):
                axes[2].text(ai, g, ACTION_SHORT[int(data[g,ai])], ha='center', va='center',
                            fontsize=7, color='white')
    else:
        axes[2].text(0.5, 0.5, 'Q matrix not available', ha='center', va='center',
                    transform=axes[2].transAxes)
        axes[2].set_title('Policy Heatmap')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()


def run_b11():
    Q, rewards = train_qlearning(n_episodes=8_000, seed=42)
    policy_df  = extract_policy(Q)
    comparison = compare_policies(Q, n_eval=30)
    img        = plot_results(rewards, policy_df, comparison, Q=Q)

    # Trạng thái VN 2026
    vn_state = (1, 1, 0, 1)
    ACTION_NAMES = ['Truyền thống', 'Cân bằng', 'Số hóa nhanh', 'AI dẫn dắt', 'Bao trùm']
    best_action_vn = int(np.argmax(Q[vn_state]))

    # Các trạng thái đặc biệt
    special_states = {
        'GDP thấp, D thấp, AI thấp, U cao': (0,0,0,2),
        'GDP cao, AI cao, U thấp':           (2,1,2,0),
        'GDP TB, D cao, AI TB, U TB':        (1,2,1,1),
        'GDP thấp, D TB, AI cao, U cao':     (0,1,2,2),
    }
    special_df = pd.DataFrame([
        {'Trạng thái': k, 'Hành động tối ưu': ACTION_NAMES[int(np.argmax(Q[v]))],
         'Q max': round(float(Q[v].max()), 3)}
        for k, v in special_states.items()
    ])

    return {
        'Q':               Q,
        'rewards':         rewards,
        'policy_df':       policy_df,
        'comparison':      comparison,
        'img':             img,
        'vn2026_action':   ACTION_NAMES[best_action_vn],
        'special_df':      special_df,
        'final_reward_ma': float(pd.Series(rewards[-500:]).mean()),
    }


if __name__ == '__main__':
    print("Đang huấn luyện Q-learning (8000 episodes)...")
    res = run_b11()
    print(f"\nHành động tối ưu cho VN 2026: {res['vn2026_action']}")
    print(f"Reward trung bình 500 episode cuối: {res['final_reward_ma']:.4f}")
    print("\nSo sánh chính sách:")
    for k, v in res['comparison'].items():
        print(f"  {k}: {v:.4f}")
    print("\nCác trạng thái đặc biệt:")
    print(res['special_df'].to_string(index=False))
