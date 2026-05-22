"""
Bài 11 Nâng Cấp — Q-learning Tabular vs Deep Q-Network (DQN)
Kỹ thuật nâng cao:
  1. Môi trường Gymnasium đầy đủ với hàm sản xuất Cobb-Douglas
  2. Q-learning tabular với epsilon-greedy decay
  3. DQN (stable-baselines3) cùng môi trường
  4. So sánh learning curve 4 agents: Q-table, DQN, rule-based (a1,a3)
  5. Policy heatmap: π*(s) theo 81 trạng thái
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings('ignore')

# ── MÔI TRƯỜNG ────────────────────────────────────────────────────────────────
class VietnamEconEnv(gym.Env):
    """
    State: (gdp_growth, digital_index, ai_capacity, unemployment_risk)
    Mỗi yếu tố: 0=low, 1=medium, 2=high  →  3^4=81 states
    Action: 5 chiến lược phân bổ ngân sách
    """
    metadata = {'render_modes': []}

    ALLOC = {
        0: np.array([0.70, 0.10, 0.10, 0.10]),  # Truyền thống
        1: np.array([0.40, 0.25, 0.15, 0.20]),  # Cân bằng
        2: np.array([0.25, 0.45, 0.15, 0.15]),  # Số hóa nhanh
        3: np.array([0.20, 0.20, 0.45, 0.15]),  # AI dẫn dắt
        4: np.array([0.30, 0.20, 0.10, 0.40]),  # Bao trùm
    }
    W = np.array([0.40, 0.25, 0.20, 0.15])      # (ΔGDP, -Δunemp, -CyberRisk, -Emission)

    def __init__(self, seed=None):
        super().__init__()
        self.action_space      = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.T = 10
        self._seed = seed

    def _state_to_idx(self, s):
        return s[0]*27 + s[1]*9 + s[2]*3 + s[3]

    def _update_economy(self, action):
        a = self.ALLOC[int(action)]
        budget = 1000
        self.K   += a[0]*budget
        self.D   += a[1]*budget / 200
        self.AI  += a[2]*budget / 30
        self.H   += a[3]*budget / 300

        alpha, beta_l, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
        L = 54.0 + self.t*0.2
        Y_new = self.A0 * self.K**alpha * L**beta_l * self.D**gamma * self.AI**delta * self.H**theta
        gdp_growth = (Y_new - self.Y_prev) / self.Y_prev if self.Y_prev > 0 else 0

        # Cyber risk tăng với AI, giảm với H
        cyber_risk = 0.3 * a[2] - 0.15 * a[3]
        emission   = 0.5 * (a[0] + a[2])
        unemp_delta= -0.2 * a[2] + 0.1 * a[3]   # AI tăng thất nghiệp, H giảm

        reward = (self.W[0]*gdp_growth*100
                 - self.W[1]*max(0, unemp_delta)
                 - self.W[2]*max(0, cyber_risk)
                 - self.W[3]*emission)

        # Cập nhật state rời rạc
        def disc(v, lo, hi): return min(2, max(0, int((v-lo)/(hi-lo)*3)))
        new_state = np.array([
            disc(gdp_growth*100, -2, 12),
            disc(self.D,         12, 35),
            disc(self.AI,        55, 110),
            disc(max(0, unemp_delta*10), 0, 2)
        ])
        self.Y_prev = Y_new
        return new_state, reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t    = 0
        self.K    = 27500.0
        self.D    = 20.3
        self.AI   = 86.0
        self.H    = 30.0
        self.A0   = 0.09
        self.Y_prev = self.A0 * self.K**0.33 * 54.0**0.42 * self.D**0.10 * self.AI**0.08 * self.H**0.07
        self.state = np.array([1, 1, 0, 1])
        return self.state, {}

    def step(self, action):
        self.state, reward = self._update_economy(action)
        self.t += 1
        done = self.t >= self.T
        return self.state, reward, done, False, {}

    # Cần flat obs cho SB3
    def flat_obs(self, obs):
        return obs[0]*27 + obs[1]*9 + obs[2]*3 + obs[3]


# ── Q-LEARNING ─────────────────────────────────────────────────────────────────
def run_qtable(n_episodes=8000, alpha=0.1, gamma_q=0.95):
    env  = VietnamEconEnv()
    Q    = np.zeros((81, 5))
    ep_rewards = []
    eval_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        s = env.flat_obs(obs)
        eps = max(0.05, 1.0 - ep / (n_episodes*0.6))
        total_r = 0
        while True:
            a = env.action_space.sample() if np.random.rand()<eps else int(np.argmax(Q[s]))
            obs2, r, done, _, _ = env.step(a)
            s2 = env.flat_obs(obs2)
            Q[s,a] += alpha*(r + gamma_q*Q[s2].max() - Q[s,a])
            s = s2; total_r += r
            if done: break
        ep_rewards.append(total_r)

        # Eval every 200 ep
        if ep % 200 == 0:
            obs, _ = env.reset()
            s = env.flat_obs(obs)
            er = 0
            while True:
                a = int(np.argmax(Q[s]))
                obs2, r, done, _, _ = env.step(a)
                s = env.flat_obs(obs2); er += r
                if done: break
            eval_rewards.append((ep, er))

    return Q, ep_rewards, eval_rewards

print("Training Q-table...")
Q_table, q_train, q_eval = run_qtable(n_episodes=8000)

# ── DQN (stable-baselines3) ────────────────────────────────────────────────────
class FlatEnv(gym.Env):
    """Wrapper: obs → flat int để Box space cho DQN"""
    def __init__(self):
        super().__init__()
        self._env = VietnamEconEnv()
        self.action_space      = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=80, shape=(1,), dtype=np.float32)

    def reset(self, **kw):
        obs, info = self._env.reset(**kw)
        return np.array([self._env.flat_obs(obs)], dtype=np.float32), info

    def step(self, action):
        obs, r, done, trunc, info = self._env.step(action)
        return np.array([self._env.flat_obs(obs)], dtype=np.float32), r, done, trunc, info

dqn_rewards_ep = []

class RewardCallback(BaseCallback):
    def __init__(self): super().__init__(verbose=0); self.ep_rewards = []; self._cur = 0
    def _on_step(self):
        self._cur += self.locals.get('rewards', np.array([0]))[0]
        dones = self.locals.get('dones', [False])
        if dones[0]: self.ep_rewards.append(self._cur); self._cur = 0
        return True

print("Training DQN (SB3)...")
dqn_env = FlatEnv()
dqn_model = DQN(
    'MlpPolicy', dqn_env,
    learning_rate=3e-4, buffer_size=10000,
    learning_starts=500, batch_size=64,
    target_update_interval=500,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=0
)
cb = RewardCallback()
dqn_model.learn(total_timesteps=80000, callback=cb)  # ~8000 episodes of T=10
dqn_rewards = cb.ep_rewards
print(f"  DQN episodes trained: {len(dqn_rewards)}")

# ── RULE-BASED AGENTS ─────────────────────────────────────────────────────────
def run_fixed(action, n_ep=500):
    env = VietnamEconEnv()
    rewards = []
    for _ in range(n_ep):
        obs, _ = env.reset(); total = 0
        while True:
            _, r, done, _, _ = env.step(action)
            total += r
            if done: break
        rewards.append(total)
    return rewards

print("Running rule-based baselines...")
rb_a1 = run_fixed(1)   # Cân bằng
rb_a3 = run_fixed(3)   # AI dẫn dắt

# ── POLICY HEATMAP ─────────────────────────────────────────────────────────────
action_names = {0:'Trad.', 1:'Cân bằng', 2:'Số hóa', 3:'AI', 4:'Bao trùm'}
policy_grid = np.argmax(Q_table, axis=1).reshape(3,27)  # gdp_level × rest

# ── VISUALIZATION ─────────────────────────────────────────────────────────────
def smooth(x, w=50): return np.convolve(x, np.ones(w)/w, 'valid')

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# 1. Learning curves Q vs DQN
ax1 = fig.add_subplot(gs[0,:2])
q_sm  = smooth(q_train, 200)
dqn_sm = smooth(dqn_rewards, 200) if len(dqn_rewards)>200 else dqn_rewards
ax1.plot(range(len(q_sm)),   q_sm,   '#2563eb', lw=2,   label='Q-table (tabular)')
ax1.plot(range(len(dqn_sm)), dqn_sm, '#ef4444', lw=2,   label='DQN (64×64 MLP)')
ax1.axhline(np.mean(rb_a1), color='gray',   ls='--', lw=1.5, label=f'Rule a1 (μ={np.mean(rb_a1):.2f})')
ax1.axhline(np.mean(rb_a3), color='orange', ls='--', lw=1.5, label=f'Rule a3 (μ={np.mean(rb_a3):.2f})')
ax1.set_title('Learning Curves — Q-table vs DQN vs Rule-based\n(reward per episode, smoothed 200-ep MA)', fontweight='bold')
ax1.set_xlabel('Episode'); ax1.set_ylabel('Tổng phần thưởng (episode)')
ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

# 2. Final performance boxplot
ax2 = fig.add_subplot(gs[0,2])
# Eval Q-table over 500 episodes
env_ev = VietnamEconEnv()
q_eval_rewards = []
for _ in range(500):
    obs, _ = env_ev.reset(); s = env_ev.flat_obs(obs); total = 0
    while True:
        a = int(np.argmax(Q_table[s]))
        obs2, r, done, _, _ = env_ev.step(a)
        s = env_ev.flat_obs(obs2); total += r
        if done: break
    q_eval_rewards.append(total)

# Eval DQN over 500 episodes
dqn_eval = []
eval_flat = FlatEnv()
for _ in range(500):
    obs, _ = eval_flat.reset(); total = 0
    while True:
        action, _ = dqn_model.predict(obs, deterministic=True)
        obs, r, done, _, _ = eval_flat.step(action)
        total += r
        if done: break
    dqn_eval.append(total)

data_bp = [q_eval_rewards, dqn_eval, rb_a1, rb_a3]
bp = ax2.boxplot(data_bp, patch_artist=True,
                 labels=['Q-table', 'DQN', 'Rule-a1', 'Rule-a3'])
colors_bp = ['#2563eb','#ef4444','#9ca3af','#f59e0b']
for patch, c in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax2.set_title('So sánh hiệu suất cuối\n(500 eval episodes)', fontweight='bold')
ax2.set_ylabel('Tổng phần thưởng'); ax2.grid(alpha=0.3, axis='y')
for i, d in enumerate(data_bp):
    ax2.annotate(f'μ={np.mean(d):.2f}', xy=(i+1, np.mean(d)),
                 xytext=(i+1.18, np.mean(d)), fontsize=8, color='black')

# 3. Q-table eval curve
ax3 = fig.add_subplot(gs[1,0])
eval_eps, eval_vals = zip(*q_eval)
ax3.plot(eval_eps, eval_vals, 'o-', color='#2563eb', lw=2, ms=5)
ax3.fill_between(eval_eps, [v*0.95 for v in eval_vals], [v*1.05 for v in eval_vals],
                 alpha=0.15, color='#2563eb')
ax3.set_title('Q-table Eval Reward\n(greedy policy, every 200 ep)', fontweight='bold')
ax3.set_xlabel('Training episode'); ax3.set_ylabel('Eval reward')
ax3.grid(alpha=0.3)

# 4. Policy heatmap (Q-table)
ax4 = fig.add_subplot(gs[1,1])
# Hiển thị action tại mỗi trạng thái (2D: gdp_growth vs digital_index, fix AI=0, U=1)
heatmap_data = np.zeros((3,3), dtype=int)
for g in range(3):
    for d in range(3):
        s_idx = g*27 + d*9 + 0*3 + 1
        heatmap_data[g,d] = np.argmax(Q_table[s_idx])

im = ax4.imshow(heatmap_data, cmap='Set1', vmin=0, vmax=4, aspect='auto')
ax4.set_xticks([0,1,2]); ax4.set_xticklabels(['D=Low','D=Med','D=High'])
ax4.set_yticks([0,1,2]); ax4.set_yticklabels(['GDP=Low','GDP=Med','GDP=High'])
for g in range(3):
    for d in range(3):
        act = heatmap_data[g,d]
        ax4.text(d, g, action_names[act], ha='center', va='center', fontsize=8,
                 color='white', fontweight='bold')
ax4.set_title('Policy π*(s) — Q-table\n(AI=low, U=med; màu=action)', fontweight='bold')
plt.colorbar(im, ax=ax4, ticks=[0,1,2,3,4],
             label='Action (0=Trad, 1=Balanced, 2=Digital, 3=AI, 4=Inclusive)')

# 5. Architecture comparison diagram
ax5 = fig.add_subplot(gs[1,2])
ax5.axis('off')
summary_text = (
    "Kiến trúc so sánh\n"
    "─────────────────────────────────────\n\n"
    "Q-TABLE (Tabular)\n"
    "  • Bảng Q: 81×5 = 405 phần tử\n"
    "  • Cập nhật: Q(s,a) ← Q+α(r+γmax-Q)\n"
    "  • ε-greedy decay: 1.0→0.05\n"
    f"  • μ reward: {np.mean(q_eval_rewards):.3f}\n\n"
    "DQN (Deep Q-Network)\n"
    "  • Mạng: obs→[64]→[64]→5\n"
    "  • Experience replay buffer: 10,000\n"
    "  • Target network update: 500 steps\n"
    f"  • μ reward: {np.mean(dqn_eval):.3f}\n\n"
    "NHẬN XÉT:\n"
    f"  DQN {'cải thiện' if np.mean(dqn_eval)>np.mean(q_eval_rewards) else 'tương đương'}"
    f" {'(+' if np.mean(dqn_eval)>np.mean(q_eval_rewards) else '('}"
    f"{abs(np.mean(dqn_eval)-np.mean(q_eval_rewards)):.2f} reward).\n"
    "  Với state space nhỏ (81 states),\n"
    "  Q-table đủ hiệu quả; DQN hữu ích\n"
    "  hơn khi mở rộng không gian trạng thái."
)
ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f9ff', alpha=0.8))

plt.suptitle(
    'Bài 11 Nâng Cấp — Q-learning Tabular vs DQN (stable-baselines3)\n'
    'Learning Curves · Policy Heatmap · Performance Boxplot · Architecture Comparison',
    fontsize=11, fontweight='bold', y=1.01)
plt.savefig('b11_enhanced.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[✓] Saved b11_enhanced.png")
print(f"\nKết quả cuối:")
print(f"  Q-table mean reward: {np.mean(q_eval_rewards):.3f} ± {np.std(q_eval_rewards):.3f}")
print(f"  DQN     mean reward: {np.mean(dqn_eval):.3f} ± {np.std(dqn_eval):.3f}")
print(f"  Rule-a1 mean reward: {np.mean(rb_a1):.3f} ± {np.std(rb_a1):.3f}")
print(f"  Rule-a3 mean reward: {np.mean(rb_a3):.3f} ± {np.std(rb_a3):.3f}")
print("[✓] Hoàn thành Bài 11 Nâng Cấp")
