"""
Bài 1 Nâng Cấp — Hàm Sản Xuất Cobb-Douglas Mở Rộng
Kỹ thuật nâng cao:
  1. OLS-CRS (constrained, CRS assumption)
  2. Ridge Regression LOO-CV chọn lambda
  3. Bootstrap CI (2000 resample)
  4. Growth accounting 3 phương pháp
  5. Dự báo 2030 với dải uncertainty
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA = {
    'year': [2020, 2021, 2022, 2023, 2024, 2025],
    'Y':  [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
    'K':  [16500,  17800,  19600,  21300,   23500,   25900],
    'L':  [53.6,   50.5,   51.7,   52.4,    52.9,    53.4],
    'D':  [12.0,   12.7,   14.3,   16.5,    18.3,    19.5],
    'AI': [55.6,   60.2,   65.4,   67.0,    73.8,    80.1],
    'H':  [24.1,   26.1,   26.2,   27.0,    28.4,    29.2],
}
df = pd.DataFrame(DATA)
n  = len(df)

lnY  = np.log(df['Y'].values)
lnK  = np.log(df['K'].values)
lnL  = np.log(df['L'].values)
lnD  = np.log(df['D'].values)
lnAI = np.log(df['AI'].values)
lnH  = np.log(df['H'].values)

# CRS transformation
y_tilde = lnY  - lnH
X_raw   = np.column_stack([lnK-lnH, lnL-lnH, lnD-lnH, lnAI-lnH])

# Standardise for Ridge
X_mean = X_raw.mean(axis=0)
X_std  = X_raw.std(axis=0); X_std[X_std==0] = 1
X_z    = (X_raw - X_mean) / X_std
y_c    = y_tilde - y_tilde.mean()

names     = ['alpha_K','beta_L','gamma_D','delta_AI']
textbook5 = np.array([0.33, 0.42, 0.10, 0.08, 0.07])

# ── OLS CRS ────────────────────────────────────────────────────────────────────
X_ols = np.column_stack([np.ones(n), X_raw])
b_ols, _, _, _ = np.linalg.lstsq(X_ols, y_tilde, rcond=None)
theta_ols  = 1 - sum(b_ols[1:])
ols5       = list(b_ols[1:]) + [theta_ols]
y_hat_ols  = X_ols @ b_ols + lnH
MAPE_ols   = np.mean(np.abs(np.exp(y_hat_ols) - df['Y'].values)/df['Y'].values*100)

print("OLS-CRS hệ số:", dict(zip(['c','α','β','γ','δ','θ'], b_ols.tolist()+[theta_ols])))
print(f"MAPE OLS = {MAPE_ols:.3f}%\n")

# ── RIDGE LOO-CV ────────────────────────────────────────────────────────────────
lambdas = np.logspace(-4, 2, 120)
loo_mse = []
for lam in lambdas:
    mse = 0
    for i in range(n):
        idx = [j for j in range(n) if j!=i]
        Xtr,ytr = X_z[idx], y_c[idx]
        br = np.linalg.solve(Xtr.T@Xtr + lam*np.eye(4), Xtr.T@ytr)
        pred = X_z[i]@br + y_tilde.mean()
        mse += (y_tilde[i] - pred)**2
    loo_mse.append(mse/n)

best_lam = lambdas[np.argmin(loo_mse)]
print(f"Ridge λ* (LOO-CV) = {best_lam:.5f}")

# Ridge estimate
br_z   = np.linalg.solve(X_z.T@X_z + best_lam*np.eye(4), X_z.T@y_c)
br_raw = br_z / X_std
icept  = y_tilde.mean() - X_mean @ br_raw

# Normalise to CRS
total  = sum(br_raw)
if total > 0:
    ridge4 = br_raw / total
else:
    ridge4 = np.array([0.33,0.42,0.10,0.08]) / sum([0.33,0.42,0.10,0.08])
theta_ridge = 1 - sum(ridge4)
ridge5 = list(ridge4) + [theta_ridge]

y_hat_ridge = (icept + X_raw @ br_raw) + lnH
MAPE_ridge  = np.mean(np.abs(np.exp(y_hat_ridge)-df['Y'].values)/df['Y'].values*100)
print(f"MAPE Ridge = {MAPE_ridge:.3f}%")

# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────
np.random.seed(42)
B = 2000
boot = np.zeros((B,5))
for b in range(B):
    idx = np.random.choice(n, n, replace=True)
    Xb,yb = X_z[idx], y_c[idx]
    try:
        brb  = np.linalg.solve(Xb.T@Xb + best_lam*np.eye(4), Xb.T@yb)
        rb   = brb / X_std
        tot  = sum(rb)
        if tot > 0:
            rn = rb/tot; boot[b] = list(rn)+[1-sum(rn)]
        else: boot[b] = boot[b-1]
    except: boot[b] = boot[b-1]

ci_lo = np.percentile(boot, 2.5,  axis=0)
ci_hi = np.percentile(boot, 97.5, axis=0)

print("\n{'Param':<10} {'Ridge-CRS':>10} {'CI Lo':>8} {'CI Hi':>8} {'Textbook':>10} {'InCI':>6}")
for nm,est,lo,hi,ref in zip(['α(K)','β(L)','γ(D)','δ(AI)','θ(H)'],
                              ridge5, ci_lo, ci_hi, textbook5):
    mk = '✓' if lo<=ref<=hi else '✗'
    print(f"  {nm:<8} {est:>10.4f} {lo:>8.4f} {hi:>8.4f} {ref:>10.4f} {mk:>6}")

# ── TFP & GROWTH ACCOUNTING ───────────────────────────────────────────────────
a,b_l,g,d,th = ridge5
df['A_t']    = df['Y'] / (df['K']**a * df['L']**b_l * df['D']**g * df['AI']**d * df['H']**th)
df['Y_ridge']= df['A_t'].mean() * df['K']**a * df['L']**b_l * df['D']**g * df['AI']**d * df['H']**th
df['MAPE_r'] = abs(df['Y_ridge']-df['Y'])/df['Y']*100

ga_rows = []
for i in range(1,n):
    dY = np.log(df.loc[i,'Y'])-np.log(df.loc[i-1,'Y'])
    dA = np.log(df.loc[i,'A_t'])-np.log(df.loc[i-1,'A_t'])
    ga_rows.append({
        'Năm': f"{int(df.loc[i-1,'year'])}-{int(df.loc[i,'year'])}",
        'ΔlnY%': round(dY*100,2),
        'TFP':  round(dA*100,2),
        'K': round(a*(np.log(df.loc[i,'K'])-np.log(df.loc[i-1,'K']))*100,2),
        'L': round(b_l*(np.log(df.loc[i,'L'])-np.log(df.loc[i-1,'L']))*100,2),
        'D': round(g*(np.log(df.loc[i,'D'])-np.log(df.loc[i-1,'D']))*100,2),
        'AI':round(d*(np.log(df.loc[i,'AI'])-np.log(df.loc[i-1,'AI']))*100,2),
        'H': round(th*(np.log(df.loc[i,'H'])-np.log(df.loc[i-1,'H']))*100,2),
    })
ga = pd.DataFrame(ga_rows)
print("\n",ga.to_string(index=False))

# ── DỰ BÁO 2030 ───────────────────────────────────────────────────────────────
K30,L30,D30,AI30,H30 = 25900*1.06**5, 53.4*1.01**5, 30.0, 100.0, 35.0
Al = df['A_t'].iloc[-1]
Y_hi   = Al*(1.012)**5 * K30**a * L30**b_l * D30**g * AI30**d * H30**th
Y_base = Al*(1.008)**5 * K30**a * L30**b_l * D30**g * AI30**d * H30**th
Y_lo   = Al*(1.004)**5 * K30**a * L30**b_l * D30**g * AI30**d * H30**th
print(f"\nDự báo GDP 2030: Lo={Y_lo:,.0f} | Base={Y_base:,.0f} | Hi={Y_hi:,.0f} nghìn tỷ VND")

# ── VISUALIZATION ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
ax1.semilogx(lambdas, loo_mse, '#2563eb', lw=2)
ax1.axvline(best_lam, color='red', ls='--', lw=1.8, label=f'λ*={best_lam:.4f}')
ax1.set_title('Ridge LOO-CV chọn λ*', fontweight='bold')
ax1.set_xlabel('λ (log)'); ax1.set_ylabel('MSE'); ax1.legend(); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0,1])
ax2.plot(df['year'],df['Y'],'ks--',ms=8,lw=2,label='Thực tế')
ax2.plot(df['year'],np.exp(y_hat_ols),'bo-',ms=6,lw=2,label=f'OLS-CRS ({MAPE_ols:.2f}%)')
ax2.plot(df['year'],df['Y_ridge'],'r^-',ms=6,lw=2,label=f'Ridge ({df["MAPE_r"].mean():.2f}%)')
ax2.set_title('Thực tế vs OLS vs Ridge', fontweight='bold')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_ylabel('GDP (nghìn tỷ VND)')

ax3 = fig.add_subplot(gs[0,2])
x = np.arange(5); w = 0.25
ols5n = np.array(ols5); ols5n = ols5n/ols5n.sum() if ols5n.sum()!=0 else ols5n
ax3.bar(x-w, ols5n,         w, label='OLS-CRS', color='#2563eb', alpha=0.85)
ax3.bar(x,   np.array(ridge5), w, label='Ridge-CRS', color='#16a34a', alpha=0.85,
        yerr=np.array([(c-l, h-c) for c,l,h in zip(ridge5,ci_lo,ci_hi)]).T, capsize=3, ecolor='#166534')
ax3.bar(x+w, textbook5,    w, label='Đề bài', color='#f59e0b', alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(['α(K)','β(L)','γ(D)','δ(AI)','θ(H)'],fontsize=9)
ax3.set_title('So sánh hệ số (±95% Boot CI)', fontweight='bold')
ax3.legend(fontsize=8); ax3.grid(axis='y', alpha=0.3)

ax4 = fig.add_subplot(gs[1,:2])
factors=['TFP','K','L','D','AI','H']
colors =['#f59e0b','#2563eb','#16a34a','#7c3aed','#ef4444','#06b6d4']
bot_pos = np.zeros(len(ga)); bot_neg = np.zeros(len(ga))
for f,c in zip(factors,colors):
    vals = ga[f].values.astype(float)
    pos = np.where(vals>=0, vals, 0); neg = np.where(vals<0, vals, 0)
    ax4.bar(ga['Năm'], pos, bottom=bot_pos, color=c, alpha=0.85, label=f)
    ax4.bar(ga['Năm'], neg, bottom=bot_neg, color=c, alpha=0.5)
    bot_pos += pos; bot_neg += neg
ax4.plot(ga['Năm'], ga['ΔlnY%'], 'ko--', ms=6, lw=2, label='ΔlnY%', zorder=5)
ax4.axhline(0, color='black', lw=0.8)
ax4.set_title('Phân rã tăng trưởng GDP (%/năm) — Ridge-CRS', fontweight='bold')
ax4.legend(loc='upper right', fontsize=8, ncol=4); ax4.grid(axis='y', alpha=0.3)

ax5 = fig.add_subplot(gs[1,2])
ax5.plot(df['year'], df['A_t'], 'o-', color='#2563eb', lw=2.5, ms=8, label='TFP lịch sử')
proj_yrs = list(range(2025,2031))
al = df['A_t'].iloc[-1]
ax5.fill_between(proj_yrs,
    [al*(1.004)**i for i in range(6)],
    [al*(1.012)**i for i in range(6)],
    alpha=0.2, color='orange', label='Dải TFP 2030')
ax5.plot(proj_yrs, [al*(1.008)**i for i in range(6)], '--', color='orange', lw=2)
ax5.annotate(f'Y₂₀₃₀\n{Y_base:,.0f}', xy=(2030, al*(1.008)**5),
             xytext=(2027.5, al*(1.010)**3), fontsize=8,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax5.set_title('TFP & Dự báo GDP 2030', fontweight='bold')
ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

plt.suptitle(
    'Bài 1 Nâng Cấp — OLS-CRS · Ridge (LOO-CV, λ*) · Bootstrap CI (B=2000)\n'
    'Growth Accounting · Dự báo 2030 (3 kịch bản TFP)',
    fontsize=11, fontweight='bold', y=1.01)
plt.savefig('b01_enhanced.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[✓] Saved b01_enhanced.png")
