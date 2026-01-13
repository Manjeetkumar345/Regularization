from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=2
)

alphas = [0,1,2,4,8,16,32]


ridge_coefs, ridge_r2 = [], []
for a in alphas:
    reg = Ridge(alpha=a)
    reg.fit(X_train, y_train)
    ridge_coefs.append(reg.coef_)
    ridge_r2.append(r2_score(y_test, reg.predict(X_test)))


lasso_coefs, lasso_r2 = [], []
for a in alphas:
    reg = Lasso(alpha=a, max_iter=5000)
    reg.fit(X_train, y_train)
    lasso_coefs.append(reg.coef_)
    lasso_r2.append(r2_score(y_test, reg.predict(X_test)))


fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 4), sharey=True
)

bars_ridge = ax1.bar(data.feature_names, ridge_coefs[0])
bars_lasso = ax2.bar(data.feature_names, lasso_coefs[0])

ymin = min(np.min(ridge_coefs), np.min(lasso_coefs)) - 0.2
ymax = max(np.max(ridge_coefs), np.max(lasso_coefs)) + 0.2
ax1.set_ylim(ymin, ymax)

ax1.set_title("Ridge")
ax2.set_title("Lasso")

for ax in (ax1, ax2):
    ax.tick_params(axis='x', rotation=45)


def update(frame):
    for bar, h in zip(bars_ridge, ridge_coefs[frame]):
        bar.set_height(h)

    for bar, h in zip(bars_lasso, lasso_coefs[frame]):
        bar.set_height(h)

    fig.suptitle(
        f"Alpha = {alphas[frame]} | "
        f"Ridge R2_score = {ridge_r2[frame]:.2f} | "
        f"Lasso R2_score = {lasso_r2[frame]:.2f}",
        fontsize=12
    )
    return (*bars_ridge, *bars_lasso)

anim = FuncAnimation(
    fig,
    update,
    frames=len(alphas),
    interval=1200
)

anim.save(
    "ridge_vs_lasso_compact.gif",
    writer=PillowWriter(fps=1)
)

plt.tight_layout()
plt.show()
