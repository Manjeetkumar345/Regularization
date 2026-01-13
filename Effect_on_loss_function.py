import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_regression


X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1,noise=20,random_state=13)
X.shape,y.shape

def ridge_loss(m, b, alpha):
    y_hat = m * X.ravel() + b
    return np.mean((y - y_hat) ** 2) + alpha * (m ** 2)

def lasso_loss(m, b, alpha):
    y_hat = m * X.ravel() + b
    return np.mean((y - y_hat) ** 2) + alpha * np.abs(m)

m_vals = np.linspace(-45, 100, 100)
alphas = [0,1,5,10,20,30,40,50,100]
b = 2.29

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Axis formatting
ax1.set_title("Ridge Regression Loss")
ax2.set_title("Lasso Regression Loss")

ax1.set_xlabel("Slope (m)")
ax2.set_xlabel("Slope (m)")

ax1.set_ylabel("Loss")
ax2.set_ylabel("Loss")

# Empty lines (updated in animation)
ridge_line, = ax1.plot([], [], lw=2)
lasso_line, = ax2.plot([], [], lw=2)


def update(frame):
    alpha = alphas[frame]

    ridge_y = [ridge_loss(m, b, alpha) for m in m_vals]
    lasso_y = [lasso_loss(m, b, alpha) for m in m_vals]

    ridge_line.set_data(m_vals, ridge_y)
    lasso_line.set_data(m_vals, lasso_y)

    ax1.set_title(f"Ridge Loss (α = {alpha})")
    ax2.set_title(f"Lasso Loss (α = {alpha})")

    ax1.relim()
    ax1.autoscale_view()

    ax2.relim()
    ax2.autoscale_view()

    return ridge_line, lasso_line

# Create animation

anim = FuncAnimation(
    fig,
    update,
    frames=len(alphas),
    interval=800,
    blit=False
)

anim.save(
    "ridge_vs_lasso_alpha_animation.gif",
    writer=PillowWriter(fps=1)
)

plt.show()
