
# ------------------------------------------------------------
# QUESTION 12(i): ACTUAL-VS-PREDICTED REGRESSION PLOT
# ------------------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(9, 6))

ax1.scatter(
    y_test,
    y_pred,
    alpha=0.65,
    label="Test observations"
)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

ax1.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle="--",
    linewidth=2,
    label="Perfect prediction"
)

ax1.set_xlabel("Actual insurance charges ($)")
ax1.set_ylabel("Predicted insurance charges ($)")
ax1.set_title("Linear Regression: Actual vs Predicted Insurance Charges")

ax1.legend()
ax1.grid(True, alpha=0.25)

fig1.tight_layout()
fig1.savefig("regression_actual_vs_predicted.png", dpi=300)

plt.show()
plt.close(fig1)


# ------------------------------------------------------------
# QUESTION 12(ii): RESIDUAL PLOT
# ------------------------------------------------------------

residuals = y_test - y_pred

fig2, ax2 = plt.subplots(figsize=(9, 6))

ax2.scatter(
    y_pred,
    residuals,
    alpha=0.65
)

ax2.axhline(
    y=0,
    linestyle="--",
    linewidth=2
)

ax2.set_xlabel("Predicted insurance charges ($)")
ax2.set_ylabel("Residual (actual - predicted) ($)")
ax2.set_title("Residual Plot for Linear Regression")

ax2.grid(True, alpha=0.25)

fig2.tight_layout()
fig2.savefig("regression_residual_plot.png", dpi=300)

plt.show()
plt.close(fig2)
