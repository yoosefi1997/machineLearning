import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([1, 3, 5, 7])

mdl_x = LinearRegression()
mdl_x.fit(x, y)

m = mdl_x.coef_[0]
c = mdl_x.intercept_
print(f"فرمول خط کشف‌شده: y = {m:.1f}*x + ({c:.1f})")

result_x = mdl_x.predict([[5]])
print(f"مقدار پیش‌بینی‌شده برای x=5 برابر است با: {result_x[0]}")

plt.scatter(x, y, color="red", label="Realy Data")
plt.plot(x, mdl_x.predict(x), color="blue", label="Regresion line")
plt.scatter([5], result_x, color="green", s=100, label="Predict-x (x=5)")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


