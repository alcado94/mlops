
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import sys
import os
import pickle
import json
from dvclive import Live

live = Live("evaluation")

# if len(sys.argv) != 3:
#     sys.stderr.write("Arguments error. Usage:\n")
#     sys.stderr.write("\tpython evaluate.py model features\n")
#     sys.exit(1)

model = joblib.load(sys.argv[1])
df_test = pd.read_csv(os.path.join(sys.argv[2], "df_test.csv"), index_col=False)

y_test = df_test['congestion']
X_test = df_test.drop(['congestion'], axis=1)
y_pred = model.predict(X_test)

live.log("mae", metrics.mean_absolute_error(y_test, y_pred))
live.log("mse", metrics.mean_squared_error(y_test, y_pred))
live.log("rmse", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

with open("metrics.txt", "w") as outfile:
    outfile.write("Mean Absolute Error: " + str(metrics.mean_absolute_error(y_test, y_pred)) + "\n")
    outfile.write("Mean Squared Error: " + str(metrics.mean_squared_error(y_test, y_pred)) + "\n")
    outfile.write("Root Mean Squared Error: " + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) + "\n")

# Plot it
importances = model.feature_importances_
forest_importances = pd.Series(importances, index=list(X_test.columns))
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.savefig("plot.png")
