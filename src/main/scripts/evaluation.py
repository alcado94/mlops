

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib


y_pred = clf.predict(X_test)


with open("metrics.txt", "w") as outfile:
    outfile.write("Mean Absolute Error: " + str(metrics.mean_absolute_error(y_test, y_pred)) + "\n")
    outfile.write("Mean Squared Error: " + str(metrics.mean_squared_error(y_test, y_pred)) + "\n")
    outfile.write("Root Mean Squared Error: " + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) + "\n")

# Plot it
importances = clf.feature_importances_
forest_importances = pd.Series(importances, index=list(X_train.columns))
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.savefig("plot.png")


joblib.dump(clf, "model.pkl")