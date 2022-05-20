from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd

# Read in data

df = pd.read_csv("data/train.csv")

df = ( 
    df.pipe(
        lambda _df:
            _df.assign(
                time = pd.to_datetime(_df.time)
            )
            .astype({
                'x': 'category',
                'y': 'category',
                'direction': 'category',
            })
    )
    .pipe(
        lambda _df:
            _df.assign(
                day = _df.time.dt.day,
                hour = _df.time.dt.hour,
                dayofweek = _df.time.dt.dayofweek,
                month = _df.time.dt.month,
                road = f'{_df.x}{_df.y}{_df.direction}'
            )
    )
    .pipe(
        lambda _df:
            _df.assign(
                road = _df.road.astype('category').cat.codes,
            )
    )
    .drop(["row_id","x","y","direction", "time"], axis=1)
)

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.drop("congestion",axis=1), df['congestion'], test_size=0.33, random_state=42)
# Fit a model
depth = 4
clf = RandomForestRegressor(max_depth=depth)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)

with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")

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