from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import pandas as pd
import joblib

def setDataTypes(df):
    return df.assign(
                time = pd.to_datetime(df.time)
            )

def createCategoryColumnRoad(df):
    return df.assign(
                road = (df.x.astype(str) + df.y.astype(str) + df.direction).astype('category').cat.codes
            )

def extractExtraInfoFromTime(df):
    return df.assign(
                day = df.time.dt.day,
                hour = df.time.dt.hour,
                minute = df.time.dt.minute,
                dayofweek = df.time.dt.dayofweek,
                month = df.time.dt.month,
                weekend = (df.time.dt.weekday >= 5),
            )

def setCongestionColumnByDayAndHour(df):
    temp = df.groupby(["day", "road", "hour", "minute"]).median().reset_index()[["day", "road", "hour", "minute", "congestion"]]
    temp = temp.rename(columns={"congestion": "day_hour_congestion"})
    df = df.merge(temp, on=["day", "road", "hour", "minute"])
    return df

def setCongestionColumnByMonth(df):
    temp = df.groupby(["month", "road"]).median().reset_index()[["month", "road", "congestion"]]
    temp = temp.rename(columns={"congestion": "month_congestion"})
    df = df.merge(temp, on=["month", "road"])
    return df

def setCongestionColumnByHourAndMonth(df):
    temp = df.groupby(["month", "road", "hour", "minute"]).median().reset_index()[["month", "road", "hour", "minute", "congestion"]]
    temp = temp.rename(columns={"congestion": "month_hour_congestion"})
    df = df.merge(temp, on=["month", "road", "hour", "minute"])
    return df

# Read in data
df = pd.read_csv("data/train.csv")

df = ( 
    df.pipe(setDataTypes)
        .pipe(createCategoryColumnRoad)
        .pipe(extractExtraInfoFromTime)
        .pipe(setCongestionColumnByDayAndHour)
        .pipe(setCongestionColumnByMonth)
        .pipe(setCongestionColumnByHourAndMonth)
        .drop(["row_id","x","y","direction", "time"], axis=1)
)

X_train, X_test, y_train, y_test = train_test_split(df.drop("congestion",axis=1), df['congestion'], test_size=0.33, random_state=42)
# Fit a model
clf = RandomForestRegressor(n_estimators=20)

clf.fit(X_train, y_train)

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