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
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV

PCA_N_COMPONENTS = 4


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

def setCongestionColumnByDayOfWeekAndHour(df):
    temp = df.groupby(["dayofweek", "road", "hour", "minute"]).median().reset_index()[["dayofweek", "road", "hour", "minute", "congestion"]]
    temp = temp.rename(columns={"congestion": "dayofweek_hour_congestion"})
    df = df.merge(temp, on=["dayofweek", "road", "hour", "minute"])
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

def transformPCA(df, label):

    pca = PCA(n_components=PCA_N_COMPONENTS, svd_solver='full')
    df = pca.fit_transform(df)

    with open("params.txt", "w") as outfile:
        outfile.write("\nInfo PCA: \n")
        outfile.write("N Compoments: " + str(PCA_N_COMPONENTS) + "\n")
        outfile.write("Explained Variance Ratio: " + str(pca.explained_variance_ratio_) + "\n")
        outfile.write("Singular values: " + str(pca.singular_values_) + "\n\n")
    
    toret = pd.DataFrame(df)
    toret['congestion'] = label.values
    return toret

# Read in data
print("Reading data...")
df = pd.read_csv("data/train.csv")

df = ( 
    df.pipe(setDataTypes)
        .pipe(createCategoryColumnRoad)
        .pipe(extractExtraInfoFromTime)
        .pipe(setCongestionColumnByDayAndHour)
        .pipe(setCongestionColumnByMonth)
        .pipe(setCongestionColumnByHourAndMonth)
        .pipe(setCongestionColumnByDayOfWeekAndHour)
        .drop(["row_id","x","y","direction", "time"], axis=1)
        .pipe(lambda _df:
            transformPCA(_df.drop("congestion",axis=1), _df['congestion'])
        )
)
print("Finished preprocessing")

X_train, X_test, y_train, y_test = train_test_split(df.drop("congestion",axis=1), df['congestion'], test_size=0.33, random_state=42)


print("Training...")
# Fit a model
#
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 4)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
clf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model

# clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state = 42)

clf.fit(X_train, y_train)

print("Finished training")

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