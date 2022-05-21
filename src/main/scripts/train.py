import sys
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

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

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

def setCongestionColumnByRoad(df):
    temp = df.groupby(["road", "hour", "minute"]).median().reset_index()[["road", "hour", "minute", "congestion"]]
    temp = temp.rename(columns={"congestion": "road_congestion"})
    df = df.merge(temp, on=[ "road", "hour", "minute"])
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
        .pipe(setCongestionColumnByRoad)
        .drop(["row_id","x","y","direction", "time"], axis=1)
        .pipe(lambda _df:
            transformPCA(_df.drop("congestion",axis=1), _df['congestion'])
        )
)
print("Finished preprocessing")

df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)

X_train = df_train.drop(['congestion'], axis=1)
y_train = df_train['congestion']


print("Training...")
# Fit a model

random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 4)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


# rf = RandomForestRegressor()
# clf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


clf = RandomForestRegressor(n_estimators=1, random_state = 42)

clf.fit(X_train, y_train)

print("Finished training")

joblib.dump(clf, "model.pkl")

df_test.to_csv(os.path.join(sys.argv[1], "df_test.csv"), index=False)


