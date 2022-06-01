import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml


params = yaml.safe_load(open("params.yaml"))['train']


# Read in data
print("Reading data...")
df = pd.read_csv(os.path.join(sys.argv[1], "df_preprocessed.csv"), index_col=False)

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



# clf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

model = Pipeline([('scale', StandardScaler()),
                 ('model', LinearRegression())])

model.fit(X_train, y_train)

print("Finished training")

joblib.dump(model, "model.pkl")

os.makedirs(os.path.join("data", "test"), exist_ok=True)
df_test.to_csv("data/test/df_test.csv", index=False)


