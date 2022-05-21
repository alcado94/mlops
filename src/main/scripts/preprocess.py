from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA

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

def setCongestionColumnByRoad(df):
    temp = df.groupby(["road", "hour", "minute"]).median().reset_index()[["road", "hour", "minute", "congestion"]]
    temp = temp.rename(columns={"congestion": "month_hour_congestion"})
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
        # .pipe(lambda _df:
        #     transformPCA(_df.drop("congestion",axis=1), _df['congestion'])
        # )
)