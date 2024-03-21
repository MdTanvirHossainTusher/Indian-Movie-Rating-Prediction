import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def wrangle(filepath):
    # Read csv file
    df = pd.read_csv(filepath, encoding="latin1")

    if type(df["Year"]) == "object":
        # Replace parenthesis and convert `Year` from obejct to float
        df["Year"] = df["Year"].str.replace(r"\(|\)", "", regex=True).astype("float64")

    else:
        # Convert `Year`` to numeric
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        # `Year` is not object and has '-' in it
        mask = (df["Year"].notnull()) & (df["Year"] < 0)

        # remove '-'
        df.loc[mask, "Year"] = df.loc[mask, "Year"].abs()

    # Remove non-numeric characters (comma, letters etc.)
    df["Votes"] = df["Votes"].replace("[^\d.]", "", regex=True).astype("float64")

    # Split genres and create binary indicator columns
    genres = df["Genre"].str.get_dummies(", ")

    # Concat old and new dataframe
    df = pd.concat([df, genres], axis=1)

    # Fill categorical null values with mode
    categorical_columns = ["Director", "Actor 1", "Actor 2", "Actor 3"]
    for column in categorical_columns:
        most_frequent_value = df[column].mode()[0]
        df[column].fillna(most_frequent_value, inplace=True)

    # Remove outliers in `Votes`
    low, high = df["Votes"].quantile([0.1, 0.9])
    mask_votes = df["Votes"].between(low, high)

    df = df[mask_votes]

    # Initialize `LabelEncoder` and convert categorical value to numerical value
    encoder = LabelEncoder()
    df["Directors"] = encoder.fit_transform(df["Director"])
    df["Lead Actor"] = encoder.fit_transform(df["Actor 1"])
    df["Supporting Actor"] = encoder.fit_transform(df["Actor 2"])
    df["Side Actor"] = encoder.fit_transform(df["Actor 3"])

    # Columns to be dropped due to above 50% null values, binary indicator columns, high cardinality and categorical value
    drop_columns = [
        "Duration",
        "Genre",
        "Name",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3",
    ]

    # Drop columns
    df.drop(columns=drop_columns, inplace=True)

    # Drop old index and reset new index
    df.reset_index(drop=True, inplace=True)

    return df


def make_prediction(data_filepath, model_filepath):
    X_test = wrangle(data_filepath)

    # Scaling the testing data
    scale = StandardScaler()
    X_test = scale.fit_transform(X_test)

    # load model
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)

    y_test_pred = model.predict(X_test)
    y_test_pred = pd.Series(y_test_pred)
    return y_test_pred
