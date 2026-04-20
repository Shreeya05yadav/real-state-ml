import pandas as pd
import numpy as np
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataTransformation:

    def get_data_transformer(self, numerical_cols, categorical_cols):

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        target_column = "PRICE"

        drop_cols = [
            "MAIN_ADDRESS",
            "ADMINISTRATIVE_AREA_LEVEL_2",
            "BROKERTITLE"
        ]

        train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
        test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # ---------- Feature Engineering ----------
        def add_features(df):
            df = df.copy()

            df["AREA_PER_ROOM"] = df["PROPERTYSQFT"] / (df["BEDS"] + df["BATH"] + 1)
            df["BATH_PER_BED"] = df["BATH"] / (df["BEDS"] + 1)
            df["TOTAL_ROOMS"] = df["BEDS"] + df["BATH"]
            df["SQFT_PER_BED"] = df["PROPERTYSQFT"] / (df["BEDS"] + 1)
            df["BATH_ROOM_DIFF"] = df["BATH"] - df["BEDS"]

            df["SIZE_CATEGORY"] = pd.cut(
                df["PROPERTYSQFT"],
                bins=[0, 500, 1000, 2000, 5000, np.inf],
                labels=["VerySmall", "Small", "Medium", "Large", "Luxury"]
            )

            # Safe location
            if "LOCALITY" in df.columns:
                df["LOCATION"] = (
                    df["STATE"].astype(str) + "_" +
                    df["LOCALITY"].astype(str) + "_" +
                    df["SUBLOCALITY"].astype(str)
                )
            else:
                df["LOCATION"] = (
                    df["STATE"].astype(str) + "_" +
                    df["SUBLOCALITY"].astype(str)
                )

            return df

        X_train = add_features(X_train)
        X_test = add_features(X_test)

        # Reduce cardinality
        top_locations = X_train["LOCATION"].value_counts().nlargest(30).index

        X_train["LOCATION"] = np.where(X_train["LOCATION"].isin(top_locations), X_train["LOCATION"], "Other")
        X_test["LOCATION"] = np.where(X_test["LOCATION"].isin(top_locations), X_test["LOCATION"], "Other")

        # Drop original location cols
        X_train.drop(columns=["STATE", "SUBLOCALITY", "LOCALITY"], inplace=True, errors="ignore")
        X_test.drop(columns=["STATE", "SUBLOCALITY", "LOCALITY"], inplace=True, errors="ignore")

        # Columns
        numerical_cols = [
            "BEDS", "BATH", "PROPERTYSQFT",
            "LATITUDE", "LONGITUDE",
            "AREA_PER_ROOM", "BATH_PER_BED",
            "TOTAL_ROOMS", "SQFT_PER_BED", "BATH_ROOM_DIFF"
        ]

        categorical_cols = ["TYPE", "LOCATION", "SIZE_CATEGORY"]

        preprocessor = self.get_data_transformer(numerical_cols, categorical_cols)

        X_train_arr = preprocessor.fit_transform(X_train)
        X_test_arr = preprocessor.transform(X_test)

        # Save preprocessor
        with open("artifacts/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        y_train = y_train.to_numpy().reshape(-1, 1)
        y_test = y_test.to_numpy().reshape(-1, 1)

        train_arr = np.concatenate([X_train_arr.toarray(), y_train], axis=1)
        test_arr = np.concatenate([X_test_arr.toarray(), y_test], axis=1)

        print("Data transformation completed")

        return train_arr, test_arr