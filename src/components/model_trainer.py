import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


class ModelTrainer:

    def initiate_model_trainer(self, train_arr, test_arr):

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor()
        }

        best_score = -1
        best_model = None

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)

            print(f"{name} R2 Score:", score)

            if score > best_score:
                best_score = score
                best_model = model

        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        print("Best model saved")
        print("Final Model Score:", best_score)

        return best_score