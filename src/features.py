import pandas as pd
import numpy as np


class FeatureEngineer:

    def clean_demographic(self, df):
        df = df.copy()

        df.columns = df.columns.str.strip()

        drop_cols = ["User_ID", "Product_ID"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        for col in ["Product_Category_2", "Product_Category_3"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        df = df.drop_duplicates()

        return df

    def clean_clickstream(self, df):
        df = df.copy()

        df.columns = df.columns.str.strip()

        drop_cols = ["Revenue"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        df = df.drop_duplicates()

        if "Weekend" in df.columns:
            df["Weekend"] = df["Weekend"].astype(int)

        return df

    def clean_purchases(self, df):
        df = df.copy()

        df.columns = df.columns.str.strip()

        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df["hour"] = df["Timestamp"].dt.hour
            df["day_of_week"] = df["Timestamp"].dt.dayofweek
            df = df.drop(columns=["Timestamp"])

        df = df.drop_duplicates()

        return df

    def aggregate_purchases(self, df):
        df = df.copy()

        user_col = None
        for possible in ["UserID", "User_ID", "user_id", "UserId"]:
            if possible in df.columns:
                user_col = possible
                break

        if user_col is None:
            df["row_id"] = range(len(df))
            user_col = "row_id"

        agg_dict = {}

        if "Amount" in df.columns:
            agg_dict["Amount"] = ["mean", "sum", "count"]

        if "hour" in df.columns:
            agg_dict["hour"] = "mean"

        if "day_of_week" in df.columns:
            agg_dict["day_of_week"] = "mean"

        purchase_agg = df.groupby(user_col).agg(agg_dict)

        purchase_agg.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col
            for col in purchase_agg.columns
        ]

        purchase_agg = purchase_agg.reset_index()

        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c != user_col]

        for col in categorical_cols:
            counts = pd.crosstab(df[user_col], df[col])
            counts.columns = [f"{col}_{value}_count" for value in counts.columns]
            purchase_agg = purchase_agg.merge(counts, left_on=user_col, right_index=True, how="left")

        return purchase_agg.drop(columns=[user_col], errors="ignore")

    def encode(self, df):
        df = df.copy()

        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))

        return df