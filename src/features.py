"""
Feature Engineering and Preprocessing Pipeline for Ames Housing.
Workshop No. 1 â€” Data Preparation & Supervised Learning Baselines.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero OrdeÃ±ez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Ordinal quality mappings
QUAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
EXPOSURE_MAP = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
FIN_TYPE_MAP = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
GARAGE_FIN_MAP = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}
CENTRAL_AIR_MAP = {"Y": 1, "N": 0}

NONE_COLS = [
    "Alley", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
    "BsmtFin Type 1", "BsmtFin Type 2", "Fireplace Qu",
    "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Pool QC", "Fence", "Misc Feature", "Mas Vnr Type"
]

ZERO_COLS = [
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath", "Garage Cars", "Garage Area",
    "Mas Vnr Area"
]

ORDINAL_COLS = [
    "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond",
    "Heating QC", "Kitchen Qual", "Fireplace Qu", "Garage Qual",
    "Garage Cond", "Pool QC", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Garage Finish", "Central Air"
]


class AmesPreprocessor:
    """
    Leakage-free preprocessing and feature engineering pipeline.
    All statistics and transformers are fitted strictly on the training set.
    """

    def __init__(self, remove_train_outliers: bool = True):
        self.remove_train_outliers = remove_train_outliers
        self.lot_frontage_medians = {}
        self.lot_frontage_global_median = 68.0
        self.categorical_modes = {}
        self.numerical_medians = {}
        self.nominal_cols = []
        self.one_hot_encoder = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific aggregate and interaction features."""
        data = df.copy()

        # Handle 'None' categorical absence
        for col in NONE_COLS:
            if col in data.columns:
                data[col] = data[col].fillna("None").astype(str)

        # Handle '0' numeric basement/garage fields
        for col in ZERO_COLS:
            if col in data.columns:
                data[col] = data[col].fillna(0).astype(float)

        # Garage Yr Blt imputed with Year Built if missing
        if "Garage Yr Blt" in data.columns and "Year Built" in data.columns:
            data["Garage Yr Blt"] = data["Garage Yr Blt"].fillna(data["Year Built"]).astype(float)

        # 1. Total Living & Usable Space
        data["TotalSF"] = data["Gr Liv Area"] + data["Total Bsmt SF"]
        data["TotalBath"] = (
            data["Full Bath"]
            + 0.5 * data["Half Bath"]
            + data["Bsmt Full Bath"]
            + 0.5 * data["Bsmt Half Bath"]
        )

        # 2. Age metrics
        yr_sold = data["Yr Sold"] if "Yr Sold" in data.columns else 2010
        data["HouseAge"] = (yr_sold - data["Year Built"]).clip(lower=0)
        data["RemodAge"] = (yr_sold - data["Year Remod/Add"]).clip(lower=0)
        data["IsRemodeled"] = (data["Year Remod/Add"] != data["Year Built"]).astype(int)

        # 3. Outdoor amenities space
        porch_cols = ["Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"]
        data["TotalPorchSF"] = data[porch_cols].sum(axis=1)

        # 4. Binary indicator features
        data["HasPool"] = (data["Pool Area"] > 0).astype(int)
        data["HasGarage"] = (data["Garage Area"] > 0).astype(int)
        data["HasFireplace"] = (data["Fireplaces"] > 0).astype(int)
        data["HasBsmt"] = (data["Total Bsmt SF"] > 0).astype(int)

        # 5. Quality interactions
        data["OverallScore"] = data["Overall Qual"] * data["Overall Cond"]

        # 6. Ordinal encodings
        for col in ["Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC", "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC"]:
            if col in data.columns:
                data[col] = data[col].map(QUAL_MAP).fillna(0)

        if "Bsmt Exposure" in data.columns:
            data["Bsmt Exposure"] = data["Bsmt Exposure"].map(EXPOSURE_MAP).fillna(0)

        for col in ["BsmtFin Type 1", "BsmtFin Type 2"]:
            if col in data.columns:
                data[col] = data[col].map(FIN_TYPE_MAP).fillna(0)

        if "Garage Finish" in data.columns:
            data["Garage Finish"] = data["Garage Finish"].map(GARAGE_FIN_MAP).fillna(0)

        if "Central Air" in data.columns:
            data["Central Air"] = data["Central Air"].map(CENTRAL_AIR_MAP).fillna(0)

        return data

    def fit(self, train_df: pd.DataFrame):
        """Fit imputation statistics, encoder, and scaler strictly on training partition."""
        data = train_df.copy()

        # Remove atypical outliers if specified (only during training fit)
        if self.remove_train_outliers:
            mask = (data["Gr Liv Area"] > 4000) & (data["SalePrice"] < 300000)
            n_outliers = mask.sum()
            if n_outliers > 0:
                print(f"[AmesPreprocessor] Removing {n_outliers} De Cock atypical training outliers (>4000 sqft with <$300k).")
                data = data[~mask].copy()

        # Compute median LotFrontage per neighborhood
        if "Lot Frontage" in data.columns and "Neighborhood" in data.columns:
            self.lot_frontage_medians = (
                data.groupby("Neighborhood")["Lot Frontage"].median().to_dict()
            )
            self.lot_frontage_global_median = data["Lot Frontage"].median()

        # Engineer features
        data = self._engineer_features(data)

        # Compute modes for categorical features and medians for numerical features
        exclude_cols = ["Order", "PID", "SalePrice"]
        feature_cols = [c for c in data.columns if c not in exclude_cols]

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                self.numerical_medians[col] = float(data[col].median()) if not pd.isna(data[col].median()) else 0.0
            else:
                mode_val = data[col].mode()
                self.categorical_modes[col] = str(mode_val[0]) if len(mode_val) > 0 else "None"

        # Separate nominal categorical vs numerical
        self.continuous_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
        self.nominal_cols = [c for c in feature_cols if c not in self.continuous_cols]

        # Fit One-Hot Encoder on nominal features
        if len(self.nominal_cols) > 0:
            self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            # Impute missing before fit
            cat_data = data[self.nominal_cols].copy()
            for col in self.nominal_cols:
                cat_data[col] = cat_data[col].fillna(self.categorical_modes.get(col, "None")).astype(str)
            self.one_hot_encoder.fit(cat_data)
            ohe_names = self.one_hot_encoder.get_feature_names_out(self.nominal_cols).tolist()
        else:
            ohe_names = []

        # Fit RobustScaler on numerical features
        self.scaler = RobustScaler()
        num_data = data[self.continuous_cols].copy()
        for col in self.continuous_cols:
            num_data[col] = num_data[col].fillna(self.numerical_medians.get(col, 0))
        self.scaler.fit(num_data)

        self.feature_names = self.continuous_cols + ohe_names
        self.is_fitted = True
        print(f"[AmesPreprocessor] Fitted successfully. Total features after encoding: {len(self.feature_names)}")
        return self

    def transform(self, df: pd.DataFrame, is_train: bool = False):
        """Transform a partition (train, val, or test) using fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("AmesPreprocessor must be fitted before transform.")

        data = df.copy()

        # For train set during model training, drop outliers if specified
        if is_train and self.remove_train_outliers and "SalePrice" in data.columns:
            mask = (data["Gr Liv Area"] > 4000) & (data["SalePrice"] < 300000)
            data = data[~mask].copy()

        # Impute Lot Frontage by neighborhood median learned on train
        if "Lot Frontage" in data.columns and "Neighborhood" in data.columns:
            def impute_lf(row):
                if pd.isna(row["Lot Frontage"]):
                    return self.lot_frontage_medians.get(
                        row["Neighborhood"], self.lot_frontage_global_median
                    )
                return row["Lot Frontage"]
            data["Lot Frontage"] = data.apply(impute_lf, axis=1)

        # Feature engineering
        data = self._engineer_features(data)

        # Numerical imputation and scaling
        num_data = data[self.continuous_cols].copy()
        for col in self.continuous_cols:
            num_data[col] = num_data[col].fillna(self.numerical_medians.get(col, 0))
        scaled_num = self.scaler.transform(num_data)

        # Nominal categorical imputation and encoding
        if self.one_hot_encoder is not None and len(self.nominal_cols) > 0:
            cat_data = data[self.nominal_cols].copy()
            for col in self.nominal_cols:
                cat_data[col] = cat_data[col].fillna(self.categorical_modes.get(col, "None")).astype(str)
            encoded_cat = self.one_hot_encoder.transform(cat_data)
            X = np.hstack([scaled_num, encoded_cat])
        else:
            X = scaled_num

        # Target extraction
        if "SalePrice" in data.columns:
            y_raw = data["SalePrice"].values
            y_log = np.log1p(y_raw)
            return X, y_log, y_raw

        return X

    @staticmethod
    def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
        """Invert logarithmic transformation back to original dollar scale."""
        return np.expm1(y_log)


if __name__ == "__main__":
    from data_loader import get_dataset_partitions

    train_df, val_df, test_df, _ = get_dataset_partitions()
    prep = AmesPreprocessor()
    prep.fit(train_df)

    X_train, y_train_log, y_train_raw = prep.transform(train_df, is_train=True)
    X_val, y_val_log, y_val_raw = prep.transform(val_df, is_train=False)
    X_test, y_test_log, y_test_raw = prep.transform(test_df, is_train=False)

    print(f"X_train: {X_train.shape} | y_train: {len(y_train_log)}")
    print(f"X_val:   {X_val.shape}   | y_val:   {len(y_val_log)}")
    print(f"X_test:  {X_test.shape}  | y_test:  {len(y_test_log)}")

