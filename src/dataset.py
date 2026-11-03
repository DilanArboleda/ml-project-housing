"""
Ames Housing Dataset unified module.
Workshop No. 1 â€” Data Preparation & Supervised Learning Baselines.

Group 1:
- Chavez Porras Miguel Alejandro (20212020033)
- Guerrero OrdeÃ±ez Luisa Fernanda (20212020099)
- Arboleda Zambrano Dilan Stive (20212020105)
"""

from pathlib import Path
from data_loader import get_dataset_partitions, set_seed, RANDOM_SEED
from features import AmesPreprocessor

def load_and_preprocess_data(seed: int = RANDOM_SEED):
    """
    Convenience pipeline to load raw data, apply train/val/test splits,
    fit preprocessor strictly on train set, and return feature matrices.
    """
    set_seed(seed)
    train_df, val_df, test_df, splits = get_dataset_partitions()

    preprocessor = AmesPreprocessor(remove_train_outliers=True)
    preprocessor.fit(train_df)

    X_train, y_train_log, y_train = preprocessor.transform(train_df, is_train=True)
    X_val, y_val_log, y_val = preprocessor.transform(val_df, is_train=False)
    X_test, y_test_log, y_test = preprocessor.transform(test_df, is_train=False)

    return {
        "X_train": X_train,
        "y_train_log": y_train_log,
        "y_train": y_train,
        "X_val": X_val,
        "y_val_log": y_val_log,
        "y_val": y_val,
        "X_test": X_test,
        "y_test_log": y_test_log,
        "y_test": y_test,
        "feature_names": preprocessor.feature_names,
        "preprocessor": preprocessor,
        "splits": splits
    }

if __name__ == "__main__":
    bundle = load_and_preprocess_data()
    print("Bundle loaded successfully.")
    print(f"Features: {len(bundle['feature_names'])}")

