from pathlib import Path
import pickle

from item_knn_movielens100k import (
    ItemKNNRecommender,
    read_ratings,
    split_validation_from_base,
    tune_k,
)


DATA_DIR = Path("ml-100k")
MODEL_PATH = Path("item_knn_model.pkl")
K_VALUES = [20, 40, 80]
SHRINKAGE = 25.0
MIN_CANDIDATE_POPULARITY = 5


def main():
    base_rows = read_ratings(DATA_DIR / "ua.base")
    inner_train_rows, valid_rows = split_validation_from_base(base_rows)

    print("Train / validation split")
    print(f"- ua.base rows: {len(base_rows):,}")
    print(f"- inner train rows from ua.base: {len(inner_train_rows):,}")
    print(f"- validation rows from ua.base only: {len(valid_rows):,}")

    best, validation_results = tune_k(
        inner_train_rows,
        valid_rows,
        K_VALUES,
        SHRINKAGE,
        MIN_CANDIDATE_POPULARITY,
    )

    print("\nValidation tuning on ua.base only")
    for k, rmse, mae in validation_results:
        print(f"- K={k:>3}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    best_k = best[0]
    print(f"\nSelected K={best_k} by validation RMSE")

    model = ItemKNNRecommender(
        k=best_k,
        shrinkage=SHRINKAGE,
        min_candidate_popularity=MIN_CANDIDATE_POPULARITY,
    ).fit(base_rows)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    print(f"\nSaved trained model to {MODEL_PATH}")
    print("ua.test was not used during training.")


if __name__ == "__main__":
    main()
