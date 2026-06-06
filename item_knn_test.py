from pathlib import Path
import pickle

from item_knn_movielens100k import (
    print_recommendations,
    rating_metrics,
    read_items,
    read_ratings,
    topn_metrics,
)


DATA_DIR = Path("ml-100k")
MODEL_PATH = Path("item_knn_model.pkl")
TOP_N = 10
SAMPLE_USER = 1


def main():
    test_rows = read_ratings(DATA_DIR / "ua.test")
    titles = read_items(DATA_DIR / "u.item")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} does not exist. Run item_knn_train.py before item_knn_test.py."
        )

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    print("Loaded trained Item-based KNN model")
    print(f"- test rows from ua.test: {len(test_rows):,}")
    print("- ua.test is used only for final evaluation")

    final_rating = rating_metrics(model, test_rows)
    final_topn = topn_metrics(model, test_rows, top_n=TOP_N)

    print("\nFinal evaluation on ua.test")
    print(f"- RMSE: {final_rating['rmse']:.4f}")
    print(f"- MAE: {final_rating['mae']:.4f}")
    print(f"- Precision@{TOP_N}: {final_topn['precision_at_10']:.4f}")
    print(f"- Recall@{TOP_N}: {final_topn['recall_at_10']:.4f}")
    print(f"- Coverage: {final_topn['coverage']:.4f}")
    print(f"- Novelty: {final_topn['novelty']:.4f}")
    print(f"- Users evaluated: {final_topn['users_evaluated']}")

    print_recommendations(model, titles, SAMPLE_USER, top_n=TOP_N)


if __name__ == "__main__":
    main()
