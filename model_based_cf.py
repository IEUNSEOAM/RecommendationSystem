import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_ratings(path):
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            user_id, item_id, rating, timestamp = line.strip().split("\t")
            rows.append((int(user_id), int(item_id), float(rating), int(timestamp)))
    return rows


def read_items(path):
    titles = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            titles[int(parts[0])] = parts[1]
    return titles


def split_validation_from_base(base_rows, valid_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_user = defaultdict(list)

    for row in base_rows:
        by_user[row[0]].append(row)

    train_rows = []
    valid_rows = []

    for user_rows in by_user.values():
        shuffled = list(user_rows)
        rng.shuffle(shuffled)

        if len(shuffled) >= 5:
            n_valid = max(1, int(len(shuffled) * valid_ratio))
        else:
            n_valid = 0

        valid_rows.extend(shuffled[:n_valid])
        train_rows.extend(shuffled[n_valid:])

    return train_rows, valid_rows


class MFRecommender:
    def __init__(
        self,
        n_factors=20,
        lr=0.005,
        reg=0.02,
        n_epochs=20,
        min_candidate_popularity=5,
        seed=42,
    ):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.min_candidate_popularity = min_candidate_popularity
        self.seed = seed

    def fit(self, rows):
        self.users = sorted({u for u, _, _, _ in rows})
        self.items = sorted({i for _, i, _, _ in rows})
        self.user_to_idx = {u: idx for idx, u in enumerate(self.users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(self.items)}

        n_users = len(self.users)
        n_items = len(self.items)

        self.user_rated_items = defaultdict(list)
        self.item_popularity = defaultdict(int)

        rating_sum = 0.0
        rating_count = 0

        for user_id, item_id, rating, _ in rows:
            self.user_rated_items[user_id].append((item_id, rating))
            self.item_popularity[item_id] += 1
            rating_sum += rating
            rating_count += 1

        self.global_mean = rating_sum / rating_count if rating_count > 0 else 3.0

        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, 0.01, (n_users, self.n_factors)).astype(np.float32)
        self.V = rng.normal(0, 0.01, (n_items, self.n_factors)).astype(np.float32)

        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        train_list = [
            (self.user_to_idx[u], self.item_to_idx[i], r)
            for u, i, r, _ in rows
        ]

        rng_shuffle = random.Random(self.seed)

        for epoch in range(self.n_epochs):
            rng_shuffle.shuffle(train_list)
            squared_error_sum = 0.0

            for uidx, iidx, rating in train_list:
                pred = (
                    self.global_mean
                    + self.b_u[uidx]
                    + self.b_i[iidx]
                    + float(self.U[uidx] @ self.V[iidx])
                )
                err = rating - pred
                squared_error_sum += err ** 2

                u_old = self.U[uidx].copy()

                self.U[uidx] += self.lr * (err * self.V[iidx] - self.reg * self.U[uidx])
                self.V[iidx] += self.lr * (err * u_old - self.reg * self.V[iidx])
                self.b_u[uidx] += self.lr * (err - self.reg * self.b_u[uidx])
                self.b_i[iidx] += self.lr * (err - self.reg * self.b_i[iidx])

            if (epoch + 1) % 5 == 0:
                train_rmse = math.sqrt(squared_error_sum / len(train_list))
                print(f"  epoch {epoch + 1:>2}/{self.n_epochs} train RMSE={train_rmse:.4f}")

        return self

    def predict(self, user_id, item_id):
        has_user = user_id in self.user_to_idx
        has_item = item_id in self.item_to_idx

        if not has_user and not has_item:
            return self._clip(self.global_mean)

        if not has_user:
            iidx = self.item_to_idx[item_id]
            return self._clip(self.global_mean + self.b_i[iidx])

        if not has_item:
            uidx = self.user_to_idx[user_id]
            return self._clip(self.global_mean + self.b_u[uidx])

        uidx = self.user_to_idx[user_id]
        iidx = self.item_to_idx[item_id]

        pred = (
            self.global_mean
            + self.b_u[uidx]
            + self.b_i[iidx]
            + float(self.U[uidx] @ self.V[iidx])
        )
        return self._clip(pred)

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_to_idx:
            return self.popular_items(top_n)

        seen = {item_id for item_id, _ in self.user_rated_items[user_id]}

        candidates = [
            item_id
            for item_id in self.items
            if item_id not in seen
            and self.item_popularity.get(item_id, 0) >= self.min_candidate_popularity
        ]

        scored = [(item_id, self.predict(user_id, item_id)) for item_id in candidates]
        scored.sort(key=lambda x: (x[1], self.item_popularity.get(x[0], 0)), reverse=True)

        return scored[:top_n]

    def popular_items(self, top_n=10):
        sorted_items = sorted(
            self.item_popularity.keys(),
            key=lambda item_id: (self.item_popularity[item_id], self.item_mean(item_id)),
            reverse=True,
        )
        return [(item_id, self.item_mean(item_id)) for item_id in sorted_items[:top_n]]

    def item_mean(self, item_id):
        if item_id not in self.item_to_idx:
            return self.global_mean
        iidx = self.item_to_idx[item_id]
        return self._clip(self.global_mean + self.b_i[iidx])

    @staticmethod
    def _clip(value):
        return min(5.0, max(1.0, float(value)))


def rating_metrics(model, rows):
    errors = []
    abs_errors = []

    for user_id, item_id, rating, _ in rows:
        pred = model.predict(user_id, item_id)
        errors.append((rating - pred) ** 2)
        abs_errors.append(abs(rating - pred))

    return {
        "rmse": math.sqrt(sum(errors) / len(errors)),
        "mae": sum(abs_errors) / len(abs_errors),
    }


def topn_metrics(model, eval_rows, top_n=10, relevant_threshold=4.0):
    eval_users = sorted({u for u, _, _, _ in eval_rows if u in model.user_to_idx})

    relevant_by_user = defaultdict(set)
    for user_id, item_id, rating, _ in eval_rows:
        if rating >= relevant_threshold:
            relevant_by_user[user_id].add(item_id)

    all_recommended = set()
    novelty_scores = []
    precisions = []
    recalls = []

    total_interactions = sum(model.item_popularity.values())

    for user_id in eval_users:
        recs = model.recommend(user_id, top_n=top_n)
        rec_items = [item_id for item_id, _ in recs]

        all_recommended.update(rec_items)

        for item_id in rec_items:
            popularity = model.item_popularity.get(item_id, 1)
            novelty_scores.append(-math.log2(popularity / total_interactions))

        relevant = relevant_by_user.get(user_id, set())
        if relevant:
            hits = len(set(rec_items) & relevant)
            precisions.append(hits / top_n)
            recalls.append(hits / len(relevant))

    return {
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "coverage": len(all_recommended) / len(model.items),
        "novelty": sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0,
        "users_evaluated": len(eval_users),
    }


def tune_factors(train_rows, valid_rows, factor_values, lr, reg, n_epochs, min_candidate_popularity):
    results = []

    for n_factors in factor_values:
        print(f"\nTraining model with n_factors={n_factors}")

        model = MFRecommender(
            n_factors=n_factors,
            lr=lr,
            reg=reg,
            n_epochs=n_epochs,
            min_candidate_popularity=min_candidate_popularity,
        ).fit(train_rows)

        metrics = rating_metrics(model, valid_rows)
        results.append((n_factors, metrics["rmse"], metrics["mae"]))

    best = min(results, key=lambda x: x[1])
    return best, results


def print_recommendations(model, titles, user_id, top_n=10):
    print(f"\nSample recommendations for user {user_id}")

    for rank, (item_id, score) in enumerate(model.recommend(user_id, top_n=top_n), start=1):
        title = titles.get(item_id, f"movieId={item_id}")
        print(f"{rank:2d}. {title} | predicted_rating={score:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Model-based CF using Matrix Factorization with SGD"
    )
    parser.add_argument("--data-dir", default="ml-100k")
    parser.add_argument("--factor-values", default="10,20,50")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--reg", type=float, default=0.02)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--min-candidate-popularity", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sample-user", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    base_rows = read_ratings(data_dir / "ua.base")
    test_rows = read_ratings(data_dir / "ua.test")
    titles = read_items(data_dir / "u.item")

    inner_train_rows, valid_rows = split_validation_from_base(base_rows)
    factor_values = [int(f.strip()) for f in args.factor_values.split(",") if f.strip()]

    print("Data split")
    print(f"- ua.base rows                      : {len(base_rows):,}")
    print(f"- inner train rows from ua.base     : {len(inner_train_rows):,}")
    print(f"- validation rows from ua.base only : {len(valid_rows):,}")
    print(f"- ua.test rows, final eval only     : {len(test_rows):,}")

    best, validation_results = tune_factors(
        inner_train_rows,
        valid_rows,
        factor_values,
        args.lr,
        args.reg,
        args.n_epochs,
        args.min_candidate_popularity,
    )

    print("\nValidation tuning on ua.base only")
    for n_factors, rmse, mae in validation_results:
        print(f"- factors={n_factors:>3}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    best_factors = best[0]
    print(f"Selected n_factors={best_factors} by validation RMSE")

    print("\nFinal training on full ua.base")
    final_model = MFRecommender(
        n_factors=best_factors,
        lr=args.lr,
        reg=args.reg,
        n_epochs=args.n_epochs,
        min_candidate_popularity=args.min_candidate_popularity,
    ).fit(base_rows)

    final_rating = rating_metrics(final_model, test_rows)
    final_topn = topn_metrics(final_model, test_rows, top_n=args.top_n)

    print("\nFinal evaluation on ua.test")
    print(f"- RMSE          : {final_rating['rmse']:.4f}")
    print(f"- MAE           : {final_rating['mae']:.4f}")
    print(f"- Precision@{args.top_n:<2} : {final_topn['precision']:.4f}")
    print(f"- Recall@{args.top_n:<2}    : {final_topn['recall']:.4f}")
    print(f"- Coverage      : {final_topn['coverage']:.4f}")
    print(f"- Novelty       : {final_topn['novelty']:.4f}")
    print(f"- Users eval'd  : {final_topn['users_evaluated']}")

    print_recommendations(final_model, titles, args.sample_user, top_n=args.top_n)


if __name__ == "__main__":
    main()
