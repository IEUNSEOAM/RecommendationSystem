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
    """Make validation only from ua.base. ua.test remains untouched."""
    rng = random.Random(seed)
    by_user = defaultdict(list)
    for row in base_rows:
        by_user[row[0]].append(row)

    train_rows = []
    valid_rows = []
    for user_rows in by_user.values():
        shuffled = list(user_rows)
        rng.shuffle(shuffled)
        n_valid = max(1, int(len(shuffled) * valid_ratio)) if len(shuffled) >= 5 else 0
        valid_rows.extend(shuffled[:n_valid])
        train_rows.extend(shuffled[n_valid:])
    return train_rows, valid_rows


class ItemKNNRecommender:
    def __init__(self, k=40, shrinkage=25.0, min_candidate_popularity=5):
        self.k = k
        self.shrinkage = shrinkage
        self.min_candidate_popularity = min_candidate_popularity

    def fit(self, rows):
        self.users = sorted({u for u, _, _, _ in rows})
        self.items = sorted({i for _, i, _, _ in rows})
        self.user_to_idx = {u: idx for idx, u in enumerate(self.users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(self.items)}

        n_users = len(self.users)
        n_items = len(self.items)
        self.ratings = np.full((n_users, n_items), np.nan, dtype=np.float32)
        self.user_rated_items = defaultdict(list)
        self.item_popularity = defaultdict(int)

        for user_id, item_id, rating, _ in rows:
            uidx = self.user_to_idx[user_id]
            iidx = self.item_to_idx[item_id]
            self.ratings[uidx, iidx] = rating
            self.user_rated_items[user_id].append((item_id, rating))
            self.item_popularity[item_id] += 1

        self.global_mean = float(np.nanmean(self.ratings))
        self.user_means = np.nanmean(self.ratings, axis=1)
        self.item_means = np.nanmean(self.ratings, axis=0)
        self.user_means = np.where(np.isnan(self.user_means), self.global_mean, self.user_means)
        self.item_means = np.where(np.isnan(self.item_means), self.global_mean, self.item_means)

        mask = ~np.isnan(self.ratings)
        centered = np.where(mask, self.ratings - self.user_means[:, None], 0.0)

        numerator = centered.T @ centered
        norms = np.sqrt(np.sum(centered * centered, axis=0))
        denominator = norms[:, None] * norms[None, :]
        similarity = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 0,
        )

        co_counts = mask.astype(np.float32).T @ mask.astype(np.float32)
        significance = co_counts / (co_counts + self.shrinkage)
        self.similarity = similarity * significance
        np.fill_diagonal(self.similarity, 0.0)

        self.neighbors = {}
        for item_idx in range(n_items):
            sims = self.similarity[item_idx]
            order = np.argsort(np.abs(sims))[::-1]
            order = [idx for idx in order if sims[idx] != 0.0][: self.k]
            self.neighbors[item_idx] = order
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_to_idx:
            return self._clip(self.item_mean(item_id))
        if item_id not in self.item_to_idx:
            return self._clip(self.user_mean(user_id))

        uidx = self.user_to_idx[user_id]
        target_idx = self.item_to_idx[item_id]
        baseline = self.user_means[uidx]
        numerator = 0.0
        denominator = 0.0

        for neighbor_idx in self.neighbors[target_idx]:
            rating = self.ratings[uidx, neighbor_idx]
            if np.isnan(rating):
                continue
            sim = float(self.similarity[target_idx, neighbor_idx])
            numerator += sim * (float(rating) - baseline)
            denominator += abs(sim)

        if denominator == 0.0:
            item_mean = self.item_means[target_idx]
            return self._clip(0.6 * baseline + 0.4 * item_mean)
        return self._clip(baseline + numerator / denominator)

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_to_idx:
            return self.popular_items(top_n)

        seen = {item_id for item_id, _ in self.user_rated_items[user_id]}
        candidates = [
            item_id
            for item_id in self.items
            if item_id not in seen and self.item_popularity.get(item_id, 0) >= self.min_candidate_popularity
        ]

        scored = [(item_id, self.predict(user_id, item_id)) for item_id in candidates]
        scored.sort(key=lambda x: (x[1], self.item_popularity.get(x[0], 0)), reverse=True)
        return scored[:top_n]

    def popular_items(self, top_n):
        return [
            item_id
            for item_id, _ in sorted(
                self.item_popularity.items(),
                key=lambda x: (x[1], self.item_mean(x[0])),
                reverse=True,
            )[:top_n]
        ]

    def user_mean(self, user_id):
        if user_id not in self.user_to_idx:
            return self.global_mean
        return float(self.user_means[self.user_to_idx[user_id]])

    def item_mean(self, item_id):
        if item_id not in self.item_to_idx:
            return self.global_mean
        return float(self.item_means[self.item_to_idx[item_id]])

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
        "precision_at_10": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall_at_10": sum(recalls) / len(recalls) if recalls else 0.0,
        "coverage": len(all_recommended) / len(model.items),
        "novelty": sum(novelty_scores) / len(novelty_scores),
        "users_evaluated": len(eval_users),
    }


def tune_k(train_rows, valid_rows, k_values, shrinkage, min_candidate_popularity):
    results = []
    for k in k_values:
        model = ItemKNNRecommender(
            k=k,
            shrinkage=shrinkage,
            min_candidate_popularity=min_candidate_popularity,
        ).fit(train_rows)
        metrics = rating_metrics(model, valid_rows)
        results.append((k, metrics["rmse"], metrics["mae"]))
    return min(results, key=lambda x: x[1]), results


def print_recommendations(model, titles, user_id, top_n=10):
    print(f"\nSample recommendations for user {user_id}")
    for rank, (item_id, score) in enumerate(model.recommend(user_id, top_n=top_n), start=1):
        title = titles.get(item_id, f"movieId={item_id}")
        print(f"{rank:2d}. {title} | predicted_rating={score:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Item-based KNN CF for MovieLens 100k ua.base/ua.test")
    parser.add_argument("--data-dir", default="ml-100k")
    parser.add_argument("--k-values", default="20,40,80")
    parser.add_argument("--shrinkage", type=float, default=25.0)
    parser.add_argument("--min-candidate-popularity", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sample-user", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_rows = read_ratings(data_dir / "ua.base")
    test_rows = read_ratings(data_dir / "ua.test")
    titles = read_items(data_dir / "u.item")

    inner_train_rows, valid_rows = split_validation_from_base(base_rows)
    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]

    print("Data split")
    print(f"- ua.base rows: {len(base_rows):,}")
    print(f"- inner train rows from ua.base: {len(inner_train_rows):,}")
    print(f"- validation rows from ua.base only: {len(valid_rows):,}")
    print(f"- ua.test rows, final evaluation only: {len(test_rows):,}")

    best, validation_results = tune_k(
        inner_train_rows,
        valid_rows,
        k_values,
        args.shrinkage,
        args.min_candidate_popularity,
    )

    print("\nValidation tuning on ua.base only")
    for k, rmse, mae in validation_results:
        print(f"- K={k:>3}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    best_k = best[0]
    print(f"Selected K={best_k} by validation RMSE")

    final_model = ItemKNNRecommender(
        k=best_k,
        shrinkage=args.shrinkage,
        min_candidate_popularity=args.min_candidate_popularity,
    ).fit(base_rows)

    final_rating = rating_metrics(final_model, test_rows)
    final_topn = topn_metrics(final_model, test_rows, top_n=args.top_n)

    print("\nFinal evaluation on ua.test")
    print(f"- RMSE: {final_rating['rmse']:.4f}")
    print(f"- MAE: {final_rating['mae']:.4f}")
    print(f"- Precision@{args.top_n}: {final_topn['precision_at_10']:.4f}")
    print(f"- Recall@{args.top_n}: {final_topn['recall_at_10']:.4f}")
    print(f"- Coverage: {final_topn['coverage']:.4f}")
    print(f"- Novelty: {final_topn['novelty']:.4f}")
    print(f"- Users evaluated: {final_topn['users_evaluated']}")

    print_recommendations(final_model, titles, args.sample_user, top_n=args.top_n)


if __name__ == "__main__":
    main()