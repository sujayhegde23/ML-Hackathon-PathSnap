"""The models compared. All expose fit / predict_proba so they are trained and scored
identically. Scaling sits inside a Pipeline, so it is learned from the training fold only."""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FAMILY = {
    "LogisticRegression": "Linear / probabilistic",
    "GaussianNB": "Probabilistic (generative)",
    "KNN": "Instance-based",
    "DecisionTree": "Tree",
    "RandomForest": "Ensemble (bagging)",
    "HistGradientBoosting": "Boosting",
    "LightGBM": "Gradient boosting (leaf-wise)",
    "MLP": "Neural network",
}

CONFIG_TEXT = {
    "LogisticRegression": "C=1.0, lbfgs, standardised",
    "GaussianNB": "var_smoothing=1e-9, standardised",
    "KNN": "k=25, distance-weighted, standardised, <=80k fit rows",
    "DecisionTree": "max_depth=12, min_samples_leaf=50",
    "RandomForest": "100 trees, max_depth=18, min_samples_leaf=20, 50% bootstrap",
    "HistGradientBoosting": "200 iters, lr=0.1, 31 leaves",
    "LightGBM": "400 trees, lr=0.05, 63 leaves, 80% row/90% col subsample",
    "MLP": "64-32 ReLU, Adam, batch 1024, early stopping",
}


def make_model(name: str, seed: int = 42):
    if name == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    if name == "GaussianNB":
        return make_pipeline(StandardScaler(), GaussianNB())
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=20, max_samples=0.5,
                                      n_jobs=-1, random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                              early_stopping=False, random_state=seed)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.9, n_jobs=-1, random_state=seed, verbose=-1)
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024,
                                                             max_iter=30, early_stopping=True, n_iter_no_change=4,
                                                             random_state=seed))
    raise KeyError(f"unknown model {name!r}")
