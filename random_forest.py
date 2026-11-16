from sklearn.ensemble import RandomForestClassifier


def build(cfg: dict, seed: int):
    p = cfg["models"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
        n_jobs=2, random_state=seed,
    )
