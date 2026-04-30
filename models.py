from sklearn.ensemble import GradientBoostingClassifier

def train_model(df):
    features = [
        "player_rate",
        "form",
        "attack_strength",
        "defense_strength",
        "is_home"
    ]

    X = df[features]
    y = df["target"]

    model = GradientBoostingClassifier()
    model.fit(X, y)

    return model, features
