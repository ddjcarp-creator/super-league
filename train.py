import pandas as pd
import pickle

from features import build_features
from model import train_model

matches = pd.read_csv("data/matches.csv")
tries = pd.read_csv("data/tries.csv")
players = pd.read_csv("data/players.csv")
teams = pd.read_csv("data/teams.csv")

df = build_features(matches, tries, players, teams)

model, features = train_model(df)

with open("models/try_model.pkl", "wb") as f:
    pickle.dump((model, features), f)

print("Model trained and saved.")
