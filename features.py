import pandas as pd
import numpy as np

def build_features(matches, tries, players, teams):
    df = tries.merge(matches, on="match_id")
    df = df.merge(players, on="player_id")
    df = df.merge(teams, on="team_id")

    # Player scoring rate
    df["player_rate"] = df["tries"] / df["appearances"]

    # Rolling form
    df = df.sort_values("date")
    df["form"] = df.groupby("player_id")["tries_scored"]\
                   .rolling(5).mean().reset_index(0, drop=True)

    # Team strength
    league_avg = teams["points_for"].mean()
    df["attack_strength"] = df["points_for"] / league_avg
    df["defense_strength"] = df["points_against"] / league_avg

    # Home advantage
    df["is_home"] = (df["team_name"] == df["home_team"]).astype(int)

    # Target: scored try or not
    df["target"] = (df["tries_scored"] > 0).astype(int)

    return df.dropna()
