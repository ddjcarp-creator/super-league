import pandas as pd

def rolling_form(tries, matches, player_id, window=5):
    df = tries.merge(matches, on="match_id")
    df = df[df["player_id"] == player_id].sort_values("date")

    df["rolling_tries"] = df["tries_scored"].rolling(window).mean()

    return df.tail(window)


def team_strength(teams):
    league_avg = teams["points_for"].mean()

    teams["attack_strength"] = teams["points_for"] / league_avg
    teams["defense_strength"] = teams["points_against"] / league_avg

    return teams


def defensive_profile(events, team):
    conceded = events[
        (events["event_type"] == "try") &
        (events["team"] != team)
    ].copy()

    conceded["zone"] = pd.cut(
        conceded["x"],
        bins=[0, 33, 66, 100],
        labels=["Left", "Middle", "Right"]
    )

    return conceded["zone"].value_counts(normalize=True)
