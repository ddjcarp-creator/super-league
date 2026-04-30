import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# -----------------------
# LOAD DATA (LOCAL FILES)
# -----------------------
@st.cache_data
def load_data():
    teams = pd.read_csv("data/teams.csv")
    players = pd.read_csv("data/players.csv")
    matches = pd.read_csv("data/matches.csv")
    tries = pd.read_csv("data/tries.csv")
    events = pd.read_csv("data/events.csv")
    return teams, players, matches, tries, events

teams, players, matches, tries, events = load_data()

# -----------------------
# FUNCTIONS
# -----------------------

def player_form(player_id):
    merged = tries.merge(matches, on="match_id")
    df = merged[merged["player_id"] == player_id]
    df = df.sort_values(by="date", ascending=False)
    last5 = df.head(5)
    return last5["tries_scored"].mean(), last5


def try_probability(player_id, team_id):
    player = players[players["player_id"] == player_id]
    player_rate = player["tries"].values[0] / player["appearances"].values[0]

    team_attack = teams[teams["team_id"] == team_id]["points_for"].values[0]
    team_attack_norm = team_attack / teams["points_for"].max()

    return min(0.6 * player_rate + 0.4 * team_attack_norm, 1.0)


def home_away(team):
    home = matches[matches["home_team"] == team]
    away = matches[matches["away_team"] == team]

    home_wr = (home["home_score"] > home["away_score"]).mean()
    away_wr = (away["away_score"] > away["home_score"]).mean()

    return home_wr, away_wr


def defensive_zones(team):
    conceded = events[
        (events["event_type"] == "try") &
        (events["team"] != team)
    ].copy()

    def zone(x):
        if x < 33:
            return "Left"
        elif x < 66:
            return "Middle"
        else:
            return "Right"

    conceded["zone"] = conceded["x"].apply(zone)
    return conceded["zone"].value_counts()


def matchup_prediction(player_id, team, opponent):
    base = try_probability(player_id, team_id)

    weakness = defensive_zones(opponent)
    if not weakness.empty:
        weak_side = weakness.idxmax()
        position = players[players["player_id"] == player_id]["position"].values[0]

        if weak_side == "Left" and position == "Right Wing":
            base *= 1.2
        elif weak_side == "Right" and position == "Left Wing":
            base *= 1.2

    return min(base, 1.0)


# -----------------------
# UI
# -----------------------

st.title("🏉 Super League Advanced Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Team",
    "Player",
    "Matchups",
    "Advanced"
])

# -----------------------
# TEAM TAB
# -----------------------
with tab1:
    team_name = st.selectbox("Select Team", teams["team_name"])
    team_data = teams[teams["team_name"] == team_name]

    st.subheader("Team Stats")
    st.dataframe(team_data)

    home_wr, away_wr = home_away(team_name)

    st.metric("Home Win Rate", f"{home_wr:.2%}")
    st.metric("Away Win Rate", f"{away_wr:.2%}")

# -----------------------
# PLAYER TAB
# -----------------------
with tab2:
    player_name = st.selectbox("Select Player", players["name"])
    player_id = players[players["name"] == player_name]["player_id"].values[0]
    team_id = players[players["name"] == player_name]["team_id"].values[0]

    avg, last5 = player_form(player_id)

    st.subheader("Form (Last 5 Games)")
    st.write(f"Average tries: {avg:.2f}")
    st.dataframe(last5)

    prob = try_probability(player_id, team_id)
    st.metric("Try Probability", f"{prob:.2%}")

# -----------------------
# MATCHUPS TAB
# -----------------------
with tab3:
    team1 = st.selectbox("Team 1", teams["team_name"], key=1)
    team2 = st.selectbox("Team 2", teams["team_name"], key=2)

    h2h = matches[
        ((matches["home_team"] == team1) & (matches["away_team"] == team2)) |
        ((matches["home_team"] == team2) & (matches["away_team"] == team1))
    ]

    st.subheader("Head-to-Head")
    st.dataframe(h2h)

# -----------------------
# ADVANCED TAB
# -----------------------
with tab4:
    team = st.selectbox("Team (Defense)", teams["team_name"], key=3)

    st.subheader("Defensive Weakness")
    zones = defensive_zones(team)
    st.bar_chart(zones)

    st.subheader("Matchup Try Prediction")

    player = st.selectbox("Player", players["name"], key=4)
    opponent = st.selectbox("Opponent", teams["team_name"], key=5)

    player_id = players[players["name"] == player]["player_id"].values[0]
    team_id = players[players["name"] == player]["team_id"].values[0]

    pred = matchup_prediction(player_id, team, opponent)

    st.metric("Predicted Try Chance", f"{pred:.2%}")
