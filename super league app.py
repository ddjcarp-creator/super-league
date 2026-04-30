import streamlit as st
import pandas as pd

from features import rolling_form, team_strength, defensive_profile
from models import matchup_model

st.set_page_config(layout="wide")

@st.cache_data
def load():
    return (
        pd.read_csv("data/teams.csv"),
        pd.read_csv("data/players.csv"),
        pd.read_csv("data/matches.csv"),
        pd.read_csv("data/tries.csv"),
        pd.read_csv("data/events.csv"),
    )

teams, players, matches, tries, events = load()

teams = team_strength(teams)

st.title("🏉 Super League Pro Analytics")

tab1, tab2, tab3 = st.tabs([
    "Team Analysis",
    "Player Analysis",
    "Match Predictions"
])

# ---------------- TEAM ----------------
with tab1:
    team = st.selectbox("Team", teams["team_name"])
    row = teams[teams["team_name"] == team]

    st.subheader("Strength Metrics")
    st.metric("Attack Strength", f"{row['attack_strength'].values[0]:.2f}")
    st.metric("Defense Strength", f"{row['defense_strength'].values[0]:.2f}")

    st.subheader("Defensive Zones")
    zones = defensive_profile(events, team)
    st.bar_chart(zones)

# ---------------- PLAYER ----------------
with tab2:
    player_name = st.selectbox("Player", players["name"])
    player = players[players["name"] == player_name].iloc[0]

    form = rolling_form(tries, matches, player["player_id"])

    st.subheader("Recent Form")
    st.line_chart(form.set_index("date")["rolling_tries"])

# ---------------- MATCHUPS ----------------
with tab3:
    player_name = st.selectbox("Select Player", players["name"], key=10)
    opponent = st.selectbox("Opponent", teams["team_name"], key=11)

    player = players[players["name"] == player_name].iloc[0]
    team_row = teams[teams["team_id"] == player["team_id"]].iloc[0]
    opp_row = teams[teams["team_name"] == opponent].iloc[0]

    prob = matchup_model(player, team_row, opp_row)

    st.metric("Try Probability", f"{prob:.2%}")
