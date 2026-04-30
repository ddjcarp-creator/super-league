import streamlit as st
import pandas as pd
import pickle

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    return (
        pd.read_csv("data/matches.csv"),
        pd.read_csv("data/tries.csv"),
        pd.read_csv("data/players.csv"),
        pd.read_csv("data/teams.csv"),
    )

@st.cache_resource
def load_model():
    with open("models/try_model.pkl", "rb") as f:
        return pickle.load(f)

matches, tries, players, teams = load_data()
model, feature_cols = load_model()

st.title("🏉 Super League Pro Analytics Engine")

# ---------------- PLAYER SELECTION ----------------
player_name = st.selectbox("Select Player", players["name"])
opponent = st.selectbox("Opponent", teams["team_name"])
is_home = st.toggle("Home Game")

player = players[players["name"] == player_name].iloc[0]
team = teams[teams["team_id"] == player["team_id"]].iloc[0]
opp = teams[teams["team_name"] == opponent].iloc[0]

# ---------------- FEATURE BUILD ----------------
player_rate = player["tries"] / player["appearances"]

recent = tries[tries["player_id"] == player["player_id"]].tail(5)
form = recent["tries_scored"].mean() if not recent.empty else 0

league_avg = teams["points_for"].mean()

attack = team["points_for"] / league_avg
defense = opp["points_against"] / league_avg

X = pd.DataFrame([{
    "player_rate": player_rate,
    "form": form,
    "attack_strength": attack,
    "defense_strength": defense,
    "is_home": int(is_home)
}])

# ---------------- PREDICTION ----------------
prob = model.predict_proba(X)[0][1]

st.metric("Try Probability", f"{prob:.2%}")

# ---------------- INSIGHT ----------------
if prob > 0.6:
    st.success("🔥 High likelihood of scoring")
elif prob > 0.3:
    st.warning("⚠️ Moderate chance")
else:
    st.error("❌ Low probability")

# ---------------- TEAM VIEW ----------------
st.subheader("Team Comparison")

col1, col2 = st.columns(2)

with col1:
    st.write("Your Team")
    st.write(team)

with col2:
    st.write("Opponent")
    st.write(opp)
