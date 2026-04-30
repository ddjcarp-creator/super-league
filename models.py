import numpy as np

def poisson_try_prob(player_rate, team_attack, opp_defense):
    lam = player_rate * team_attack / opp_defense

    # Probability of scoring at least 1 try
    prob = 1 - np.exp(-lam)

    return min(prob, 1.0)


def matchup_model(player, team_row, opp_row):
    player_rate = player["tries"] / player["appearances"]

    team_attack = team_row["attack_strength"]
    opp_defense = opp_row["defense_strength"]

    return poisson_try_prob(player_rate, team_attack, opp_defense)
