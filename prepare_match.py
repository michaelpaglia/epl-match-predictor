"""
Soccer match simulation using Bayesian modeling to predict match outcomes based on team and player statistics.
Uses PyMC for probabilistic modeling and FBref for soccer statistics data.
"""

# https://soccerdata.readthedocs.io/en/latest/datasources/FBref.html
import soccerdata as sd
import pymc as pm
import numpy as np
# Custom files:
import prepare_rosters as rosters
import football_web_pages_endpoint as fb_wp_endpoint
import config
import trace_analysis

# Load team rosters and current league table (as of this project's submission: 11/18/24, should be GW11!)
EPL_roster = rosters.process_dict(config.FILE)
league_table = fb_wp_endpoint.fetch_team_stats()
league = "ENG-Premier League"
season = 2024


def get_team_and_player_stats(p_league: str, p_season: int, p_team_name: str, p_team_roster: list) -> dict:
    """
    Retrieves comprehensive statistics for a team and its players from FBref.

    Args:
        p_league (str): League (e.g., 'ENG-Premier League')
        p_season (int): Season year (e.g., 2024)
        p_team_name (str): Name of the team (e.g. "Liverpool" (up the Reds))
        p_team_roster (list): Team roster

    Returns:
        dict: Dictionary containing statistical categories for both team and players,
              including standard stats, keeper stats, shooting, passing, defense, etc.

              This will be used ultimately to derive a "strength" coefficient
    """
    fbref = sd.FBref(leagues=p_league, seasons=p_season)

    # From the FBRef API, these are all the categories we will be using
    stat_types = ['standard', 'keeper', 'keeper_adv', 'shooting', 'passing', 'passing_types', 'goal_shot_creation',
                  'defense', 'possession', 'playing_time', 'misc']

    stats = {}
    # Get stats for each category defined
    for stat_type in stat_types:
        # Gets all player stats
        player_category_stats = fbref.read_player_season_stats(stat_type=stat_type)
        # Gets specific team stats
        team_category_stats = player_category_stats.xs(p_team_name, level='team')
        # Maps player to team
        player_stats_on_this_team = team_category_stats[
            team_category_stats.index.get_level_values('player').isin(p_team_roster)]
        stats[stat_type] = player_stats_on_this_team

    # Add standard stats for the team
    team_stats = fbref.read_team_season_stats(stat_type='standard')
    stats['team'] = team_stats.xs(p_team_name, level="team")

    return stats


def get_team_goals_conceded():
    """
    Creates a dictionary of goals conceded by each team in the current season.

    Returns:
        dict: {team: goals_conceded}
    """
    return {team['name']: team['all-matches']['against'] for team in league_table['league-table']['teams']}


def get_team_matches_played():
    """
    Creates a dictionary of matches played by each team in the current season.

    Returns:
        dict: {team: matches_played}
    """
    return {team["name"]: team["all-matches"]["played"] for team in league_table['league-table']['teams']}


def get_league_max_values(p_league: str, p_season: int) -> dict:
    """
    Retrieves maximum values for all statistics across all teams in the league.
    The main usage here is the normalization for each team relative to others

    Args:
        p_league (str): League
        p_season (int): Season year

    Returns:
        dict: Maximum values for all statistical categories.
    """
    fbref = sd.FBref(leagues=p_league, seasons=p_season)

    # Use STAT CATEGORIES derived from FBREF API to get the stat types in
    stat_types = {stat_path[0] for category in config.STAT_CATEGORIES.values()
                  for stat_path in category.values()}

    # Get all relevant stats for each stat type
    stats = {
        stat_type: fbref.read_team_season_stats(stat_type=stat_type)
        for stat_type in stat_types
    }

    max_values = {}
    # Calculate maximum values for each statistic
    for category in config.STAT_CATEGORIES.values():
        for stat_name, stat_path in category.items():
            # Get type from the tuple
            stat_type = stat_path[0]
            # Certain stats have a tuple of three (type, ___, ___) that requires specific unpacking
            # e.g. 'goals': ('standard', 'Performance', 'Gls') -> 'standard'
            if len(stat_path) == 3:
                value = stats[stat_type][(stat_path[1], stat_path[2])].max()
            else:  # len(stat_path) == 2
                value = stats[stat_type][stat_path[1]].max()

            # Convert the challenges and possession to decimals, as possession in particular is usually a + b = 1
            if stat_name in ['challenge_success', 'possession']:
                value = value / 100

            max_values[stat_name] = value

    # Add goals conceded using league table statistics
    max_values['goals_conceded'] = max(get_team_goals_conceded().values())

    return max_values


def calculate_league_averages(league: str, season: int) -> dict:
    """
    Calculates league-wide averages for all relevant statistics.

    Args:
        league (str): League identifier
        season (int): Season year

    Returns:
        dict: Average values for each statistic across the league
    """
    fbref = sd.FBref(leagues=league, seasons=season)

    # Get all unique stat types needed
    stat_types = {stat_path[0] for category in config.STAT_CATEGORIES.values()
                  for stat_path in category.values()}

    # Get stats only once for each type
    stats = {
        stat_type: fbref.read_team_season_stats(stat_type=stat_type)
        for stat_type in stat_types
    }

    # Calculate averages for each statistic
    averages = {}
    for category in config.STAT_CATEGORIES.values():
        for stat_name, stat_path in category.items():
            stat_type = stat_path[0]
            if len(stat_path) == 3:
                values = stats[stat_type][(stat_path[1], stat_path[2])]
            else:  # len(stat_path) == 2
                values = stats[stat_type][stat_path[1]]

            # Special handling for percentages
            if stat_name in ['challenge_success', 'possession']:
                values = values / 100

            averages[stat_name] = values.mean()

    return averages


def normalize_team_stats(p_team_strength: dict, p_league_max_values: dict) -> dict:
    """
    Normalizes team statistics against league maximum values.

    Args:
        p_team_strength (dict): Team statistics to normalize contained in the dictionary
        p_league_max_values (dict): Maximum values for each statistic in the league

    Returns:
        dict: Normalized team statistics (0-1 scale)
    """
    return {
        # part/whole normalization, STDEV not used here
        stat: p_team_strength[stat] / p_league_max_values[stat]
        for stat in config.STAT_CATEGORIES['offense'].keys() | config.STAT_CATEGORIES['defense'].keys()
        if stat != 'challenge_success'  # Reason being here is that challenge_success is already normalized
    }


def calculate_player_strength(p_player_stats: dict) -> dict:
    """
    Calculates individual player strength metrics from their statistics.

    Args:
        p_player_stats (dict): Dictionary containing player statistics across different categories

    Returns:
        dict: Processed strength metrics for the player
    """
    strength = {}

    for category in config.STAT_CATEGORIES.values():
        for stat_name, stat_path in category.items():
            # Certain stats have a tuple of three that requires specific unpacking
            # Similar reasoning applied above already!
            if len(stat_path) == 3:
                value = p_player_stats[stat_path[0]][(stat_path[1], stat_path[2])].iloc[0]
            else:
                value = p_player_stats[stat_path[0]][stat_path[1]].iloc[0]

            # Handle tackle success rate as this is a percentage as well
            if stat_name == 'challenge_success':
                tackles = p_player_stats['defense'][('Challenges', 'Tkl')].iloc[0]
                value = value / 100 if tackles > 0 else 0

            strength[stat_name] = value

    return strength


def calculate_team_strength(p_team_stats: dict, p_team_roster: list, p_team_name: str) -> dict:
    """
    Calculates team strength by aggregating individual player statistics

    Args:
        p_team_stats (dict): Team statistics across categories
        p_team_roster (list): List of players to include in calculation
        p_team_name (str): Name of the team

    Returns:
        dict: Aggregated team strength metrics including offensive and defensive statistics
    """
    # Calculate individual player strengths
    player_strengths = [
        calculate_player_strength({
            # Find the player
            k: v.loc[v.index.get_level_values('player') == player]
            for k, v in p_team_stats.items() if k != 'team'
        })
        for player in p_team_roster
    ]

    # Sum up individual player statistics for each player
    strength = {
        stat: sum(player[stat] for player in player_strengths)
        for stat in config.STAT_CATEGORIES['offense'].keys() | config.STAT_CATEGORIES['defense'].keys()
    }

    # Define metrics for the team, using ROSTER_MAP just to map between different APIs and their team names
    team_metrics = {
        'possession': p_team_stats['team']['Poss'].iloc[0] / 100,
        'xG_per_90': p_team_stats['team']['Per 90 Minutes', 'npxG'].iloc[0],
        'xAG_per_90': p_team_stats['team']['Per 90 Minutes', 'xAG'].iloc[0],
        'goals_conceded': get_team_goals_conceded().get(config.ROSTER_MAP[p_team_name], 0),
        'matches_played': get_team_matches_played().get(config.ROSTER_MAP[p_team_name], 0)
    }

    # Unpack each dictionary
    return {**strength, **team_metrics}


def create_priors(league_averages: dict):
    """
    Creates informative priors centered on league averages for each statistical category

    Args:
        league_averages (dict): Dictionary of league-wide average statistics

    Returns:
        dict: Prior variables
    """
    priors = {}

    # Create log-scale priors centered on log (for numerical stability) of league averages
    offensive_metrics = ['goals', 'xG', 'xAG']
    defensive_metrics = ['tackles_won', 'interceptions', 'shots_blocked', 'clearances', 'challenge_success']

    for metric in offensive_metrics:
        # League average log
        mu = np.log(league_averages[metric]) if league_averages[metric] > 0 else 0
        # Prior for this metric, weighted log
        priors[f'log_w_{metric}'] = pm.Normal(f'log_w_{metric}', mu=mu, sigma=0.5)
        # For positive weight
        priors[f'w_{metric}'] = pm.Deterministic(f'w_{metric}', pm.math.exp(priors[f'log_w_{metric}']))

    for metric in defensive_metrics:
        # League average log
        mu = np.log(league_averages[metric])
        # Prior for this metric, weighted log
        priors[f'log_w_{metric}'] = pm.Normal(f'log_w_{metric}', mu=mu, sigma=0.5)
        # For positive weight
        priors[f'w_{metric}'] = pm.Deterministic(f'w_{metric}', pm.math.exp(priors[f'log_w_{metric}']))

    return priors


def simulate_match(p_team1_stats: dict, p_team2_stats: dict, p_team1_roster: list, p_team2_roster: list,
                   p_league_max_values: dict, p_team1_name: str, p_team2_name: str):
    """
    THE BIG GUY!

    Simulates a match between two teams using Bayesian modeling

    Using PyMC model, this simulates match outcomes based on team strengths, player statistics,
    and historical performance (as in, this season). The model considers attack and defense strengths,
    possession impact (intuitively, the more a team has the ball, the more likely they are to score),
    and various other metrics to predict goal/expected goal (xG) distributions.

    Args:
        p_team1_stats (dict): Statistics for the first team
        p_team2_stats (dict): Statistics for the second team
        p_team1_roster (list): First team's roster of players
        p_team2_roster (list): Second team's roster of players
        p_league_max_values (dict): Maximum values for statistics in the league to derive a percentile
        p_team1_name (str): Name of the first team
        p_team2_name (str): Name of the second team

    Returns:
        PyMC trace: Posterior samples from the simulation model
    """
    # Calculate team strengths
    team1_strength = calculate_team_strength(p_team1_stats, p_team1_roster, p_team1_name)
    team2_strength = calculate_team_strength(p_team2_stats, p_team2_roster, p_team2_name)

    # Normalize team statistics to derive a percentile
    team1_norm_stats = normalize_team_stats(team1_strength, p_league_max_values)
    team2_norm_stats = normalize_team_stats(team2_strength, p_league_max_values)

    # Add non-normalized stats back, as in, these don't need to be normalized
    team1_norm_stats['challenge_success'] = team1_strength['challenge_success']
    team1_norm_stats['possession'] = team1_strength['possession']
    team2_norm_stats['challenge_success'] = team2_strength['challenge_success']
    team2_norm_stats['possession'] = team2_strength['possession']

    # Get league averages for all stats identified above
    league_averages = calculate_league_averages(league, season)

    with pm.Model() as model:
        # Use log-scale for weights to improve stability:
        # - Goals
        # - xG
        # - xAG
        # - Tackles
        # - Blocks
        # - Clearances
        # - Challenges
        # Use informative prior centered around league average
        priors = create_priors(league_averages)

        def calculate_attack_strength(norm_stats):
            """
            Calculate a tensor to capture the attack strength of a team using goals, xG, and xAG
            :param norm_stats: Normalized stats for the above
            :return: New tensor
            """
            return (
                    norm_stats['goals'] * priors['w_goals'] +
                    norm_stats['xG'] * priors['w_xG'] +
                    norm_stats['xAG'] * priors['w_xAG']
            )

        def calculate_defense_strength(norm_stats):
            """
            Calculate a tensor to capture the defense strength of a team using 'tackles_won', 'interceptions', 'shots_blocked', 'clearances', 'challenge_success'
            :param norm_stats: Normalized stats for the above
            :return: New tensor
            """
            return (
                    norm_stats['tackles_won'] * priors['w_tackles_won'] +
                    norm_stats['interceptions'] * priors['w_interceptions'] +
                    norm_stats['shots_blocked'] * priors['w_shots_blocked'] +
                    norm_stats['clearances'] * priors['w_clearances'] +
                    norm_stats['challenge_success'] * priors['w_challenge_success']
            )

        # Need to re-work this section, getting lots of divergences

        # Get log scale of each team's strength
        log_team1_attack = pm.Deterministic('log_team1_attack',
                                            pm.math.log(
                                                # Add super small number for stability / prevent explosions
                                                calculate_attack_strength(team1_norm_stats) + 1e-6) +
                                            pm.math.log1pexp(
                                                # Possession is a percentage up to 1.0, e.g. team1 has possession 0.6
                                                # of time
                                                (team1_norm_stats['possession'] - 0.5)))

        log_team2_attack = pm.Deterministic('log_team2_attack',
                                            pm.math.log(
                                                calculate_attack_strength(team2_norm_stats) + 1e-6) +
                                            pm.math.log1pexp(
                                                (team2_norm_stats['possession'] - 0.5)))

        log_team1_defense = pm.Deterministic('log_team1_defense',
                                             pm.math.log(
                                                 calculate_defense_strength(team1_norm_stats) + 1e-6) +
                                             pm.math.log1pexp(
                                                 # When one team has possession, the other does not
                                                 (0.5 - team1_norm_stats['possession'])))

        log_team2_defense = pm.Deterministic('log_team2_defense',
                                             pm.math.log(
                                                 calculate_defense_strength(team2_norm_stats) + 1e-6) +
                                             pm.math.log1pexp(
                                                 (0.5 - team2_norm_stats['possession'])))

        # Transform to positive weights
        team1_attack = pm.Deterministic('team1_attack', pm.math.exp(log_team1_attack))
        team2_attack = pm.Deterministic('team2_attack', pm.math.exp(log_team2_attack))
        team1_defense = pm.Deterministic('team1_defense', pm.math.exp(log_team1_defense))
        team2_defense = pm.Deterministic('team2_defense', pm.math.exp(log_team2_defense))

        # Weibull parameters on log scale, choose a low variance and high center around the mean for more "in-tune"
        # results
        log_a_team1 = pm.Normal('log_a_team1', mu=np.log(5), sigma=0.1)
        log_a_team2 = pm.Normal('log_a_team2', mu=np.log(5), sigma=0.1)

        # Transform for positive weights, the alpha here represents the patterns of scoring
        a_team1 = pm.Deterministic('α_team1', pm.math.exp(log_a_team1))
        a_team2 = pm.Deterministic('α_team2', pm.math.exp(log_a_team2))

        # Derive a "strength" ratio by comparing one team's attack to another's defense
        team1_strength_ratio = team1_attack / (team1_attack + team2_defense + 1e-6)
        team2_strength_ratio = team2_attack / (team2_attack + team1_defense + 1e-6)

        # Thesis: Average time between goals increases as more goals are scored
        # Average time between goals here: https://www.soccerstats.com/stats.asp?page=10
        # First table is EPL
        # Transform for positive weights, the beta here represents time between goals
        normalized_max_time = 68 / 90  # 90 minutes in a game

        b_team1 = pm.Deterministic('b_team1',
                                   90 * (1 - pm.math.minimum(normalized_max_time, team1_strength_ratio)))

        b_team2 = pm.Deterministic('b_team2',
                                   90 * (1 - pm.math.minimum(normalized_max_time, team2_strength_ratio)))

        # Expected goals calculation using cumulative hazard
        def weibull_cumulative_hazard(t, alpha, beta):
            return (t / beta) ** alpha

        # Calculate expected goals, the inspiration for using the Weibull renewal was from here:
        # https://medium.com/analytics-vidhya/distribution-of-premier-league-goals-855c909c6955
        # t=90 minutes in a game

        # Weibull really helps model "time until a goal is scored" quite well while also accounting for the renewal,
        # e.g. accumulation of a goal opportunity over the course of 90 minutes and a reset once a goal is scored
        team1_xg = pm.Deterministic('team1_xg',
                                    pm.math.maximum(0.5, weibull_cumulative_hazard(90, a_team1, b_team1)))
        team2_xg = pm.Deterministic('team2_xg',
                                    pm.math.maximum(0.5, weibull_cumulative_hazard(90, a_team2, b_team2)))

        # Poisson distribution for goal counts based on expected goals
        team1_goals = pm.Poisson('team1_goals', mu=team1_xg)
        team2_goals = pm.Poisson('team2_goals', mu=team2_xg)

        trace = pm.sample(
            # Sample 10k, burn 1k
            draws=10_000,
            tune=1_000,
            chains=4,
            cores=1,
            return_inferencedata=True,
            target_accept=0.95,
            init='advi'
        )

    return trace


# --------------------------------------------------------------------------------------------------------------------
league_max_stats = get_league_max_values(league, season)
# Use mappings from config and print each team's index number from 1
print("Available teams:")
for i, team in enumerate(sorted(config.ROSTER_MAP.keys()), 1):
    print(f"{i}. {team}")

# User input
while True:
    try:
        choice1 = int(input("\nPick 1st team (number): "))
        choice2 = int(input("Pick 2nd team (number): "))

        if 1 <= choice1 <= len(config.ROSTER_MAP) and 1 <= choice2 <= len(config.ROSTER_MAP) and choice1 != choice2:
            # Get a list of all teams from the config, sorted alphabetically
            teams = sorted(config.ROSTER_MAP.keys())
            team_1 = teams[choice1 - 1]  # 0-indexed in map, 1 indexed in UI for readability
            team_2 = teams[choice2 - 1]  # ^^
            break
        else:
            print("Please enter valid numbers 1-30")
    except ValueError:
        print("Please enter valid numbers 1-30")

print(f"\nYou chose: {team_1} vs {team_2}")
print("\nFetching Bayesian analysis! Give me a couple minutes...")

trace = simulate_match(get_team_and_player_stats(league, season, team_1, EPL_roster.get(str(team_1), []))
                       , get_team_and_player_stats(league, season, team_2, EPL_roster.get(str(team_2), []))
                       , EPL_roster.get(team_1, []), EPL_roster.get(team_2, []),
                       league_max_stats, team_1, team_2)

# Print graphs and analyze the weights
analysis = trace_analysis.TraceAnalysis(trace)
analysis.analyze_weights()
analysis.analyze_results(team_1, team_2)
