import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class TraceAnalysis:
    def __init__(self, trace: az.InferenceData):
        """
        Args:
            trace: pm.Sample()
        """
        self.trace = trace
        self.attack_weights = ['w_goals', 'w_xG', 'w_xAG']
        self.defense_weights = ['w_tackles_won', 'w_interceptions', 'w_shots_blocked', 'w_clearances',
                                'w_challenge_success']

    def analyze_weights(self: az.InferenceData):
        """Analyze the learned feature weights"""

        # Attack weights
        print("\nAttack weights:")
        for weight in self.attack_weights:
            # Get the mean for this weight
            mean_weight = float(self.trace.posterior[weight].mean())
            # Get 95% HDI for this weight
            hdi = az.hdi(self.trace.posterior[weight])
            print(f"{weight}: {mean_weight:.3f} (95% HDI: [{float(hdi[weight][0]):.3f}, {float(hdi[weight][1]):.3f}])")

        # Defense weights
        print("\nDefense weights:")
        for weight in self.defense_weights:
            # Get the mean for this weight
            mean_weight = float(self.trace.posterior[weight].mean())
            # Get 95% HDI for this weight
            hdi = az.hdi(self.trace.posterior[weight])
            print(f"{weight}: {mean_weight:.3f} (95% HDI: [{float(hdi[weight][0]):.3f}, {float(hdi[weight][1]):.3f}])")

    def analyze_results(self: az.InferenceData, team1_name: str, team2_name: str):
        # Plot attack weights
        az.plot_posterior(self.trace, var_names=['w_goals', 'w_xG', 'w_xAG'])

        # Plot defense weights
        # 'tackles_won', 'interceptions', 'shots_blocked', 'clearances', 'challenge_success'
        az.plot_posterior(self.trace, var_names=['w_tackles_won', 'w_interceptions',
                                                 'w_shots_blocked', 'w_clearances', 'w_challenge_success'])

        # Plot team variables
        az.plot_posterior(self.trace, var_names=['team1_xg', 'team2_xg'])

        # Unwrap possible values for goals
        team1_goals = self.trace.posterior['team1_goals'].values.flatten()
        team2_goals = self.trace.posterior['team2_goals'].values.flatten()

        # "wins" is just if the number of goals for team x > team y based on the flattened value vector
        team1_wins = np.sum(team1_goals > team2_goals) / len(team1_goals)
        team2_wins = np.sum(team2_goals > team1_goals) / len(team2_goals)
        draws = np.sum(team1_goals == team2_goals) / len(team1_goals)

        print(f"\nMatch Prediction:")
        print(f"{team1_name} win %: {team1_wins:.2%}")
        print(f"{team2_name} win %: {team2_wins:.2%}")
        print(f"Draw %: {draws:.2%}")

        # Same analysis for xG
        team1_xg = self.trace.posterior['team1_xg'].values.flatten()
        team2_xg = self.trace.posterior['team2_xg'].values.flatten()

        print(f"\nxG with 95% HDI:")
        team1_xg_hdi = az.hdi(team1_xg)
        team2_xg_hdi = az.hdi(team2_xg)
        print(f"{team1_name} xG: {np.mean(team1_xg):.2f} (95% HDI: {team1_xg_hdi[0]:.2f} - {team1_xg_hdi[1]:.2f})")
        print(f"{team2_name} xG: {np.mean(team2_xg):.2f} (95% HDI: {team2_xg_hdi[0]:.2f} - {team2_xg_hdi[1]:.2f})")

        def get_matrix_of_goals(p_team1_goals_scored, p_team2_goals_scored, max_goals=9):
            """
            Gets a matrix of all possible scorelines to use in a heatmap plot
            :param p_team1_goals_scored: List of goals scored in simulation by team 1
            :param p_team2_goals_scored: List of goals scored in simulation by team 2
            :param max_goals: Max goals to store in the matrix, e.g. 9
            :return: Matrix of possible scoreline pairs and their probability
            """
            # Set max_goals to 9, as it was the most amount of goals scored last season
            scores = np.zeros((max_goals + 1, max_goals + 1))
            # Zip all possible scorelines into a pandas Series
            scorelines = pd.Series(list(zip(p_team1_goals_scored, p_team2_goals_scored))).value_counts()
            total_sims = len(p_team1_goals_scored)

            # Fill the matrix with probabilities
            for (t1g, t2g), count in scorelines.items():
                if t1g <= max_goals and t2g <= max_goals:
                    # Get probability of each pair
                    scores[t1g, t2g] = count / total_sims

            return scores

        def analyze_heatmap(p_team1_goals_scored, p_team2_goals_scored, p_team1_name, p_team2_name, max_goals=9):
            """
            Prepares and returns a heatmap of possible scoreline probabilities based on the matrix obtained above
            :param p_team1_goals_scored: List of goals scored in simulation by team 1
            :param p_team2_goals_scored: List of goals scored in simulation by team 2
            :param p_team1_name: Team 1 name, for labeling
            :param p_team2_name: Team 2 name, for labeling
            :param max_goals: Max goals to store in the matrix, e.g. 9
            :return:
            """
            plt.figure(figsize=(10, 8))

            # Prepare heatmap using matrix in above helper method
            sns.heatmap(get_matrix_of_goals(p_team1_goals_scored, p_team2_goals_scored, max_goals),
                        annot=True,
                        # For percentage probabilities
                        fmt='.1%',
                        xticklabels=range(max_goals + 1),
                        yticklabels=range(max_goals + 1),
                        cbar_kws={'label': 'Probability'})

            # Labels for each team and their probability distributions
            plt.xlabel(f'{p_team2_name} Goals')
            plt.ylabel(f'{p_team1_name} Goals')
            plt.title('Scoreline Probability Distribution')

            return plt

        analyze_heatmap(team1_goals, team2_goals, team1_name, team2_name)
        plt.show()

        az.plot_posterior(self.trace, var_names=['team1_goals', 'team2_goals'], hdi_prob=0.95)
        plt.show()
