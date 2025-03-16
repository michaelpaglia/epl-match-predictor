import pandas as pd


def read_excel(file_path):
    """
    Read the Excel file and return a pandas DataFrame.
    """
    return pd.read_excel(file_path)


def clean_data(df):
    """
    Clean the data by removing any NaN values and stripping whitespace.
    """
    return df.dropna(how='all').map(lambda x: x.strip() if isinstance(x, str) else x)


def create_team_roster_dict(df):
    """
    Add all players to "teams" to create rosters
    """
    team_roster = {}
    for column in df.columns:
        players = df[column].dropna().tolist()
        team_roster[column] = players
    return team_roster


def process_dict(file_path):
    """
    Read the Excel file with all players, create a dataframe, and then team rosters
    """
    df = read_excel(file_path)
    cleaned_df = clean_data(df)
    team_roster = create_team_roster_dict(cleaned_df)
    return team_roster
