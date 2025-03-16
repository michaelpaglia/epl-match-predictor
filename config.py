# Keep in mind, this uses rosters for opening day. I didn't account for injuries or anything weird but feel free to
# adjust the Excel file if you want to tinker with players.
FILE = "epl_teams.xlsx"
# APIs called different teams by slightly different string names
ROSTER_MAP = {
    "Manchester City": "Manchester City",
    "Liverpool": "Liverpool",
    "Brighton": "Brighton & Hove Albion",
    "Arsenal": "Arsenal",
    "Newcastle Utd": "Newcastle United",
    "Brentford": "Brentford",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "AFC Bournemouth",
    "Nott'ham Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "Chelsea": "Chelsea",
    "Fulham": "Fulham",
    "West Ham": "West Ham United",
    "Manchester Utd": "Manchester United",
    "Leicester City": "Leicester City",
    "Crystal Palace": "Crystal Palace",
    "Ipswich Town": "Ipswich Town",
    "Wolves": "Wolverhampton Wanderers",
    "Southampton": "Southampton",
    "Everton": "Everton"
}
# Defined from https://soccerdata.readthedocs.io/en/latest/datasources/FBref.html
STAT_CATEGORIES = {
    'offense': {
        'goals': ('standard', 'Performance', 'Gls'),
        'xG': ('standard', 'Expected', 'xG'),
        'xAG': ('standard', 'Expected', 'xAG')
    },
    'defense': {
        'tackles_won': ('defense', 'Tackles', 'TklW'),
        'interceptions': ('defense', 'Int'),
        'shots_blocked': ('defense', 'Blocks', 'Sh'),
        'clearances': ('defense', 'Clr'),
        'challenge_success': ('defense', 'Challenges', 'Tkl%')
    }
}