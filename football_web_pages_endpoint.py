import requests

with open('API_KEY.txt', 'r') as f:
    # Dear TAs:
    # Feel free to use this, it's free and has no hard cap on requests
    rapid_api_key = f.read().strip()


def fetch_team_stats(team_id=1):
    # RE: https: // www.footballwebpages.co.uk / api
    # team_id=1 gets the league table for all teams
    url = "https://football-web-pages1.p.rapidapi.com/league-table.json?comp=1&team=1&sort=normal"

    querystring = {"team": str(team_id)}

    headers = {
        'x-rapidapi-host': "football-web-pages1.p.rapidapi.com",
        # TAs, you're free to use my own API key, there's no limit on it
        'x-rapidapi-key': rapid_api_key
    }
    # Return JSON from API call
    return requests.request("GET", url, headers=headers, params=querystring).json()
