import requests


def fetch_team_stats(team_id=1):
    url = "https://football-web-pages1.p.rapidapi.com/league-table.json?comp=1&team=1&sort=normal"

    querystring = {"team": str(team_id)}

    headers = {
        'x-rapidapi-host': "football-web-pages1.p.rapidapi.com",
        'x-rapidapi-key': "3e2b58e614msh8a389e9652783ebp136699jsnad69d1dc4701"
    }

    return requests.request("GET", url, headers=headers, params=querystring).json()
