import statsapi
import json

if __name__ == "__main__":
    # Check roster position for Ohtani (Dodgers)
    dodgers_id = 119
    print(f"Checking Dodgers roster (ID {dodgers_id})...")
    try:
        roster = statsapi.get('team_roster', {'teamId': dodgers_id, 'season': 2025, 'rosterType': 'fullSeason'})
        for p in roster.get('roster', []):
            if 'Ohtani' in p['person']['fullName']:
                print(f"Found Ohtani: {p['person']['fullName']}, Pos: {p['position']['abbreviation']}")
            if 'Yamamoto' in p['person']['fullName']:
                print(f"Found Yamamoto: {p['person']['fullName']}, Pos: {p['position']['abbreviation']}")
    except Exception as e:
        print(f"Error fetching roster: {e}")

    # Check specific ID
    pid = 660271 # Ohtani
    print(f"Checking ID {pid}...")
    try:
        stats = statsapi.player_stat_data(pid, group="pitching", type="season", sportId=1)
        print("\n--- Method 1: player_stat_data ---")
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"Method 1 failed: {e}")
        params = {
            "stats": "season",
            "group": "pitching",
            "season": season,
            "gameType": "R"
        }
        raw_stats = statsapi.get(endpoint, params)
        print("\n--- Method 2: Direct endpoint ---")
        print(json.dumps(raw_stats, indent=2))
    except Exception as e:
        print(f"Method 2 failed: {e}")

if __name__ == "__main__":
    # debug_player_stats()
    # Check specific ID
    pid = 678994 # Yamamoto
    print(f"Checking ID {pid}...")
    try:
        stats = statsapi.player_stat_data(pid, group="pitching", type="season", sportId=1)
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(e)
