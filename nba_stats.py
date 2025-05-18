import sys
import pandas as pd
import argparse
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, playercareerstats

VALID_STATS = ['PTS', 'AST', 'REB', 'BLK', 'STL']

def is_valid_stat(stat):
    """Check if the provided stat is one of the valid NBA statistics."""
    return stat.upper() in VALID_STATS

class NBAData:
    """Class for handling NBA player data retrieval and formatting using the nba_api."""

    def __init__(self, name):
        """Initialize with player name, fetch player ID and status."""
        self.name = name.title()
        self.player_id = self.lookup_player_id()
        self.is_active = self.check_if_active()

    def __str__(self):
        """String representation showing a summary profile of the player."""
        p = self.get_player_profile()
        return (
            f"\n\U0001F4CB Player Profile for {p['Name']}:\n"
            f"  Team: {p['Team']}\n"
            f"  Position: {p['Position']}\n"
            f"  Height: {p['Height']}\n"
            f"  Weight: {p['Weight']}\n"
            f"  Seasons Active: {p['From Year']} to {p['To Year']}"
        )

    def lookup_player_id(self):
        """Search the NBA API to get the player's unique ID."""
        try:
            all_players = players.get_players()
            match = [p for p in all_players if p['full_name'].lower() == self.name.lower()]
            if match:
                return match[0]['id']
            else:
                raise ValueError(f"Player '{self.name}' not found in NBA API.")
        except Exception as e:
            raise RuntimeError(f"Error looking up player ID: {e}")

    def check_if_active(self):
        """Check if the player is currently on an active NBA roster."""
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=self.player_id)
            data = info.get_data_frames()[0]
            return bool(data.loc[0].get('ROSTERSTATUS', 0))
        except Exception as e:
            raise RuntimeError(f"Error checking active status: {e}")

    def get_player_profile(self):
        """Fetch the player's profile including team, position, height, etc."""
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=self.player_id)
            data = info.get_data_frames()[0]
            return {
                'Name': self.name,
                'Weight': data.loc[0].get('WEIGHT', 'N/A'),
                'Team': data.loc[0].get('TEAM_NAME', 'N/A'),
                'Position': data.loc[0].get('POSITION', 'N/A'),
                'Height': data.loc[0].get('HEIGHT', 'N/A'),
                'From Year': data.loc[0].get('FROM_YEAR', 'N/A'),
                'To Year': data.loc[0].get('TO_YEAR', 'N/A')
            }
        except Exception as e:
            raise RuntimeError(f"Error retrieving player profile: {e}")

    def get_career_averages(self):
        """Calculate and return career averages for core statistics."""
        try:
            career = playercareerstats.PlayerCareerStats(player_id=self.player_id)
            df = career.get_data_frames()[1]
            if df.empty or df['GP'].values[0] == 0:
                return {}
            gp = df['GP'].values[0]
            return {
                'PTS': round(df['PTS'].values[0] / gp, 1),
                'AST': round(df['AST'].values[0] / gp, 1),
                'REB': round(df['REB'].values[0] / gp, 1),
                'BLK': round(df['BLK'].values[0] / gp, 1),
                'STL': round(df['STL'].values[0] / gp, 1),
            }
        except Exception as e:
            raise RuntimeError(f"Error retrieving career averages: {e}")

    def get_recent_games(self, season='2024-25', num_games=10):
        """Retrieve recent game statistics for the given season and game count."""
        if not self.is_active:
            raise ValueError(f"{self.name} is not an active player. Recent game logs unavailable.")
        try:
            logs = playergamelog.PlayerGameLog(player_id=self.player_id, season=season)
            df = logs.get_data_frames()[0]
            if df.empty:
                raise ValueError(f"No game log data available for {self.name} in season {season}.")
            df = df[['GAME_DATE', 'PTS', 'AST', 'REB', 'BLK', 'STL']].head(num_games)
            for col in ['PTS', 'AST', 'REB', 'BLK', 'STL']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            raise RuntimeError(f"Error retrieving recent games: {e}")

class StatFormula:
    """Class for applying statistical analysis and threshold computation."""

    def __init__(self, data, stat, threshold=None):
        """Initialize with game data, stat type, and optional threshold value."""
        self.data = data
        self.stat = stat.upper()
        self.threshold = threshold

    def compute(self):
        """Compute average or probability of exceeding threshold for the stat."""
        try:
            values = self.data[self.stat]
            if self.threshold is not None:
                count = (values >= self.threshold).sum()
                return round(count / len(values), 2)
            return round(values.mean(), 2)
        except KeyError:
            raise ValueError(f"Stat '{self.stat}' not found in game data.")
        except Exception as e:
            raise RuntimeError(f"Error computing stat formula: {e}")

def parse_args(arglist):
    """Parse command-line arguments for player selection and prediction options."""
    parser = argparse.ArgumentParser(description="NBA stat viewer and predictor")
    parser.add_argument("--player", type=str, help="NBA player name")
    parser.add_argument("--stat", type=str, choices=VALID_STATS, help="Stat to analyze")
    parser.add_argument("--threshold", type=float, help="Threshold for stat prediction")
    parser.add_argument("--games", type=int, choices=[5, 10], default=None)
    parser.add_argument("--season", type=str, default="2024-25")
    parser.add_argument("--show-data", action="store_true")
    return parser.parse_args(arglist)

def main(args):
    """Main control function for running the NBA stat predictor interactively or with args."""
    try:
        if not args.player:
            args.player = input("Enter NBA player full name (e.g. LeBron James): ")

        player = NBAData(args.player)

        print(f"\nCareer averages for {player.name}:")
        print(player)

        career_stats = player.get_career_averages()
        if career_stats:
            for stat, val in career_stats.items():
                print(f"  {stat}: {val}")
        else:
            print("  No career data available.")

        if player.is_active:
            proceed = input("\nWould you like to predict stats for next game? (type: yes or no): ").strip().lower()
            if proceed != 'yes':
                return

            if not args.games:
                while True:
                    try:
                        args.games = int(input("\nDo you want to analyze the last 5 or 10 games? Enter 5 or 10: "))
                        if args.games in [5, 10]:
                            break
                        else:
                            print("Please enter either 5 or 10.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")

            args.stat = input("Which stat would you like to predict? (PTS, AST, REB, BLK, STL): ").strip().upper()
            if not is_valid_stat(args.stat):
                raise ValueError(f"'{args.stat}' is not a valid stat.")
            try:
                args.threshold = float(input(f"Enter threshold for {args.stat} (e.g., 20): "))
            except ValueError:
                args.threshold = None

            recent = player.get_recent_games(season=args.season, num_games=args.games)
            print(f"\nRecent games played (last {args.games} games):")
            print(recent)

            predictor = StatFormula(recent, args.stat, args.threshold)
            result = predictor.compute()
            if args.threshold is not None:
                print(f"\nPrediction: {player.name} has a {result*100:.1f}% chance of ≥ {args.threshold} {args.stat}.")
            else:
                print(f"\nAverage {args.stat} over last {args.games} games: {result:.2f}")

        elif args.stat:
            print(f"\nNote: {player.name} is retired. Prediction only works for active players.")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
