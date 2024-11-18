import pandas as pd
import csv

def load_data(filepath):
    """Loads the CSV file into a DataFrame."""
    return pd.read_csv(filepath)

def split_data(data, train_start, train_end, test_start, test_end):
    """
    Splits the data into training, testing, and evaluation sets.

    Args:
        data (pd.DataFrame): Complete dataset.
        train_start (int): Start index for training data.
        train_end (int): End index for training data.
        test_start (int): Start index for testing data.
        test_end (int): End index for testing data.

    Returns:
        tuple: (train_data, test_data, unseen_data)
    """
    train_data = data.iloc[train_start:train_end]
    test_data = data.iloc[test_start:test_end]
    unseen_data = data.iloc[train_end:]
    return train_data, test_data, unseen_data

# Athlete loader

class Athlete:
    def __init__(self, ID, Name, Sex, Age, Height, Weight, Team, NOC, Games, Year, Season, City, Sport, Event, Medal):
        self.ID = int(ID)
        self.Name = Name
        self.Sex = Sex
        self.Age = 0 if Age == 'NA' else float(Age)
        self.Height = 0 if Height == 'NA' else float(Height)
        self.Weight = 0 if Weight == 'NA' else float(Weight)
        self.Team = Team
        self.NOC = NOC
        self.Games = Games
        self.Year = 0 if Year == 'NA' else int(Year)
        self.Season = Season
        self.City = City
        self.Sport = Sport
        self.Event = Event
        self.Medal = Medal

    def __repr__(self):
        return f"Athlete({self.Name}, {self.Sport})"

def load_athletes():
    """Loads the athlete data from the CSV file.

    Returns:
        List: A list of Athlete objects.
    """
    athletes = []
    with open(r'data\athlete_events.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            athlete = Athlete(
                row['ID'], row['Name'], row['Sex'], row['Age'], row['Height'], row['Weight'],
                row['Team'], row['NOC'], row['Games'], row['Year'], row['Season'], row['City'],
                row['Sport'], row['Event'], row['Medal']
            )
            athletes.append(athlete)
    return athletes.sort(key=lambda x: (x.Year, x.ID))

# Sanction loader

class Sanction:
    def __init__(self, sport, substance_reason, sanction_terms, sanction_announced):
        self.sport = sport
        self.substance_reason = substance_reason
        self.sanction_terms = sanction_terms
        self.sanction_announced = sanction_announced

    def __repr__(self):
        return f"Sanction({self.sport}, {self.substance_reason})"

def load_sanction():
    """Loads the sanction data from the CSV file.

    Returns:
        List: A list of Sanction objects.
    """
    sanctions = []
    with open(r'data\athlete_sanction.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sanction = Sanction(
                row['sport'], row['substance_reason'], row['sanction_terms'], row['sanction_announced']
            )
            sanctions.append(sanction)
    return sanctions.sort(key=lambda x: (x.sanction_announced, x.sport))

# NOC maptable
def load_noc_map():
    """Loads the NOC-region map from the CSV file.

    Returns:
        Dict: A dictionary mapping NOC to region and notes.
    """
    noc_map = {}
    with open(r'data\noc_regions.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            noc_map[row['NOC']] = (row['region'], row['notes'])
        return noc_map
    
# Women's competition loader
def load_womens_competition_intro_time():
    """Loads the introduction time of women's competition from the CSV file.

    Returns:
        Dict: A dictionary mapping sport to the year of introduction.
    """
    womens_competition_intro_time = {}
    with open(r'data\introduction_of_women_olympic_sports.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            womens_competition_intro_time[row['sport']] = row['year']
        return womens_competition_intro_time
    
# # Index calculation function
# def calculate_indexes(visitors, coverage, tickets, athletes, intro_time, participation, facilities, magnitude,
#                        audience, budget, race, regions, age_pct, drugs, injuries, cheating, gen_pct, involvement):
#     """this function calculates the indexes for the given parameters"""
#     popularity_index = 0.405 * visitors + 0.115 * coverage + 0.480 * tickets
#     gender_equity_index = 0.633 * athletes + 0.106 * intro_time + 0.260 * participation
#     sustainability_index = 0.122 * facilities + 0.057 * magnitude + 0.263 * audience + 0.558 * budget
#     inclusivity_index = 0.633 * race + 0.260 * regions + 0.106 * age_pct
#     safety_fair_play_index = 0.230 * drugs + 0.122 * injuries + 0.648 * cheating
#     creativity_relativity_index = 0.833 * gen_pct + 0.167 * involvement

#     return [popularity_index, gender_equity_index, sustainability_index, inclusivity_index, safety_fair_play_index, creativity_relativity_index]