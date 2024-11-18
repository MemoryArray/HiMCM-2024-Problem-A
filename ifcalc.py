import csv
import numpy as np
from collections import defaultdict

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
    """Loads the athlete data from the CSV file."""
    athletes = []
    with open('data/athlete_events.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            athlete = Athlete(
                row['ID'], row['Name'], row['Sex'], row['Age'], row['Height'], row['Weight'],
                row['Team'], row['NOC'], row['Games'], row['Year'], row['Season'], row['City'],
                row['Sport'], row['Event'], row['Medal']
            )
            athletes.append(athlete)
    return athletes

# Sanction loader
class Sanction:
    def __init__(self, sport, substance_reason, sanction_terms, sanction_announced):
        self.sport = sport
        self.substance_reason = substance_reason
        self.sanction_terms = sanction_terms
        self.sanction_announced = sanction_announced

    def __repr__(self):
        return f"Sanction({self.sport}, {self.substance_reason})"

def load_sanctions():
    """Loads the sanction data from the CSV file."""
    sanctions = []
    with open('data/sanction.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sanction = Sanction(
                row['sport'], row['substance_reason'], row['sanction_terms'], row['sanction_announced']
            )
            sanctions.append(sanction)
    return sanctions

# NOC maptable
def load_noc_map():
    """Loads the NOC-region map from the CSV file."""
    noc_map = {}
    with open('data/noc_regions.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            noc_map[row['NOC']] = (row['region'], row['notes'])
    return noc_map

# Women's competition loader
def load_womens_competition_intro_time():
    """Loads the introduction time of women's competition from the CSV file."""
    womens_competition_intro_time = {}
    with open('data/introduction_of_women_olympic_sports.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            womens_competition_intro_time[row['sport']] = row['year']
    return womens_competition_intro_time

# Index calculation function
def calculate_indexes(visitors, coverage, tickets, athletes, intro_time, participation, facilities, magnitude,
                       audience, budget, race, regions, age_pct, drugs, injuries, cheating, gen_pct, involvement):
    """this function calculates the indexes for the given parameters"""
    popularity_index = 0.405 * visitors + 0.115 * coverage + 0.480 * tickets
    gender_equity_index = 0.633 * athletes + 0.106 * intro_time + 0.260 * participation
    sustainability_index = 0.122 * facilities + 0.057 * magnitude + 0.263 * audience + 0.558 * budget
    inclusivity_index = 0.633 * race + 0.260 * regions + 0.106 * age_pct
    safety_fair_play_index = 0.230 * drugs + 0.122 * injuries + 0.648 * cheating
    creativity_relativity_index = 0.833 * gen_pct + 0.167 * involvement

    return [popularity_index, gender_equity_index, sustainability_index, inclusivity_index, safety_fair_play_index, creativity_relativity_index]


# Function to calculate indexes for all sports
def calculate_indexes_for_all_sports():
    athletes = load_athletes()
    sanctions = load_sanctions()
    noc_map = load_noc_map()
    womens_competition_intro_time = load_womens_competition_intro_time()

    # Initialize a dictionary to store the calculated indexes for each sport
    sport_indexes = defaultdict(list)

    # Group athletes by sport
    sport_data = defaultdict(lambda: {'athletes': [], 'sanctions': []})
    
    # Collect athlete data by sport
    for athlete in athletes:
        sport_data[athlete.Sport]['athletes'].append(athlete)
    
    # Collect sanction data by sport
    for sanction in sanctions:
        sport_data[sanction.sport]['sanctions'].append(sanction)
    
    # Process the data for each sport
    for sport, data in sport_data.items():
        # Calculate parameters for each sport
        athletes_count = len(data['athletes'])  # Number of athletes in the sport
        intro_time = int(womens_competition_intro_time.get(sport, 1900))  # Year when women's competition was introduced
        participation = 45  # Placeholder value, replace with actual calculation
        facilities = 20  # Placeholder value
        magnitude = 30  # Placeholder value
        audience = 10000  # Placeholder value
        budget = 100000  # Placeholder value
        race = 20  # Placeholder value
        regions = 30  # Placeholder value
        age_pct = 50  # Placeholder value
        drugs = len([s for s in data['sanctions'] if s.substance_reason != 'NA'])  # Number of sanctions for drugs
        injuries = 3  # Placeholder value
        cheating = 2  # Placeholder value
        gen_pct = 60  # Placeholder value
        involvement = 50  # Placeholder value

        # Calculate indexes for the sport
        indexes = calculate_indexes(
            1000, 200, 300, athletes_count, intro_time, participation, facilities, magnitude,
            audience, budget, race, regions, age_pct, drugs, injuries, cheating, gen_pct, involvement
        )

        # Store the indexes for the sport
        sport_indexes[sport] = indexes

    return sport_indexes

# Call the function and print the results
sport_indexes = calculate_indexes_for_all_sports()
for sport, indexes in sport_indexes.items():
    print(f"Sport: {sport}")
print(f"Indexes: Popularity: {indexes[0]}, Gender Equity: {indexes[1]}, Sustainability: {indexes[2]}, Inclusivity: {indexes[3]}, Safety/Fair Play: {indexes[4]}, Creativity/Relativity: {indexes[5]}")
