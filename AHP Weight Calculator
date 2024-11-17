import numpy as np
from scipy.linalg import eig

class Sport:
    def __init__(self, name, attributes, criteria_comparisons):
        """
        Initializes the sport with its name and a dictionary of attributes.
        Attributes should be a dictionary with the attribute names as keys
        and their corresponding values (the raw data for the sport) as values.
        
        criteria_comparisons is a matrix representing the pairwise comparisons
        between criteria based on AHP.
        """
        self.name = name
        self.attributes = attributes
        
        # Apply AHP process to calculate the criteria weights
        self.weights = self.calculate_criteria_weights(criteria_comparisons)
        
    def calculate_criteria_weights(self, comparisons):
        """
        Apply AHP method to calculate the normalized weights for each criterion.
        'comparisons' is a square matrix where each element (i, j) represents the 
        relative importance of criterion i over criterion j.
        """
        # Convert the comparison matrix to a numpy array
        comparison_matrix = np.array(comparisons)

        # Calculate the eigenvector corresponding to the largest eigenvalue
        # This gives the priority vector (the weights for each criterion)
        eigvals, eigvecs = eig(comparison_matrix)

        # The eigenvector corresponding to the largest eigenvalue (the one with the largest real part)
        max_eigenvalue_index = np.argmax(np.real(eigvals))
        priority_vector = np.real(eigvecs[:, max_eigenvalue_index])

        # Normalize the priority vector so that the weights sum to 1
        normalized_weights = priority_vector / np.sum(priority_vector)

        # Return the weights as a dictionary
        return {
            'gender_equality': normalized_weights[0],
            'popularity': normalized_weights[1],
            'sustainability': normalized_weights[2],
            'inclusivity': normalized_weights[3],
            'creativity_and_relativity': normalized_weights[4],
            'safety_and_fairness': normalized_weights[5]
        }
        
    def calculate_criteria_score(self, criteria_name):
        """
        Given the criteria name, calculates the weighted score for that criteria
        based on the sport's attributes.
        """
        if criteria_name == 'gender_equality':
            return (self.attributes['total_women_athletes'] * self.weights['gender_equality'] +
                    self.attributes['total_men_athletes'] * self.weights['gender_equality'] +
                    self.attributes['time_of_introduction_of_women_sports'] * self.weights['gender_equality'] +
                    self.attributes['womens_participation'] * self.weights['gender_equality'])
        
        elif criteria_name == 'popularity':
            return (self.attributes['daily_visitors_to_website'] * self.weights['popularity'] +
                    self.attributes['media_coverage'] * self.weights['popularity'])
        
        elif criteria_name == 'sustainability':
            return (self.attributes['new_facility_count'] * self.weights['sustainability'] +
                    self.attributes['event_magnitude'] * self.weights['sustainability'] +
                    self.attributes['audience_magnitude'] * self.weights['sustainability'] +
                    self.attributes['public_appeal'] * self.weights['sustainability'] +
                    self.attributes['social_safety'] * self.weights['sustainability'] +
                    self.attributes['budget_balance'] * self.weights['sustainability'] +
                    self.attributes['financial_risk'] * self.weights['sustainability'])
        
        elif criteria_name == 'inclusivity':
            return (self.attributes['athletes_race_and_country'] * self.weights['inclusivity'] +
                    self.attributes['regions'] * self.weights['inclusivity'])
        
        elif criteria_name == 'creativity_and_relativity':
            return self.attributes['percentage_of_athletes_per_generation'] * self.weights['creativity_and_relativity']
        
        elif criteria_name == 'safety_and_fairness':
            return (self.attributes['sanction_of_drugs'] * self.weights['safety_and_fairness'] +
                    self.attributes['injury_rate'] * self.weights['safety_and_fairness'] +
                    self.attributes['severity_rate'] * self.weights['safety_and_fairness'] +
                    self.attributes['cheating_instances'] * self.weights['safety_and_fairness'] +
                    self.attributes['judges_influence'] * self.weights['safety_and_fairness'])
        
        return 0
    
    def calculate_final_score(self):
        """
        Calculates the final normalized score for the sport, summing up the
        weighted scores of each criterion and normalizing the result to a range [0, 1].
        """
        total_score = 0
        
        # Calculate the weighted scores for each criterion
        for criteria in self.weights.keys():
            total_score += self.calculate_criteria_score(criteria)
        
        # Normalize the total score to be between 0 and 1 (simple normalization)
        return max(0, min(1, total_score))


# Criteria comparison matrix for AHP process (based on my weights above)
criteria_comparisons = [
    [1, 1/3, 3, 5, 7, 9],       # gender_equality
    [3, 1, 5, 7, 9, 5],         # popularity
    [1/3, 1/5, 1, 3, 5, 7],     # sustainability
    [1/5, 1/7, 1/3, 1, 3, 5],   # inclusivity
    [1/7, 1/9, 1/5, 1/3, 1, 3], # creativity_and_relativity
    [1/9, 1/5, 1/7, 1/5, 1/3, 1]  # safety_and_fairness
]

# Example attributes for a sport
sport_attributes = {
    'total_women_athletes': 500,
    'total_men_athletes': 1000,
    'time_of_introduction_of_women_sports': 15,
    'womens_participation': 45,
    'daily_visitors_to_website': 20000,
    'media_coverage': 70,
    'new_facility_count': 10,
    'event_magnitude': 8,
    'audience_magnitude': 9,
    'public_appeal': 6,
    'social_safety': 7,
    'budget_balance': 6,
    'financial_risk': 3,
    'athletes_race_and_country': 80,
    'regions': 5,
    'percentage_of_athletes_per_generation': 60,
    'sanction_of_drugs': 3,
    'injury_rate': 2,
    'severity_rate': 1,
    'cheating_instances': 0,
    'judges_influence': 1
}

# Create an instance of the Sport class
sport = Sport("Football", sport_attributes, criteria_comparisons)

# Calculate the final score for the sport
final_score = sport.calculate_final_score()

print(f"Final score for {sport.name}: {final_score:.2f}")
