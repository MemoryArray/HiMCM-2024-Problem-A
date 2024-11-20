import matplotlib.pyplot as plt

# Data: Selected sports and their popularity indexes
sports = [
    "Basketball", "Football", "Swimming", "Weightlifting", "Wrestling",
    "Tennis", "Boxing", "Volleyball", "Athletics", "Gymnastics"
]
popularity_indexes = [
    0.1201, 0.0999, 0.0817, 0.0664, 0.0546,
    0.0245, 0.0363, 0.0305, 0.0131, 0.0037
]

# Sorting the sports and popularity indexes by popularity index (highest to lowest)
sorted_sports, sorted_popularity_indexes = zip(*sorted(zip(sports, popularity_indexes), key=lambda x: x[1], reverse=True))

# Define the custom repeating color palette
colors = [
    '#63b2ee', '#76da91', '#f8cb7f', '#f89588', '#7cd6cf', '#9192ab',
    '#7898e1', '#efa666', '#eddd86', '#9987ce'
]

# Create a figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot a horizontal bar graph with custom colors
bars = ax.barh(sorted_sports, sorted_popularity_indexes, color=colors)

# Set labels and title
ax.set_xlabel('Popularity Index', fontsize=14, family='Times New Roman')
ax.set_ylabel('Olympic Sport', fontsize=14, family='Times New Roman')
ax.set_title('Top 10 Olympic Sports by Popularity Index', fontsize=16, family='Times New Roman')

# Change font of tick labels
plt.xticks(fontsize=12, family='Times New Roman')
plt.yticks(fontsize=12, family='Times New Roman')

# Display the plot
plt.tight_layout()
plt.show()
