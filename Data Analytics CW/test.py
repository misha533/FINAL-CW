import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load and clean data
data = pd.read_csv('recipes.csv')
print(f"Initial dataset contains {len(data)} recipes")

# Drop rows with missing values in key columns
data = data.dropna(subset=['rating_avg', 'rating_val'])
print(f"After cleaning, {len(data)} recipes remain")

# Drop unnamed column if it exists
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Create rating count bins (we'll use this for the boxplot)
bins = [0, 5, 10, 20, 50, 100, np.inf]
labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '100+']
data['rating_bin'] = pd.cut(data['rating_val'], bins=bins, labels=labels)

# --- Visualizations for the relationship between rating_avg and rating_val ---

# 1. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='rating_val', y='rating_avg', data=data, alpha=0.6)
plt.title('Relationship between Average Rating and Number of Ratings')
plt.xlabel('Number of Ratings (rating_val)')
plt.ylabel('Average Rating (rating_avg)')
plt.grid(True)
plt.show()

# 2. Boxplot of Average Rating by Rating Count Bin
plt.figure(figsize=(10, 6))
sns.boxplot(x='rating_bin', y='rating_avg', data=data, order=labels, palette='coolwarm')
plt.title('Average Rating by Number of Ratings Group')
plt.xlabel('Number of Ratings Group (rating_bin)')
plt.ylabel('Average Rating (rating_avg)')
plt.show()

# --- Comments on the relationship ---

print("\nComments on the relationship between Average Rating and Number of Ratings:")
print("Looking at the scatter plot, we can observe the distribution of recipes based on how many ratings they have and their average rating.")
print("It might be noticeable that recipes with very few ratings can have a wider range of average ratings (both very high and very low).")
print("As the number of ratings increases, the average ratings might tend to cluster more tightly, possibly around a certain range.")
print("\nThe boxplot further illustrates this by showing the distribution of average ratings within different groups of rating counts.")
print("We can see if the spread of average ratings (the height of the boxes and the whiskers) changes as the number of ratings increases.")
print("For example, recipes in the '1-5' rating bin might have a larger spread in their average ratings compared to recipes in the '100+' bin.")

# You can add a suggested threshold here if you want, based on the visualizations.
threshold = 5
print(f"\nBased on the visualizations, a possible threshold for considering a rating as more significant could be above {threshold} ratings.")
print("Recipes with fewer ratings might have an average rating that is more influenced by a small number of very positive or negative reviews.")