import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
 
# Task 1: Load Data and Preprocessing
# Reference: Week 4 - "CO2106 L4 - Data Preprocessing & Visualisation.pdf" Slide 17-18 (Missing values)
# Concept: Data cleaning by removing missing values
df = pd.read_csv('recipes.csv')
df.dropna(inplace=True)  # Week 4 - "CO2106 L4" Slide 17: dropna() removes rows with missing values
 
print("\nSummary Statistics for Average Rating:\n", df[['rating_avg']].describe())  # Week 4 - "CO2106 L4" Slide 10: describe() shows summary statistics
 
 
# Task 2: Bootstrapping Confidence Interval 
# Reference: Week 6 - "CO2106 L41 - Confidence Intervals.pdf" Slide 34-36 (Bootstrapping)
# Concept: Using resampling to estimate confidence intervals
np.random.seed(42)
bootstrap_means = []
 
for _ in range(1000):  # Week 6 - "CO2106 L41" Slide 36: Resampling with replacement
    sample = df['rating_avg'].sample(n=min(100, len(df)), replace=True)
    bootstrap_means.append(sample.mean())
 
ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])  # Week 6 - "CO2106 L41" Slide 28: Percentile method for CI
print(f"\n95% Confidence Interval for Average Ratings: ({ci_lower:.2f}, {ci_upper:.2f})")
print(f"Interpretation of Confidence Interval: We are 95% confident that the true average rating across all recipes lies between {ci_lower:.2f} and {ci_upper:.2f}.")  # Week 6 - "CO2106 L41" Slide 26-28: CI interpretation
 
 
# Top 10 Highest Rated + Most Popular
# Reference: Week 4 - "CO2106 L4 - Data Preprocessing & Visualisation.pdf" Slide 10 (Descriptive Analysis)
# Concept: Using groupby and aggregation functions
 
# Apply threshold for significance
threshold = 5  # Reference: Week 6 - "CO2106 L41" Slide 40 (Outliers and significance)
reliable_recipes = df[df['rating_val'] >= threshold]  # Filtering based on threshold
 
# Week 4 - "CO2106 L4" Slide 10: Groupby and aggregation
avg_ratings_reliable = reliable_recipes.groupby('title', as_index=False).agg({
    'rating_avg': 'mean',
    'rating_val': 'sum'
})
 
top_10_avg_rated = avg_ratings_reliable.sort_values(
    by=['rating_avg', 'rating_val'],
    ascending=[False, False]
).head(10)
 
print("\nTop 10 Highest Average Rated Recipes (with ≥ 5 ratings):")
print(top_10_avg_rated[['title', 'rating_avg', 'rating_val']])
 
# Most popular recipe (based on count)
most_popular_recipe = df.groupby('title')['rating_val'].count().idxmax()  # Week 4 - "CO2106 L4" Slide 10: count() aggregation
print(f"\nMost Popular Recipe (Highest Number of Ratings): {most_popular_recipe}")
 
 
# Task 3a: Distribution of Number of Ratings (Log Scale)
# Reference: Week 4 - "CO2106 L4 - Data Preprocessing & Visualisation.pdf" Slide 30-31 (Histograms)
# Concept: Visualizing distributions with log transformation
def plot_rating_distribution_with_binning(df, bins=30):
    plt.figure(figsize=(12, 6))
    df['log_ratings'] = np.log1p(df['rating_val'])  # Week 4 - "CO2106 L4" Slide 30: Log transform for right-skewed data
    sns.histplot(df['log_ratings'], bins=bins, kde=True, color="skyblue", edgecolor='black')
    plt.xlabel("Logarithm of (Number of Ratings + 1)")
    plt.ylabel("Number of Recipes")
    plt.title("Distribution of Number of Ratings (Log Scale)")
    plt.xticks(np.log1p([1, 5, 10, 25, 50, 100, 200, 500, 800]),
               ["1", "5", "10", "25", "50", "100", "200", "500", "800"])
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()
 
plot_rating_distribution_with_binning(df)
 
 
# Task 3b: Relationship Between Number of Ratings and Average Rating
# Reference: Week 4 - "CO2106 L4 - Data Preprocessing & Visualisation.pdf" Slide 32-33 (Scatter Plots)
# Concept: Visualizing relationships between variables
df_rating_counts = df.groupby('title').agg(
    rating_avg=('rating_avg', 'mean'),
    num_ratings=('rating_val', 'count')
).reset_index()
 
df_rating_counts = df_rating_counts[df_rating_counts['rating_avg'] <= 5]  # Filtering outliers (Week 4 - "CO2106 L4" Slide 40)
 
print(f"\nSuggested threshold for significance: {threshold} ratings")
print(f"Justification: Recipes with fewer than {threshold} ratings may represent outliers or may not be reliable due to small sample size. Statistically, a larger number of ratings (n >= {threshold}) tends to provide a more stable and reliable estimate of the average rating, aligning with basic statistical principles regarding sample size and representativeness.")  # Week 6 - "CO2106 L41" Slide 23 (CLT)
 
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df_rating_counts, x='num_ratings', y='rating_avg', alpha=0.7)  # Week 4 - "CO2106 L4" Slide 32: Scatter plot
plt.xscale('log')  # Log scale for better visualization
plt.xlabel("Number of Ratings (Log Scale)")
plt.ylabel("Average Rating")
plt.title("Relationship Between Number of Ratings and Average Rating")
plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold: {threshold} Ratings')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
print("\nRelationship between average rating and number of ratings:")
print("Visual analysis shows dispersion decreases as sample size increases.")  # Week 6 - "CO2106 L41" Slide 23 (CLT)
print("Consistent with Central Limit Theorem – larger samples yield more stable estimates.")
 
significant_ratings = df[df['rating_val'] >= threshold]
print(f"\nNumber of recipes with statistically significant ratings (>= {threshold}): {len(significant_ratings)}")