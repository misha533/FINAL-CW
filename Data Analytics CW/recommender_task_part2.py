import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Load data
df = pd.read_csv('recipes.csv').dropna(subset=['title', 'ingredients'])
 
# Task 4a: Week 7 - Create feature vectors (Slide 6,12)
df['combine_features'] = (
    'title_' + df['title'] + ' ' +
    'rating_' + df['rating_avg'].astype(str) + ' ' +
    'reviews_' + df['rating_val'].astype(str) + ' ' +
    'time_' + df['total_time'].astype(str) + ' ' +
    'cat_' + df['category'] + ' ' +
    'cuisine_' + df['cuisine'] + ' ' +
    'ingr_' + df['ingredients'].str.replace(',', ' ')
)
 
# Task 4b: Week 7 - Vectorize features (Slide 12,17)
count_vec = CountVectorizer(stop_words='english', min_df=2)
count_matrix = count_vec.fit_transform(df['combine_features'])
 
# Task 4c: Week 7 - Recommend using cosine similarity (Slide 15-17,25)
def get_recommendations(target_recipe, n=10):
    try:
        # Find recipe index
        idx = df[df['title'].str.lower() == target_recipe.lower()].index[0]
        # Matrix-vector product (O(n²) operation)
        sim_scores = cosine_similarity(count_matrix[idx:idx + 1], count_matrix)[0]
        # Exclude self and get top N
        sim_scores = [(i, s) for i, s in enumerate(sim_scores) if i != idx]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:n]
        # Print results
        print(f"\nTop {n} recommendations for '{target_recipe}':")
        print(f"{'Rank':<5} {'Recipe Title':<40} {'Similarity':<10}")
        for i, (r_idx, score) in enumerate(sim_scores, 1):
            title = df.iloc[r_idx]['title'][:40] + ('...' if len(df.iloc[r_idx]['title']) > 40 else '')
            print(f"{i:<5} {title:<40} {score:.3f}")
    except IndexError:
        print(f"Recipe '{target_recipe}' not found")
 
# Execute the recommendation
get_recommendations("Chicken and coconut curry")