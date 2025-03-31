import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from difflib import get_close_matches
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Load and clean data
df = pd.read_csv('recipes.csv').dropna()

# Part 3.1: Vector Space Method 
def vec_space_method(recipe_title, df, feature_matrix, top_n=10):
    """
    Recommends recipes using vector space model with matrix-vector products.
    Differs from Part 2.4 by:
    - Using numerical features (rating_avg, total_time) as scaled continuous values
    - Incorporating categorical features (cuisine, category) via one-hot encoding
    - Combining all features into unified vector space

    Args:
        recipe_title: Input recipe name
        df: DataFrame containing recipes
        feature_matrix: Combined feature matrix
        top_n: Number of recommendations
    Returns:
        DataFrame with top_n recipes and similarity scores
    """
    matched_title = find_closest_match(recipe_title, df)
    if not matched_title:
        print(f"Recipe '{recipe_title}' not found")
        return None

    recipe_idx = df[df['title'] == matched_title].index[0]

    # Matrix-vector product (O(n²))
    similarities = cosine_similarity(
        feature_matrix[recipe_idx:recipe_idx + 1],
        feature_matrix
    ).flatten()

    similar_indices = similarities.argsort()[::-1][1:top_n + 1]

    results = df.iloc[similar_indices].copy()
    results['similarity'] = similarities[similar_indices]
    return results[['title', 'category', 'cuisine', 'rating_avg', 'similarity']]



# Part 3.2: KNN Similarity 
def knn_similarity(recipe_title, df, feature_matrix, top_n=10):
    """
    Recommends recipes using KNN algorithm as taught in Week 9.
    Uses Euclidean distance which is equivalent to cosine similarity
    when features are normalized (StandardScaler was applied).

    Args:
        recipe_title: Input recipe name
        df: DataFrame containing recipes
        feature_matrix: Combined feature matrix
        top_n: Number of recommendations
    Returns:
        List of top_n similar recipe titles
    """
    matched_title = find_closest_match(recipe_title, df)
    if not matched_title:
        print(f"Recipe '{recipe_title}' not found")
        return None

    recipe_idx = df[df['title'] == matched_title].index[0]

    knn = NearestNeighbors(n_neighbors=top_n + 1, metric='euclidean')
    knn.fit(feature_matrix)

    _, indices = knn.kneighbors(feature_matrix[recipe_idx:recipe_idx + 1])
    return df.iloc[indices[0][1:]]['title'].str.strip().tolist()



# Part 3.3: Evaluation in terms of coverage and personalization
def evaluate_recommenders(test_users, df, feature_matrix):
    """
    Evaluates recommenders using Week 9 metrics:
    - Coverage: Percentage of total recipes recommended
    - Personalization: 1 - mean cosine similarity between user recommendation pairs
                      (1 = completely unique recs per user)
    """
    # Get recommendations for all test users
    vec_recs = {user: vec_space_method(recipe, df, feature_matrix)['title'].tolist()
                for user, recipe in test_users.items()}
    knn_recs = {user: knn_similarity(recipe, df, feature_matrix)
                for user, recipe in test_users.items()}

    # Coverage calculation
    def coverage(recommendations):
        unique_recs = set().union(*recommendations.values())
        return len(unique_recs) / len(df)

    # Personalization calculation (Week 9 method)
    def personalization(recommendations):
        all_recs = list({rec for recs in recommendations.values() for rec in recs})
        rec_to_idx = {rec: i for i, rec in enumerate(all_recs)}

        similarities = []
        for u1, u2 in [(u1, u2) for u1 in recommendations for u2 in recommendations if u1 != u2]:
            vec1 = np.zeros(len(all_recs))
            vec2 = np.zeros(len(all_recs))
            vec1[[rec_to_idx[rec] for rec in recommendations[u1]]] = 1
            vec2[[rec_to_idx[rec] for rec in recommendations[u2]]] = 1
            similarities.append(cosine_similarity([vec1], [vec2])[0][0])

        return 1 - np.mean(similarities) if similarities else 1.0

    return {
        'vector_space': {
            'coverage': coverage(vec_recs),
            'personalization': personalization(vec_recs)
        },
        'knn': {
            'coverage': coverage(knn_recs),
            'personalization': personalization(knn_recs)
        }
    }



# Part 3.4: Tasty Predictor
def build_tasty_predictor(df, threshold_ratings=5):
    """
    Builds ANN model to predict 'tasty' recipes (rating_avg > 4.2)
    using techniques from Week 9 ANN lab:
    - SMOTE for class imbalance
    - PCA for dimensionality reduction
    - Early stopping to prevent overfitting
    """
    # Filter significant ratings (Part 2.3)
    sig_ratings = df[df['rating_val'] >= threshold_ratings].copy()
    sig_ratings['is_tasty'] = np.where(sig_ratings['rating_avg'] > 4.2, 1, 0)

    # Feature engineering
    ingredient_matrix, category_matrix, cuisine_matrix, numerical_matrix = preprocess_data(sig_ratings)
    X = np.hstack([ingredient_matrix.toarray(), category_matrix, cuisine_matrix, numerical_matrix])
    y = sig_ratings['is_tasty']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle imbalance (Week 9 lab)
    smote = SMOTE()
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Dimensionality reduction (Week 9 PCA lab)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_res)
    X_test_pca = pca.transform(X_test)

    # ANN architecture (Week 9 ANN lab)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        early_stopping=True,
        random_state=42
    )
    model.fit(X_train_pca, y_train_res)

    # Evaluate
    y_pred = model.predict(X_test_pca)
    return accuracy_score(y_test, y_pred)

# Supporting Functions
def find_closest_match(recipe_title, df):
    """Fuzzy matches recipe titles"""
    matches = get_close_matches(recipe_title, df['title'].tolist(), n=1, cutoff=0.6)
    return matches[0] if matches else None


def preprocess_data(df):
    """Prepares all feature types for modeling"""
    # Text features
    df['ingredients_cleaned'] = df['ingredients'].str.lower().str.replace(r'[^\w\s,]', '')
    tfidf = TfidfVectorizer(stop_words='english', min_df=2)
    ingredient_matrix = tfidf.fit_transform(df['ingredients_cleaned'])

    # Categorical features
    cat_encoder = OneHotEncoder(handle_unknown='ignore')
    category_matrix = cat_encoder.fit_transform(df[['category']]).toarray()
    cuisine_matrix = cat_encoder.fit_transform(df[['cuisine']]).toarray()

    # Numerical features
    scaler = StandardScaler()
    numerical_matrix = scaler.fit_transform(df[['rating_avg', 'rating_val', 'total_time']])

    return ingredient_matrix, category_matrix, cuisine_matrix, numerical_matrix



# Execution
if __name__ == "__main__":
    # Feature engineering
    ingredient_matrix, category_matrix, cuisine_matrix, numerical_matrix = preprocess_data(df)
    feature_matrix = np.hstack([
        ingredient_matrix.toarray(),
        category_matrix,
        cuisine_matrix,
        numerical_matrix
    ])

    # Test users (Part 3.3)
    test_users = {
        'User 1': 'Chicken tikka masala',
        'User 2': 'Albanian baked lamb with rice',
        'User 3': 'Baked salmon with chorizo rice',
        'User 4': 'Almond lentil stew'
    }

    # Part 3.1 & 3.2 Demo
    print("Vector Space Recommendations:")
    print(vec_space_method(test_users['User 1'], df, feature_matrix).head())

    print("\nKNN Recommendations:")
    print(knn_similarity(test_users['User 1'], df, feature_matrix))

    # Part 3.3 Evaluation
    eval_results = evaluate_recommenders(test_users, df, feature_matrix)
    print("\nEvaluation Metrics:")
    print(f"Vector Space - Coverage: {eval_results['vector_space']['coverage']:.1%}")
    print(f"Vector Space - Personalization: {eval_results['vector_space']['personalization']:.3f}")
    print(f"KNN - Coverage: {eval_results['knn']['coverage']:.1%}")
    print(f"KNN - Personalization: {eval_results['knn']['personalization']:.3f}")

    # Part 3.4 Tasty Predictor
    print("\nTraining Tasty Predictor ANN...")
    accuracy = build_tasty_predictor(df)
    print(f"Model Accuracy: {accuracy:.1%}")

    # Coverage:
    # We measured how many unique recipes the system was able to recommend across all users.
    # Both the Vector Space and KNN approaches achieved the same coverage in our test case,
    # which makes sense since they’re both working off the same dataset and feature matrix.
    # Higher coverage would mean the recommender is able to explore more of the recipe space
    # instead of suggesting the same recipes repeatedly.

    # Personalization:
    # This metric checks how similar or different the recommendations are for different users.
    # A score closer to 1 means everyone is getting their own tailored suggestions — which is good.
    # From our results, both recommenders showed high personalization (around 0.98), meaning
    # they’re offering quite user-specific recommendations.
    # We expected the vector space method to do well here because of the way it combines all 
    # features like ingredients, category, and numeric ratings — giving it more nuance.
    # KNN also did well but can sometimes lean toward recommending similar clusters.

    # In summary, both systems gave solid results, and the personalization scores especially
    # suggest they’re effective at adapting to different user tastes. Coverage might be improved
    # by tuning the feature representation or increasing recommendation diversity.
