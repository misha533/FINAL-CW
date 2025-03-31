
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import math
import re

sys.stdout.reconfigure(encoding='utf-8')

# Week 5: Data Pre-processing - Data Frame Structure
# Reqired columns for structured output
REQUIRED_COLUMNS = ['title', 'total_time', 'image', 'ingredients', 'rating_val',
                    'rating_count', 'category', 'cuisine', 'diet', 'vegan', 'vegetarian', 'url']

def collect_page_data(url):
    # Week 2: Ethical Web Scraping
    # Setting headers to identify ourselves ethically as said in the powerpoints
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
       
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        # Error handling as per Week 6: Testing  - Handling failures
        return None

    # Week 2: Web Crawling - DOM Parsing
    # Parsing HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    recipe_data = {}

    # Extracting recipe title (Week 2: Understanding website data)
    title = soup.find('h1', class_='ssrcss-pbttu9-Heading')
    recipe_data["title"] = title.get_text(strip=True) if title else "Title Not Found"

    # Extracting cooking time (Week 2:  Data extraction)
    totaltime = soup.find('dd')
    recipe_data["total_time"] = totaltime.get_text(strip=True) if totaltime else "Time Not Found"

    # Extracting recipe image URL (Week 2:  Extracting specific elements)
    image = soup.find('img', class_='ssrcss-11yxrdo-Image')
    recipe_data["image"] = image['src'] if image and 'src' in image.attrs else "Image Not Found"

    # Extracting ingredients list (Week 2: Extracting lists)
    ingredients = soup.select('ul.ssrcss-1ynsflq-UnorderedList li')
    recipe_data["ingredients"] = ', '.join([item.get_text(strip=True) for item in ingredients]) if ingredients else "Ingredients Not Found"

    # Extracting cuisine (if available)
    cuisine = soup.select_one('div.ssrcss-1if1xbo-TagList')
    recipe_data["cuisine"] = cuisine.get_text(strip=True) if cuisine else "Not Listed"

    # Week 5: Data Pre-processing - Handling structured data (JSON)
    # Extracting category from JSON-LD metadata
    recipe_data["category"] = "Not Found (No JSON-LD detected)"
    category = soup.find("script", type="application/ld+json")

    if category:
        try:
            data = json.loads(category.string)

            # First try @graph array (Week 5: Nested data structures)
            if "@graph" in data and isinstance(data["@graph"], list):
                for item in data["@graph"]:
                    if "recipeCategory" in item:
                        recipe_data["category"] = item["recipeCategory"]
                        break

            # Then try direct recipeCategory (Week 5: Direct field access)
            if recipe_data["category"] == "Not Found (No JSON-LD detected)" and "recipeCategory" in data:
                if isinstance(data["recipeCategory"], list):
                    recipe_data["category"] = ', '.join(data["recipeCategory"])
                else:
                    recipe_data["category"] = data["recipeCategory"]

            # Additional fallback (Week 5:  Alternative data paths)
            if recipe_data["category"] == "Not Found (No JSON-LD detected)":
                about = data.get("@graph", [{}])[0].get("about", {})
                if isinstance(about, dict) and "name" in about:
                    recipe_data["category"] = about["name"]

        except json.JSONDecodeError as e:
            print(f"JSON decode error for {url}: {e}")
            recipe_data["category"] = "Not Found (Invalid JSON)"
        except Exception as e:
            print(f"Unexpected error parsing JSON from {url}: {e}")
            recipe_data["category"] = "Not Found (Parse Error)"

    # Extracting dietary information (Week 2: Ethical data collection)
    diet = soup.select('dd a.ssrcss-jx182a-InlineLink')
    diet_info = [d.get_text(strip=True) for d in diet] if diet else ["Diet Info Not Found"]
    recipe_data["diet"] = ', '.join(diet_info)
    recipe_data["vegan"] = 'Vegan' in diet_info
    recipe_data["vegetarian"] = 'Vegetarian' in diet_info

    # Extracting rating value (Week 5:  Data validation)
    rating = soup.select_one('div[data-testid="recipe-rating"] span:nth-of-type(1)')
    recipe_data["rating_val"] = "N/A"
    if rating:
        try:
            rating_text = rating.get_text(strip=True)
            # Using regex to extract numeric rating (Week 5: Data cleaning)
            match = re.search(r'(\d+(\.\d+)?)', rating_text)
            if match:
                rating_num = float(match.group(1))
                factor = 10 ** (math.floor(math.log10(abs(rating_num))))
                recipe_data["rating_val"] = round(rating_num / factor) * factor
        except Exception as e:
            print(f"Error parsing rating for {url}: {e}")

    # Extracting rating count (Week 5: Data extraction)
    rating_count = soup.select_one('div[data-testid="recipe-rating"] span:nth-of-type(3)')
    recipe_data["rating_count"] = "N/A"
    if rating_count:
        try:
            rating_count_text = rating_count.get_text(strip=True)
            recipe_data["rating_count"] = "".join(filter(str.isdigit, rating_count_text)) or "N/A"
        except Exception as e:
            print(f"Error parsing rating count for {url}: {e}")

    # Setting default cuisine (Week 5:  Handling missing values)
    recipe_data["cuisine"] = "Not Listed"

    # Storing source URL (Week 2:  Data provenance)
    recipe_data["url"] = url

    print(f"Scraped: {recipe_data['title']} | Category: {recipe_data['category']}")
    # Creating structured DataFrame (Week 5: Data Frames)
    return pd.DataFrame([recipe_data])[REQUIRED_COLUMNS]


# Test URLs including error case (Week 6: Testing and Test cases)
urls = [
    "https://www.bbc.co.uk/food/recipes/avocado_pasta_with_peas_31700",
    "https://www.bbc.co.uk/food/recipes/mulligatawny_soup_68949",
    "https://www.bbc.co.uk/food/recipes/mary_berrys_lasagne_al_16923",
    "https://www.bbc.co.uk/food/recipes/invalid_recipe"  # Negative test case for error handling 
]

# Processing URLs (Week 2: Batch processing)
all_dfs = []
for url in urls:
    df = collect_page_data(url)
    if df is not None:
        all_dfs.append(df)

# Combining results (Week 5: Data aggregation)
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    print("\nDataFrame Columns:", final_df.columns.tolist())
    # Week 5:  Data storage
    final_df.to_csv("bbc_recipes.csv", index=False, encoding='utf-8')
    print("\nRecipes saved to: bbc_recipes.csv")
    print(final_df)
else:
    print("No recipes were successfully scraped.")