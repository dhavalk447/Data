import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Netflix movie dataset
# Replace with the actual path if you have a CSV file, e.g. 'netflix_movies.csv'
# For now, let's simulate the dataset loading process
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/netflix_titles.csv'
data = pd.read_csv(url)

# Step 2: Data Exploration
# Display first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Data Types
print("\nData types:")
print(data.dtypes)

# Step 3: Data Cleaning
# Handle missing values - Dropping rows with missing 'rating' and 'release_year'
data_cleaned = data.dropna(subset=['rating', 'release_year'])

# Convert 'release_year' to integer if it's not
data_cleaned['release_year'] = data_cleaned['release_year'].astype(int)

# Fill missing 'director' and 'cast' with 'Unknown' if desired
data_cleaned['director'].fillna('Unknown', inplace=True)
data_cleaned['cast'].fillna('Unknown', inplace=True)

# Step 4: Data Visualization
# 4.1: Distribution of movie ratings
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=data_cleaned, palette='Set2')
plt.title('Distribution of Movie Ratings')
plt.xticks(rotation=45)
plt.show()

# 4.2: Distribution of movie release years
plt.figure(figsize=(12, 6))
sns.histplot(data_cleaned['release_year'], bins=30, kde=True, color='purple')
plt.title('Distribution of Movie Release Years')
plt.xlabel('Release Year')
plt.ylabel('Frequency')
plt.show()

# 4.3: Top 10 most popular genres (by count of movies)
# Assuming genre is a string or list of genres separated by commas
data_cleaned['genres'] = data_cleaned['listed_in'].str.split(', ')
genres = data_cleaned.explode('genres')['genres'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=genres.index, y=genres.values, palette='Blues')
plt.title('Top 10 Most Popular Movie Genres')
plt.xticks(rotation=45)
plt.show()

# 4.4: Movies released per year
movies_per_year = data_cleaned.groupby('release_year').size()

plt.figure(figsize=(12, 6))
sns.lineplot(x=movies_per_year.index, y=movies_per_year.values, marker='o', color='teal')
plt.title('Number of Movies Released per Year')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.show()

# Step 5: Advanced Analysis
# 5.1: Correlation between duration and rating (scatter plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration', y='rating', data=data_cleaned)
plt.title('Duration vs Rating')
plt.xlabel('Duration (minutes)')
plt.ylabel('Rating')
plt.show()

# 5.2: Most popular directors (based on number of movies directed)
top_directors = data_cleaned['director'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_directors.index, y=top_directors.values, palette='viridis')
plt.title('Top 10 Directors with Most Movies')
plt.xticks(rotation=45)
plt.show()

# Step 6: Analyzing Netflix Originals vs Non-Originals
# Create a new column to identify Netflix Originals
data_cleaned['is_original'] = data_cleaned['title'].str.contains('Netflix', case=False, na=False)

# Plot the count of Netflix Originals vs Non-Originals
plt.figure(figsize=(10, 6))
sns.countplot(x='is_original', data=data_cleaned, palette='pastel')
plt.title('Count of Netflix Originals vs Non-Originals')
plt.xticks([0, 1], ['Non-Original', 'Original'])
plt.show()
