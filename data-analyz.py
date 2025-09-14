# ==========================================
# Assignment: Analyzing Data with Pandas and Visualizing Results with Matplotlib
# ==========================================

# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# -------------------------------
# Task 1: Load and Explore the Dataset
# -------------------------------

try:
    # Load Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame   # Converts sklearn Bunch object to pandas DataFrame
    
    print("Dataset loaded successfully!\n")
    
    # Display first few rows
    print("First 5 rows of dataset:")
    print(df.head(), "\n")
    
    # Check data types and missing values
    print("Dataset Info:")
    print(df.info(), "\n")
    
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")
    
    # Clean data (in this case, Iris dataset has no missing values)
    df = df.dropna()
    
except FileNotFoundError:
    print("File not found. Please check dataset path.")
except Exception as e:
    print(f"An error occurred: {e}")

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

# Basic statistics
print("Descriptive Statistics:")
print(df.describe(), "\n")

# Grouping: average measurement per species
grouped = df.groupby("target").mean()
print("Average measurements per species:")
print(grouped, "\n")

# Mapping numeric target to species name for readability
df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# Interesting finding: which species has the longest average petal length?
max_species = grouped["petal length (cm)"].idxmax()
print(f"Interesting Finding: Species with the longest average petal length is {iris.target_names[max_species]}.\n")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

plt.style.use("seaborn-v0_8")  # for better visuals

# 1. Line Chart (example: sepal length trend across samples)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Sepal Length Trend Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart (average petal length per species)
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator="mean", palette="viridis")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (sepal length vs petal length, colored by species)
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="deep")
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

print("All tasks completed with analysis and visualizations!")