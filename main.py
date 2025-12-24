# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_openml

# Load the Iris dataset
iris = fetch_openml(name='iris', version=1, as_frame=True)
df = iris.frame

# Display available features
features = list(df.columns)
print("Available features:", features)

# Select a few columns
selected_features = ['sepallength', 'petalwidth', 'class']
print("Selected features:", selected_features)

# -------------------------------
# Histogram Visualization
# -------------------------------
fig, axs = plt.subplots(1, len(selected_features), figsize=(20, 3))

for ax, feature in zip(axs, selected_features):
    ax.hist(df[feature], bins=5, color='skyblue', edgecolor='black')
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")

plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

reference_feature = 'class'
y = df[reference_feature]

fig, axs = plt.subplots(1, len(selected_features) - 1, figsize=(15, 3))

for ax, feature in zip(axs, selected_features[:-1]):
    ax.scatter(df[feature], y, alpha=0.6)
    ax.set_xlabel(feature)
    ax.set_ylabel(reference_feature)

plt.suptitle("Scatter Plots: Feature vs Class")
plt.tight_layout()
plt.show()

reference_feature = 'class'
comparison_feature = 'petalwidth'

plt.figure(figsize=(8, 6))
plt.scatter(df[reference_feature], df[comparison_feature], alpha=0.6)
plt.xlabel(reference_feature)
plt.ylabel(comparison_feature)
plt.title("Class vs Petal Width")

# Save the plot
plt.savefig("correlation_plot.png")
plt.show()
