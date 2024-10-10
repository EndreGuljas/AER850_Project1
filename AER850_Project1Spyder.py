import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#Step 1

df = pd.read_csv("data/Project_1_Data.csv")

#Step 2

#statistical analysis + histograms


print(df.head())
print(df.info())
print(df.describe())

x = df['X']        
y = df['Y']        
z = df['Z']  

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title("3D Scatter Plot")


# Plot the points
scatter = ax.scatter(x, y, z, c=z, s=20, cmap='viridis', alpha=0.7)

# Add labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a color bar for the points
fig.colorbar(scatter, ax=ax, label='Colour Scale',shrink=0.5, pad=0.15)

# Show the plot
plt.show()


#Step 3 correlation matrix

# try x vs y, x vs z, and y vs z

correlation_matrix = df.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()




