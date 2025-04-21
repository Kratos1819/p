import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Generate synthetic wine data
np.random.seed(42)
df = pd.DataFrame({
    'alcohol': np.random.normal(10, 2, 1000),
    'acidity': np.random.normal(5, 1, 1000),
    'pH': np.random.normal(3.2, 0.3, 1000),
    'quality': np.random.randint(3, 10, 1000)
})

# Feature statistics
features = ['alcohol', 'acidity', 'pH']
stats = df[features].agg(['mean', 'std']).T
print(stats)

# Plot Gaussian fits
for f in features:
    sns.histplot(df[f], bins=30, stat="density", kde=True, color='skyblue', alpha=0.6)
    x = np.linspace(df[f].min(), df[f].max(), 100)
    plt.plot(x, norm.pdf(x, stats.loc[f, 'mean'], stats.loc[f, 'std']), 'r')
    plt.title(f'Gaussian Fit - {f}')
    plt.show()

# Correlation with quality
print("\nCorrelation with Quality:")
print(df.corr()['quality'].drop('quality'))