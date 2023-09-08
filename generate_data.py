import pandas as pd
import numpy as np

np.random.seed(0)
data = {
    'feature_1': np.random.rand(100),
    'feature_2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)
