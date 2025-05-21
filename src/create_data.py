
import pandas as pd

# Create version 1 of dataset
df_v1 = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
df_v1.to_csv("data/test/data.csv", index=False)
