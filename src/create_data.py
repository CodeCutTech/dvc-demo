
import pandas as pd

# Create version 2 of dataset
df_v2 = pd.DataFrame({"feature": [10, 20, 30], "target": [1, 0, 1]})
df_v2.to_csv("data/test/data.csv", index=False)
