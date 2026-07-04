# Notebooks

Optional exploratory notebooks for GameLens AI.

The main app does **not** depend on any notebook - everything needed to run
the project lives in `app.py` and `src/`. This folder is a place for
exploratory data analysis (EDA) if you want to dig into the generated data.

## Suggested notebook: `tracking_eda.ipynb`

Ideas for what to explore once you have run the app at least once (so that
`outputs/tracking_data.csv` exists):

1. Load `outputs/tracking_data.csv` with pandas.
2. Inspect how many rows, unique `track_id`s, and frames were captured.
3. Plot movement distribution per player.
4. Check how stable track IDs are over time (ID switches).
5. Experiment with different zone definitions or movement thresholds.

```python
import pandas as pd

df = pd.read_csv("../outputs/tracking_data.csv")
print(df.shape)
print(df["track_id"].nunique(), "unique players")
df.head()
```
