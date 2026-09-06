import pandas as pd
from pathlib import Path

def load_and_score(csv_path="data/villages.csv"):
    path = Path(csv_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    df = pd.read_csv(path, sep=r"\s+")

    if "land_change" not in df.columns:
        df["land_change"] = 0.0

    df["hazard_score"] = (
        0.40 * df["slope"] +
        0.35 * df["rainfall"] +
        0.25 * df["land_change"]
    )

    df["population_norm"] = (
        df["population"] / df["population"].max()
    )

    df["access_factor"] = 1 - df["access_score"]

    # Hazard weight badhaya (70 -> 85) taaki hazard hamesha
    # dominant rahe, chahe population extreme ho
    df["relocation_score"] = (
        0.85 * df["hazard_score"] +
        0.10 * df["population_norm"] +
        0.05 * df["access_factor"]
    )

    pop_threshold = df["population"].quantile(0.75)
    df["high_population_flag"] = df["population"] > pop_threshold

    def action_for(row):
        score = row["hazard_score"]
        pop = row["population"]

        if score > 0.75:
            action = "Immediate Verification"
        elif score > 0.55:
            action = "Relocation Assessment"
        elif score > 0.35:
            action = "Enhanced Monitoring"
        else:
            action = "Routine Monitoring"

        if pop > 250 and action == "Routine Monitoring":
            action = "Enhanced Monitoring"

        return action

    df["recommended_action"] = df.apply(action_for, axis=1)
    df = df.sort_values("relocation_score", ascending=False)
    return df

if __name__ == "__main__":
    result = load_and_score()
    print(result[[
        "village", "hazard_score", "relocation_score",
        "recommended_action", "high_population_flag"
    ]])

