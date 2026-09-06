from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scoring import load_and_score

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/villages")
def get_villages():
    df = load_and_score()
    return df.to_dict(orient="records")

@app.get("/api/worklist")
def get_worklist():
    df = load_and_score()
    top = df[["village", "hazard_score", "relocation_score", "recommended_action", "high_population_flag"]]
    return top.to_dict(orient="records")

@app.get("/api/villages/{name}")
def get_village_detail(name: str):
    df = load_and_score()
    row = df[df["village"] == name]
    if row.empty:
        return {"error": "village not found"}
    return row.to_dict(orient="records")[0]