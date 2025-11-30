from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from inference_with_XAI_package import InsiderThreatSystem

# --- Configuration ---
IF_PATH = 'models/unsupervised/iforest_model.pkl'
XGB_PATH = 'models/supervised/XGBClassifier_model.pkl'

# --- Initialize App & Model ---
app = FastAPI(title="Insider Threat Detection API", version="1.0")

print("Initializing System...")
if os.path.exists(IF_PATH) and os.path.exists(XGB_PATH):
    # Initialize the engine once on startup
    engine = InsiderThreatSystem(IF_PATH, XGB_PATH)
else:
    print("CRITICAL ERROR: Models not found. Check paths.")
    engine = None

# --- Endpoints ---

@app.get("/")
def health_check():
    """Simple check to see if API is running."""
    if engine:
        return {"status": "online", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}

@app.post("/analyze")
def analyze_user(data: dict):
    """
    Takes a JSON object representing user logs.
    Returns Risk Score (0-100), Anomaly Score (0-100), and Explanation.
    """
    if not engine:
        raise HTTPException(status_code=500, detail="Models not loaded on server.")
    
    # Run the unified pipeline
    result = engine.analyze_user(data)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result

# --- Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)