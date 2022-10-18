from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

from Immo_eliza.preprocessing.cleaning_data import preprocess

app = FastAPI()

class ScoringItem(BaseModel):
    area: int
    property_type: str #"APARTMENT" | "HOUSE" | "OTHERS",
    rooms_number: int
    zip_code: int
    land_area: int | None #Optional[int]
    garden: bool | None #Optional[bool],
    garden_area: int | None #Optional[int],
    equipped_kitchen: bool | None #Optional[bool],
    full_address: str | None #Optional[str],
    swimming_pool: bool | None #Optional[bool],
    furnished: bool | None #Optional[bool],
    open_fire: bool | None #Optional[bool],
    terrace: bool | None #Optional[bool],
    terrace_area: int | None #Optional[int],
    facades_number: int | None #Optional[int],
    building_state: str | None # Optional["NEW" | "GOOD" | "TO RENOVATE" | "JUST RENOVATED" | "TO REBUILD"]
    
class Input(BaseModel):
    data: ScoringItem

with open("../model/immo_scaler.pkl","rb") as scalefile:
    scaler = pickle.load(scalefile)

with open("../model/immo_model.pkl","rb") as modelfile:
    model = pickle.load(modelfile)

    
@app.post('/')
async def scoring_endpoint(item: Input):
    return (preprocess(item))

# def predict(clean_data):
#     """
#     Function that takes immo_eliza preprocessed data as an input and return a price as output.
#     :input
#     :output
#     """
#     clean_data = 
#     output_price_prediction = {
#   "prediction": Optional[float],
#   "status_code": Optional[int]
# }
#     return output_price_prediction
