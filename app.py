# In your app.py file, create an API that contains:

# A route at / that accept:
# GET request and return "alive" if the server is alive.
# A route at /predict that accept:
# POST request that receives the data of a house in JSON format.
# GET request returning a string to explain what the POST expect (data and format).

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

from preprocessing.cleaning_data import preprocess

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
    

with open("model/immo_scaler.pkl","rb") as scalefile:
    scaler = pickle.load(scalefile)
    
with open("model/immo_poly_features.pkl","rb") as polyfeaturesfile:
    poly_features = pickle.load(polyfeaturesfile)

with open("model/immo_model.pkl","rb") as modelfile:
    poly_model = pickle.load(modelfile)

    
@app.post('/')
async def scoring_endpoint(item: Input):
    preprocess_input = preprocess(item)
    array_input = np.array([preprocess_input])
    X_scaled_imput = scaler.transform(array_input)
    input_predict = poly_model.predict(poly_features.fit_transform(X_scaled_imput))
    # return item
    return float(input_predict)
        

    


# @app.post('/')
# async def scoring_endpoint(item:ScoringItem):
#     df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
#     yhat = model.predict(df)
#     status_code = "?"
#     return {"prediction": int(yhat), "status_code": str(status_code)}