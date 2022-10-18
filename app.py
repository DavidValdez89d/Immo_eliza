# In your app.py file, create an API that contains:

# A route at / that accept:
# GET request and return "alive" if the server is alive.
# A route at /predict that accept:
# POST request that receives the data of a house in JSON format.
# GET request returning a string to explain what the POST expect (data and format).

from fastapi import FastAPI
from pydantic import BaseModel
import pickle


app = FastAPI()

class ScoringItem(BaseModel):
    area: int
    property_type: str #"APARTMENT" | "HOUSE" | "OTHERS",
    rooms_number: int
    zip_code: int
    land_area:int #Optional[int],
    garden: bool #Optional[bool],
    garden_area: int #Optional[int],
    equipped_kitchen: bool #Optional[bool],
    full_address: str #Optional[str],
    swimming_pool: bool #Optional[bool],
    furnished: bool #Optional[bool],
    open_fire: bool #Optional[bool],
    terrace: bool #Optional[bool],
    terrace_area: int #Optional[int],
    facades_number: int #Optional[int],
    building_state: str #Optional["NEW" | "GOOD" | "TO RENOVATE" | "JUST RENOVATED" | "TO REBUILD"

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    status_code = "?"
    return {"prediction": int(yhat), "status_code": str(status_code)}