import httpx
from pydantic import BaseModel
from typing import List

# Define the models
class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False

class PredictionIn(BaseModel):
    nutrition_input: List[float]
    ingredients: List[str]
    params: Params

class Recipe(BaseModel):
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]

class PredictionOut(BaseModel):
    output: List[Recipe] = []

# Test function to send a request to the FastAPI application with an increased timeout
async def test_predict():
    async with httpx.AsyncClient(timeout=30.0) as client:  # Increase timeout to 30 seconds
        # Prepare the input data
        prediction_input = PredictionIn(
            ingredients=["tomato", "onion", "garlic"],
            nutrition_input=[50.0, 30.0, 10.0, 5.0, 2.0, 1.5, 2.0, 3.0, 1.2],
            params=Params(n_neighbors=3, return_distance=False).dict()
        )
        
        # Send the POST request to the FastAPI endpoint
        response = await client.post("http://127.0.0.1:8000/predict/", json=prediction_input.dict())

        # Check the response
        if response.status_code == 200:
            print("Response: ", response.json()['output'])  # Print the response JSON
        else:
            print(f"Error: {response.status_code}, {response.text}")

# Run the test
import asyncio
asyncio.run(test_predict())
