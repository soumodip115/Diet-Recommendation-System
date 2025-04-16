from fastapi import FastAPI
from pydantic import BaseModel,conlist
from typing import List,Optional
import pandas as pd
from model import recommend,output_recommended_recipes


dataset=pd.read_csv('../Data/dataset.csv',compression='gzip')

app = FastAPI()


class params(BaseModel):
    n_neighbors:int=5
    return_distance:bool=False

class PredictionIn(BaseModel):
    nutrition_input:conlist(float, min_length=9, max_length=9)
    ingredients:list[str]=[]
    params:Optional[params]


class Recipe(BaseModel):
    Name:str
    CookTime:str
    PrepTime:str
    TotalTime:str
    RecipeIngredientParts:list[str]
    Calories:float
    FatContent:float
    SaturatedFatContent:float
    CholesterolContent:float
    SodiumContent:float
    CarbohydrateContent:float
    FiberContent:float
    SugarContent:float
    ProteinContent:float
    RecipeInstructions:list[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None



# Ensure that the numerical fields are cast to strings when generating the response
def output_recommended_recipes(recommendation_dataframe):
    output = []
    for _, row in recommendation_dataframe.iterrows():
        cleaned_ingredients = [ingredient.strip(' "c()') for ingredient in row["RecipeIngredientParts"].split(",")]
        cleaned_instructions = [instruction.strip(' "c()') for instruction in row["RecipeInstructions"].split(",")]
        recipe = Recipe(
            Name=row["Name"],
            CookTime=str(row["CookTime"]),  # Convert to string
            PrepTime=str(row["PrepTime"]),  # Convert to string
            TotalTime=str(row["TotalTime"]),  # Convert to string
            RecipeIngredientParts=row["RecipeIngredientParts"].split(","),
            Calories=row["Calories"],
            FatContent=row["FatContent"],
            SaturatedFatContent=row["SaturatedFatContent"],
            CholesterolContent=row["CholesterolContent"],
            SodiumContent=row["SodiumContent"],
            CarbohydrateContent=row["CarbohydrateContent"],
            FiberContent=row["FiberContent"],
            SugarContent=row["SugarContent"],
            ProteinContent=row["ProteinContent"],
            RecipeInstructions=row["RecipeInstructions"].split(" , ")
        )

        # Append each recipe individually to the output list
        output.append(recipe)

    # Returning each recipe as a separate line in the response
    return (output)




@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict/",response_model=PredictionOut)
def update_item(prediction_input:PredictionIn):
    recommendation_dataframe=recommend(dataset,prediction_input.nutrition_input,prediction_input.ingredients,prediction_input.params.dict())
    output=output_recommended_recipes(recommendation_dataframe)
    if output is None:
        return {"output":None}
    else:
        return {"output":output}

