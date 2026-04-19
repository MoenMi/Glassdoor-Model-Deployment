from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request model for the /predict endpoint"""
    title: str = Field(default="", examples=["Great team, weak pay"])
    pros: str = Field(default="", examples=["Supportive coworkers and flexible schedule"])
    cons: str = Field(default="", examples=["Compensation below market and slow promotions"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Great people, mediocre pay",
                    "pros": "Good work life balance and nice manager",
                    "cons": "Salary is low and promotions are slow"
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Response model for the /predict endpoint"""
    prediction: int
    label: str
    probability_recommend: float | None = None
    probability_not_recommend: float | None = None
