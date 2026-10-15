from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TreatmentGuidance(BaseModel):
    crop: str
    disease: str
    overview: str = ""
    symptoms: List[str] = Field(default_factory=list)
    prevention: List[str] = Field(default_factory=list)
    non_chemical_control: List[str] = Field(default_factory=list)
    verified_treatment: List[str] = Field(default_factory=list)
    safety_precautions: List[str] = Field(default_factory=list)
    source_url: str = ""
    last_verified_at: str = ""
    verification_status: str = "unverified"


class ClassScore(BaseModel):
    label: str
    score: float


class PredictionResponse(BaseModel):
    crop: str
    predicted_disease: str
    confidence: Optional[float] = None
    status: str = "preliminary"
    model_name: str
    overview: str = ""
    treatment_guidance: Optional[TreatmentGuidance] = None
    source_references: List[str] = Field(default_factory=list)
    human_review_required: bool = True
    top_predictions: List[ClassScore] = Field(default_factory=list)


class CropInfo(BaseModel):
    name: str
    display_name: str
    emoji: str
    model_id: str
    num_classes: int
    is_verified: bool
    notes: str = ""


class SupportedCropsResponse(BaseModel):
    crops: List[CropInfo]


class CropClassesResponse(BaseModel):
    crop: str
    model_id: str
    num_classes: int
    classes: List[str]
