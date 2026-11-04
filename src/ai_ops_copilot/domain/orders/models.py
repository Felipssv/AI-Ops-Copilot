from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field

class Region(StrEnum):    
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    CENTRAL = "central"

class OrderCategory(StrEnum):
    RESTAURANT = "restaurant"
    FAST_FOOD = "fast_food"
    MARKET = "market"
    PHARMACY = "pharmacy"
    BAKERY = "bakery"
    

VALID_REGIONS: frozenset[str] = frozenset(r.value for r in Region)

VALID_ORDER_CATEGORIES: frozenset[str] = frozenset(c.value for c in OrderCategory)

REQUIRED_ORDER_COLUMNS: tuple[str, ...] =  (
    "order_id",
    "created_at",
    "region",
    "order_category",
    "order_value",
    "estimated_prep_minutes",
    "estimated_delivery_minutes",
    "actual_prep_minutes",
    "actual_delivery_minutes",
    "is_late",
)

ALLOWED_ML_FEATURES: tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "region", 
    "order_category", 
    "order_value", 
    "estimated_prep_minutes", 
    "estimated_delivery_minutes",
)

FORBIDDEN_LEAKAGE_COLUMNS: frozenset[str] = frozenset({
    "order_id", 
    "created_at", 
    "actual_prep_minutes", 
    "actual_delivery_minutes", 
    "actual_total_minutes", 
    "delay_minutes",
    "is_late"
})

MIN_ORDER_VALUE = 0.01
MAX_ORDER_VALUE = 10000.0

MIN_ESTIMATED_MINUTES = 1
MAX_ESTIMATED_MINUTES = 180

MIN_ACTUAL_MINUTES = 1
MAX_ACTUAL_MINUTES = 360

DEFAULT_LATE_TOLERANCE_MINUTES = 5

def compute_promised_minutes(estimated_prep_minutes: int, estimated_delivery_minutes: int) -> int:
    return estimated_prep_minutes + estimated_delivery_minutes

def compute_actual_minutes(actual_prep_minutes: int, actual_delivery_minutes: int) -> int:
    return actual_prep_minutes + actual_delivery_minutes

def evaluate_is_late(estimated_prep_minutes: int,estimated_delivery_minutes: int, actual_prep_minutes: int, actual_delivery_minutes: int, late_tolerance_minutes: int = DEFAULT_LATE_TOLERANCE_MINUTES) -> bool:
    prep = compute_promised_minutes(estimated_prep_minutes, estimated_delivery_minutes)
    actual = compute_actual_minutes(actual_prep_minutes, actual_delivery_minutes)

    return actual > (prep + late_tolerance_minutes)

class OrderPredictionFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    region: Region
    order_category: OrderCategory
    order_value: float = Field(..., gt=0.0, le=MAX_ORDER_VALUE)
    estimated_prep_minutes: int = Field(..., ge=MIN_ESTIMATED_MINUTES, le=MAX_ESTIMATED_MINUTES)
    estimated_delivery_minutes: int = Field(..., ge=MIN_ESTIMATED_MINUTES, le=MAX_ESTIMATED_MINUTES)

    @property
    def is_weekend(self) -> bool:
        return self.day_of_week in (5,6)
    