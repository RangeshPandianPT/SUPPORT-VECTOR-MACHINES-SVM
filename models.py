from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
import datetime

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    prediction = Column(String, index=True)
    confidence = Column(Float)
    
    # Store top 2 features for simplicity
    top_feature_1_name = Column(String, nullable=True)
    top_feature_1_val = Column(Float, nullable=True)
    top_feature_2_name = Column(String, nullable=True)
    top_feature_2_val = Column(Float, nullable=True)
