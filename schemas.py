"""
Database Schemas for AI Personal Finance Autopilot

Each Pydantic model represents a collection in MongoDB. The collection name is the lowercase class name.
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field
from datetime import date, datetime

class User(BaseModel):
    email: str
    name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class Subscription(BaseModel):
    user_id: str
    merchant_name: str
    plan_name: Optional[str] = None
    amount: float
    currency: str = Field(..., description="ISO currency, e.g., USD, EUR")
    billing_frequency: Literal['weekly','monthly','quarterly','yearly','one_time']
    next_billing_date: Optional[date] = None
    category: Optional[str] = None
    status: Literal['active','paused','canceled'] = 'active'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class Transaction(BaseModel):
    user_id: str
    merchant_name: Optional[str] = None
    amount: float
    currency: str
    billing_frequency: Optional[str] = None
    first_charge_date: Optional[date] = None
    next_billing_date: Optional[date] = None
    is_recurring: Optional[bool] = None
    category: Optional[str] = None
    raw_text: Optional[str] = None
    source: Literal['manual','text','screenshot','pdf','csv'] = 'manual'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UploadedFile(BaseModel):
    user_id: str
    file_type: Literal['screenshot','pdf','csv']
    original_filename: str
    storage_path: str
    status: Literal['uploaded','processing','processed','failed'] = 'uploaded'
    created_at: Optional[datetime] = None

class Action(BaseModel):
    user_id: str
    subscription_id: str
    type: Literal['cancel','negotiate']
    status: Literal['draft','queued','sent','failed'] = 'draft'
    email_subject: Optional[str] = None
    email_body: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
