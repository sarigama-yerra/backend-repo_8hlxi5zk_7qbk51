import os
import base64
from datetime import datetime, timedelta, timezone, date
from typing import Optional, List, Literal

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import Subscription, Transaction, UploadedFile, Action, User

# Optional OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None

app = FastAPI(title="AI Personal Finance Autopilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change")
JWT_EXP_MIN = int(os.getenv("JWT_EXP_MIN", "60"))


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXP_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AnalyzeTextRequest(BaseModel):
    raw_text: str


def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("sub")
        email = payload.get("email")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db["user"].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        user["id"] = str(user["_id"])
        user.pop("_id", None)
        user.pop("password", None)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authorization")


@app.get("/")
def root():
    return {"message": "AI Personal Finance Autopilot API"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "collections": [],
        "openai": "✅ Configured" if OPENAI_API_KEY else "❌ Not Configured",
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"⚠️ {str(e)[:100]}"
    return response


# Auth endpoints
@app.post("/api/auth/register")
def register(req: RegisterRequest):
    existing = db["user"].find_one({"email": req.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_doc = {
        "email": req.email.lower(),
        "name": req.name,
        "password": hash_password(req.password),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = db["user"].insert_one(user_doc)
    token = create_token(str(result.inserted_id), req.email.lower())
    return {"token": token}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = db["user"].find_one({"email": req.email.lower()})
    if not user or not verify_password(req.password, user.get("password", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(str(user["_id"]), user["email"])
    return {"token": token}


@app.get("/api/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return current_user


# Subscriptions CRUD
@app.get("/api/subscriptions")
def list_subscriptions(current_user: dict = Depends(get_current_user)):
    docs = db["subscription"].find({"user_id": current_user["id"]}).sort("created_at", -1)
    items = []
    for d in docs:
        d["id"] = str(d["_id"]) ; d.pop("_id", None)
        items.append(d)
    return items


@app.post("/api/subscriptions")
def create_subscription(sub: Subscription, current_user: dict = Depends(get_current_user)):
    data = sub.model_dump()
    data["user_id"] = current_user["id"]
    data["created_at"] = datetime.now(timezone.utc)
    data["updated_at"] = datetime.now(timezone.utc)
    res = db["subscription"].insert_one(data)
    data["id"] = str(res.inserted_id)
    return data


@app.put("/api/subscriptions/{subscription_id}")
def update_subscription(subscription_id: str, sub: Subscription, current_user: dict = Depends(get_current_user)):
    data = sub.model_dump()
    data["user_id"] = current_user["id"]
    data["updated_at"] = datetime.now(timezone.utc)
    res = db["subscription"].update_one({"_id": ObjectId(subscription_id), "user_id": current_user["id"]}, {"$set": data})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Subscription not found")
    d = db["subscription"].find_one({"_id": ObjectId(subscription_id)})
    d["id"] = str(d["_id"]) ; d.pop("_id", None)
    return d


@app.delete("/api/subscriptions/{subscription_id}")
def delete_subscription(subscription_id: str, current_user: dict = Depends(get_current_user)):
    res = db["subscription"].delete_one({"_id": ObjectId(subscription_id), "user_id": current_user["id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return {"success": True}


# Analyze text into transactions/subscriptions
@app.post("/api/analyze/text")
def analyze_text(req: AnalyzeTextRequest, current_user: dict = Depends(get_current_user)):
    raw = req.raw_text.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="raw_text is required")

    extracted = {
        "merchant_name": None,
        "amount": None,
        "currency": None,
        "billing_frequency": None,
        "first_charge_date": None,
        "next_billing_date": None,
        "is_recurring": None,
        "category": None,
    }

    if openai_client:
        try:
            sys_prompt = (
                "Extract structured billing data from the user text. Return a strict JSON with keys: "
                "merchant_name, amount (number), currency (ISO code), billing_frequency (weekly|monthly|quarterly|yearly|one_time), "
                "first_charge_date (YYYY-MM-DD or null), next_billing_date (YYYY-MM-DD or null), is_recurring (true/false), category."
            )
            user_msg = f"Text:\n{raw}\n\nReturn JSON only."
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
            )
            content = resp.choices[0].message.content
            import json
            extracted = json.loads(content)
        except Exception:
            # fallback heuristic
            pass

    # Heuristic fallbacks
    if extracted.get("currency") is None:
        for code in ["USD","EUR","GBP","JPY","AUD","CAD","INR"]:
            if code in raw:
                extracted["currency"] = code
                break
    if extracted.get("amount") is None:
        import re
        m = re.search(r"(\d+[\.,]?\d*)", raw)
        if m:
            try:
                extracted["amount"] = float(m.group(1).replace(",",""))
            except Exception:
                pass

    tx = {
        "user_id": current_user["id"],
        "merchant_name": extracted.get("merchant_name"),
        "amount": extracted.get("amount"),
        "currency": extracted.get("currency"),
        "billing_frequency": extracted.get("billing_frequency"),
        "first_charge_date": extracted.get("first_charge_date"),
        "next_billing_date": extracted.get("next_billing_date"),
        "is_recurring": extracted.get("is_recurring"),
        "category": extracted.get("category"),
        "raw_text": raw,
        "source": "text",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    tx_id = db["transaction"].insert_one(tx).inserted_id

    # Upsert subscription if recurring
    sub_doc = None
    if extracted.get("is_recurring") and extracted.get("merchant_name") and extracted.get("amount"):
        sub_query = {"user_id": current_user["id"], "merchant_name": extracted["merchant_name"], "status": {"$ne": "canceled"}}
        existing = db["subscription"].find_one(sub_query)
        payload = {
            "user_id": current_user["id"],
            "merchant_name": extracted.get("merchant_name"),
            "plan_name": None,
            "amount": extracted.get("amount"),
            "currency": extracted.get("currency") or "USD",
            "billing_frequency": extracted.get("billing_frequency") or "monthly",
            "next_billing_date": extracted.get("next_billing_date"),
            "category": extracted.get("category"),
            "status": "active",
            "updated_at": datetime.now(timezone.utc),
        }
        if existing:
            db["subscription"].update_one({"_id": existing["_id"]}, {"$set": payload})
            sub_doc = db["subscription"].find_one({"_id": existing["_id"]})
        else:
            payload["created_at"] = datetime.now(timezone.utc)
            sid = db["subscription"].insert_one(payload).inserted_id
            sub_doc = db["subscription"].find_one({"_id": sid})
        sub_doc["id"] = str(sub_doc["_id"]) ; sub_doc.pop("_id", None)

    tx_doc = db["transaction"].find_one({"_id": tx_id})
    tx_doc["id"] = str(tx_id) ; tx_doc.pop("_id", None)
    return {"transaction": tx_doc, "subscription": sub_doc}


# File upload and processing
@app.post("/api/uploads")
async def upload_file(
    file: UploadFile = File(...),
    file_type: Literal['screenshot','pdf','csv'] = Form(...),
    current_user: dict = Depends(get_current_user),
):
    content = await file.read()
    storage_dir = os.path.join("logs", "uploads")
    os.makedirs(storage_dir, exist_ok=True)
    filename = f"{datetime.now(timezone.utc).timestamp()}_{file.filename}"
    storage_path = os.path.join(storage_dir, filename)
    with open(storage_path, "wb") as f:
        f.write(content)

    file_doc = {
        "user_id": current_user["id"],
        "file_type": file_type,
        "original_filename": file.filename,
        "storage_path": storage_path,
        "status": "uploaded",
        "created_at": datetime.now(timezone.utc),
    }
    fid = db["uploadedfile"].insert_one(file_doc).inserted_id

    processed = []
    if file_type == 'csv':
        import csv, io
        reader = csv.DictReader(io.StringIO(content.decode(errors='ignore')))
        for row in reader:
            merchant = row.get('merchant') or row.get('description') or row.get('payee')
            amount = row.get('amount') or row.get('debit') or row.get('credit')
            currency = row.get('currency') or 'USD'
            try:
                amount_val = float(amount.replace(',', '')) if amount else None
            except Exception:
                amount_val = None
            tx = {
                "user_id": current_user["id"],
                "merchant_name": merchant,
                "amount": amount_val,
                "currency": currency,
                "source": "csv",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            db["transaction"].insert_one(tx)
            processed.append(tx)
        db["uploadedfile"].update_one({"_id": fid}, {"$set": {"status": "processed"}})
    elif file_type in ['screenshot'] and openai_client:
        try:
            b64 = base64.b64encode(content).decode()
            data_url = f"data:image/png;base64,{b64}"
            prompt = "Extract transactions (merchant, amount, currency, date, recurring yes/no) as JSON array."
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial data extractor. Return JSON only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ],
                temperature=0.1,
            )
            import json
            arr = json.loads(resp.choices[0].message.content)
            for it in arr:
                tx = {
                    "user_id": current_user["id"],
                    "merchant_name": it.get('merchant') or it.get('merchant_name'),
                    "amount": it.get('amount'),
                    "currency": it.get('currency') or 'USD',
                    "source": "screenshot",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                db["transaction"].insert_one(tx)
                processed.append(tx)
            db["uploadedfile"].update_one({"_id": fid}, {"$set": {"status": "processed"}})
        except Exception:
            db["uploadedfile"].update_one({"_id": fid}, {"$set": {"status": "failed"}})
    else:
        # mark as processing not supported yet
        db["uploadedfile"].update_one({"_id": fid}, {"$set": {"status": "processing"}})

    return {"file_id": str(fid), "processed": processed}


# Actions: cancel/negotiate email generation
class GenerateActionRequest(BaseModel):
    subscription_id: str
    type: Literal['cancel','negotiate']


@app.post("/api/actions/generate")
def generate_action(req: GenerateActionRequest, current_user: dict = Depends(get_current_user)):
    sub = db["subscription"].find_one({"_id": ObjectId(req.subscription_id), "user_id": current_user["id"]})
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")

    merchant = sub.get("merchant_name", "the service")
    amount = sub.get("amount")
    currency = sub.get("currency", "USD")

    base_prompt = (
        f"Write a short, clear, polite email to {merchant} to "
        + ("cancel my subscription effective immediately." if req.type == 'cancel' else "request a lower rate based on loyalty and budget.")
        + " Keep it under 150 words. Include account unspecified placeholders."
    )
    subject = ("Request to Cancel Subscription" if req.type == 'cancel' else f"Request to Reduce Subscription Rate")
    body = None

    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You write concise professional emails."},
                    {"role": "user", "content": base_prompt},
                ],
                temperature=0.3,
            )
            body = resp.choices[0].message.content
        except Exception:
            body = None

    if body is None:
        body = (
            f"Hello {merchant} Support,\n\n"
            + ("Please cancel my subscription effective immediately and confirm the cancellation." if req.type == 'cancel' else f"I would like to reduce my subscription price. Could you offer a lower rate? My current charge is {amount} {currency}.")
            + "\n\nAccount email: <your email>\nAccount ID: <your account id>\n\nThank you,\n<Your Name>"
        )

    action_doc = {
        "user_id": current_user["id"],
        "subscription_id": req.subscription_id,
        "type": req.type,
        "status": "draft",
        "email_subject": subject,
        "email_body": body,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    aid = db["action"].insert_one(action_doc).inserted_id
    action_doc["id"] = str(aid)
    return action_doc


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
