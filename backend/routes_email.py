"""
Contact / Email support endpoint using Gmail SMTP.

Requires environment variables:
  GMAIL_ADDRESS      — sender Gmail address
  GMAIL_APP_PASSWORD — 16-char app password (requires 2FA on the Gmail account)
  SUPPORT_EMAIL      — destination for contact submissions
"""

import os
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient

from clerk_auth import get_current_user
from models import ClerkUser

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Contact"])

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL")
MONGO_URI = os.getenv("MONGO_URI")


def _contact_col():
    if not MONGO_URI:
        return None
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client["stratos_orders"]["contact_submissions"]


class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str


class ContactResponse(BaseModel):
    success: bool
    message: str


@router.post("/contact", response_model=ContactResponse)
async def send_contact(request: ContactRequest, user: ClerkUser = Depends(get_current_user)):
    """Send a contact/support email via Gmail SMTP."""
    if not all([GMAIL_ADDRESS, GMAIL_APP_PASSWORD, SUPPORT_EMAIL]):
        raise HTTPException(
            status_code=503,
            detail="Email service not configured. Missing GMAIL_ADDRESS, GMAIL_APP_PASSWORD, or SUPPORT_EMAIL.",
        )

    # Build email
    msg = MIMEMultipart("alternative")
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = SUPPORT_EMAIL
    msg["Subject"] = f"[Stratos Support] {request.subject}"
    msg["Reply-To"] = request.email

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px;">
        <h2 style="color: #1e40af;">New Support Request</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 8px; font-weight: bold; color: #64748b;">From</td>
                <td style="padding: 8px;">{request.name}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold; color: #64748b;">Email</td>
                <td style="padding: 8px;">{request.email}</td></tr>
            <tr><td style="padding: 8px; font-weight: bold; color: #64748b;">User ID</td>
                <td style="padding: 8px;">{user.user_id}</td></tr>
        </table>
        <hr style="border: 1px solid #e2e8f0; margin: 16px 0;">
        <div style="padding: 12px; background: #f8fafc; border-radius: 8px;">
            <p style="white-space: pre-wrap; color: #334155;">{request.message}</p>
        </div>
        <p style="color: #94a3b8; font-size: 12px; margin-top: 24px;">
            Sent via Stratos AI Support System
        </p>
    </div>
    """
    msg.attach(MIMEText(html_body, "html"))

    # Send via Gmail SMTP
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        log.info(f"Support email sent from {request.email} re: {request.subject}")
    except smtplib.SMTPAuthenticationError:
        log.error("Gmail SMTP authentication failed — check GMAIL_APP_PASSWORD")
        raise HTTPException(status_code=503, detail="Email service authentication failed")
    except Exception as e:
        log.error(f"Failed to send email: {e}")
        raise HTTPException(status_code=503, detail="Failed to send email")

    # Store in MongoDB for record-keeping
    try:
        col = _contact_col()
        if col is not None:
            col.insert_one({
                "name": request.name,
                "email": request.email,
                "subject": request.subject,
                "message": request.message,
                "user_id": user.user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
    except Exception as e:
        log.warning(f"Could not store contact submission: {e}")

    return ContactResponse(success=True, message="Your message has been sent. We'll get back to you soon!")
