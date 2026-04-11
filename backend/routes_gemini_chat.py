"""
AI Assistant for inventory ordering & replenishment.

Completely independent from the existing Claude/LangGraph chat in routes_chat.py.
Uses Claude Haiku via Anthropic SDK with tool calling.
"""
import os
import json
import logging
import smtplib
import uuid
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import anthropic

from clerk_auth import get_current_user
from models import ClerkUser

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["AIChat"])

# ── MongoDB connections ───────────────────────────────────────────────────────

MONGO_URI = os.getenv("MONGO_URI")
_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI not set")
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def _inventory_col():
    return _get_client()["inventory_forecasting"]["inventory_snapshot"]


def _vendors_col():
    return _get_client()["stratos_orders"]["vendors"]


def _orders_col():
    return _get_client()["stratos_orders"]["orders"]


def _conversations_col():
    return _get_client()["stratos_orders"]["conversations"]


# ── Claude setup ─────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"

_anthropic_client: Optional[anthropic.Anthropic] = None


def _get_claude():
    global _anthropic_client
    if _anthropic_client is None:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'), override=True)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


SYSTEM_PROMPT = """You are Stratos AI, a concise inventory management assistant. Stores: S001-S005. Products: P0001-P0020.

RULES FOR PLACING ORDERS — FOLLOW EXACTLY:
1. You CANNOT call place_order unless the user has told you ALL THREE: store ID, product ID(s), and exact quantity for each product.
2. If the user says "place an order" but is missing any of those three, ASK for the missing info. Do NOT guess.
3. If quantity is missing, ask "How many units would you like to order?" — do NOT pick a number yourself.
4. Before calling place_order, show a summary and ask "Shall I confirm this order?" — WAIT for the user to say yes.
5. Only call place_order AFTER the user explicitly confirms.

NEVER auto-decide quantities. NEVER place orders without explicit user confirmation on every detail.

For other queries: use tools freely to check inventory, low stock, vendors, order history. Be concise. Use markdown formatting."""

# ── Tool definitions (Anthropic format) ───────────────────────────────────────

TOOLS = [
    {
        "name": "check_inventory",
        "description": "Check current stock levels for a store. Optionally filter by product ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Store ID (e.g. S001)"},
                "product_id": {"type": "string", "description": "Optional product ID (e.g. P0010)"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "get_low_stock_items",
        "description": "Find products with stock below a given threshold at a store.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Store ID"},
                "threshold": {"type": "integer", "description": "Stock threshold (default 50)"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "get_reorder_suggestions",
        "description": "Get reorder suggestions for a store based on low stock items and available vendors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Store ID"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "place_order",
        "description": "Place a purchase order to replenish inventory. ONLY call after user confirms store, products, quantities, AND says yes to the summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Store ID"},
                "items": {
                    "type": "array",
                    "description": "Items to order",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string", "description": "Product ID"},
                            "quantity": {"type": "integer", "description": "Quantity confirmed by user"},
                        },
                        "required": ["product_id", "quantity"],
                    },
                },
            },
            "required": ["store_id", "items"],
        },
    },
    {
        "name": "get_order_history",
        "description": "Get past orders for a store.",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_id": {"type": "string", "description": "Store ID"},
            },
            "required": ["store_id"],
        },
    },
    {
        "name": "list_vendors",
        "description": "List available vendors, optionally filtered by product category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Product category (e.g. Snacks, Beverages)"},
            },
        },
    },
]

# ── Function implementations ──────────────────────────────────────────────────


def _exec_check_inventory(store_id: str, product_id: Optional[str] = None) -> dict:
    query = {"Store ID": store_id}
    if product_id:
        query["Product ID"] = product_id
    projection = {"_id": 0, "Product ID": 1, "Category": 1, "Current Stock": 1}
    docs = list(_inventory_col().find(query, projection))
    if not docs:
        return {"error": "No data found for store %s%s" % (store_id, " product %s" % product_id if product_id else "")}
    items = [{"pid": d["Product ID"], "cat": d["Category"], "stock": d["Current Stock"]} for d in docs]
    return {"store": store_id, "items": items}


def _exec_get_low_stock_items(store_id: str, threshold: int = 50) -> dict:
    docs = list(_inventory_col().find(
        {"Store ID": store_id, "Current Stock": {"$lt": threshold}},
        {"_id": 0, "Product ID": 1, "Category": 1, "Current Stock": 1},
    ).sort("Current Stock", 1))
    items = [{"pid": d["Product ID"], "cat": d["Category"], "stock": d["Current Stock"]} for d in docs]
    return {"store": store_id, "threshold": threshold, "items": items, "count": len(items)}


def _exec_get_reorder_suggestions(store_id: str) -> dict:
    low_items = list(_inventory_col().find(
        {"Store ID": store_id, "Current Stock": {"$lt": 100}},
        {"_id": 0, "Product ID": 1, "Category": 1, "Current Stock": 1, "Lead Time Days": 1},
    ).sort("Current Stock", 1))
    suggestions = []
    for item in low_items:
        vendor = _vendors_col().find_one({"category": item["Category"]}, {"_id": 0, "name": 1, "min_order_qty": 1, "price_per_unit": 1})
        suggestions.append({
            "pid": item["Product ID"],
            "cat": item["Category"],
            "stock": item["Current Stock"],
            "vendor": vendor.get("name", "N/A") if vendor else "N/A",
            "min_qty": vendor.get("min_order_qty", 0) if vendor else 0,
            "price": vendor.get("price_per_unit", 0) if vendor else 0,
        })
    return {"store": store_id, "suggestions": suggestions, "count": len(suggestions)}


def _send_order_confirmation_email(order_id, store_id, items, total_units, total_cost, user_email=None):
    """Send order confirmation email via Gmail SMTP."""
    gmail_addr = os.getenv("GMAIL_ADDRESS")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD")
    if not all([gmail_addr, gmail_pass, user_email]):
        log.info("Skipping order email — Gmail not configured or no user email")
        return
    try:
        items_html = ""
        for item in items:
            items_html += (
                "<tr><td style='padding:8px;border:1px solid #e2e8f0;'>%s</td>"
                "<td style='padding:8px;border:1px solid #e2e8f0;'>%s</td>"
                "<td style='padding:8px;border:1px solid #e2e8f0;'>%d</td>"
                "<td style='padding:8px;border:1px solid #e2e8f0;'>%s</td>"
                "<td style='padding:8px;border:1px solid #e2e8f0;'>$%.2f</td>"
                "<td style='padding:8px;border:1px solid #e2e8f0;'>$%.2f</td></tr>"
            ) % (item['product_id'], item['category'], item['quantity'],
                 item['vendor_name'], item['unit_price'], item['quantity'] * item['unit_price'])

        html_body = """
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
            <div style="background:#1e40af;color:white;padding:20px;border-radius:8px 8px 0 0;">
                <h2 style="margin:0;">Order Confirmation</h2>
                <p style="margin:5px 0 0 0;opacity:0.9;">Stratos AI - Inventory Management</p>
            </div>
            <div style="padding:20px;background:#f8fafc;border:1px solid #e2e8f0;border-top:none;border-radius:0 0 8px 8px;">
                <p>Your order has been placed and inventory updated.</p>
                <table style="width:100%%;margin:16px 0;">
                    <tr><td style="padding:4px 0;font-weight:bold;color:#64748b;">Order ID</td><td>%s</td></tr>
                    <tr><td style="padding:4px 0;font-weight:bold;color:#64748b;">Store</td><td>%s</td></tr>
                    <tr><td style="padding:4px 0;font-weight:bold;color:#64748b;">Total Units</td><td>%d</td></tr>
                    <tr><td style="padding:4px 0;font-weight:bold;color:#64748b;">Total Cost</td><td style="font-weight:bold;color:#1e40af;">$%.2f</td></tr>
                </table>
                <h3 style="color:#334155;margin-top:20px;">Items</h3>
                <table style="width:100%%;border-collapse:collapse;margin-top:8px;">
                    <tr style="background:#1e40af;color:white;">
                        <th style="padding:8px;text-align:left;">Product</th>
                        <th style="padding:8px;text-align:left;">Category</th>
                        <th style="padding:8px;text-align:left;">Qty</th>
                        <th style="padding:8px;text-align:left;">Vendor</th>
                        <th style="padding:8px;text-align:left;">Unit Price</th>
                        <th style="padding:8px;text-align:left;">Subtotal</th>
                    </tr>%s
                </table>
                <p style="color:#94a3b8;font-size:12px;margin-top:24px;">Automated confirmation from Stratos AI.</p>
            </div>
        </div>""" % (order_id, store_id, total_units, total_cost, items_html)

        msg = MIMEMultipart("alternative")
        msg["From"] = gmail_addr
        msg["To"] = user_email
        msg["Subject"] = "[Stratos] Order Confirmed - %s for Store %s" % (order_id, store_id)
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(gmail_addr, gmail_pass)
            server.send_message(msg)
        log.info("Order email sent to %s for %s", user_email, order_id)
    except Exception as e:
        log.warning("Could not send order email: %s", e)


_current_user_email: Optional[str] = None


def _exec_place_order(store_id: str, items: List[Dict]) -> dict:
    if not _inventory_col().find_one({"Store ID": store_id}):
        return {"error": "Store %s not found" % store_id}

    order_id = "ORD-%s" % uuid.uuid4().hex[:8].upper()
    order_items = []
    total_units = 0

    for item in items:
        pid = item["product_id"]
        qty = item["quantity"]
        product = _inventory_col().find_one({"Store ID": store_id, "Product ID": pid}, {"_id": 0})
        if not product:
            return {"error": "Product %s not found at store %s" % (pid, store_id)}
        vendor = _vendors_col().find_one({"category": product["Category"]}, {"_id": 0})
        unit_price = vendor["price_per_unit"] if vendor else 0
        order_items.append({
            "product_id": pid, "category": product["Category"],
            "quantity": qty, "unit_price": unit_price,
            "vendor_name": vendor["name"] if vendor else "Unknown",
        })
        total_units += qty
        _inventory_col().update_one(
            {"Store ID": store_id, "Product ID": pid},
            {"$inc": {"Current Stock": qty, "Total Units Received": qty}},
        )

    order_doc = {
        "order_id": order_id, "store_id": store_id, "items": order_items,
        "total_units": total_units,
        "total_cost": round(sum(i["quantity"] * i["unit_price"] for i in order_items), 2),
        "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _orders_col().insert_one(order_doc)

    try:
        from routes_inventory import refresh_data
        refresh_data()
    except Exception as e:
        log.warning("Could not refresh dashboard cache: %s", e)

    _send_order_confirmation_email(
        order_id=order_id, store_id=store_id, items=order_items,
        total_units=total_units, total_cost=order_doc["total_cost"],
        user_email=_current_user_email,
    )
    return {
        "success": True, "order_id": order_id, "store_id": store_id,
        "items": order_items, "total_units": total_units, "total_cost": order_doc["total_cost"],
    }


def _exec_get_order_history(store_id: str) -> dict:
    docs = list(_orders_col().find({"store_id": store_id}, {"_id": 0}).sort("created_at", -1).limit(20))
    return {"store": store_id, "orders": docs, "count": len(docs)}


def _exec_list_vendors(category: Optional[str] = None) -> dict:
    query = {"category": category} if category else {}
    docs = list(_vendors_col().find(query, {"_id": 0}))
    return {"vendors": docs, "count": len(docs)}


FUNCTION_MAP = {
    "check_inventory": _exec_check_inventory,
    "get_low_stock_items": _exec_get_low_stock_items,
    "get_reorder_suggestions": _exec_get_reorder_suggestions,
    "place_order": _exec_place_order,
    "get_order_history": _exec_get_order_history,
    "list_vendors": _exec_list_vendors,
}

# ── Conversation history ──────────────────────────────────────────────────────


def _get_active_convo_id(user_id):
    doc = _conversations_col().find_one({"user_id": user_id, "active": True}, {"convo_id": 1})
    return doc["convo_id"] if doc else None


def _load_history(user_id, convo_id=None):
    if not convo_id:
        convo_id = _get_active_convo_id(user_id)
    if not convo_id:
        return []
    doc = _conversations_col().find_one({"user_id": user_id, "convo_id": convo_id})
    return doc.get("messages", []) if doc else []


def _save_history(user_id, messages, convo_id=None):
    if not convo_id:
        convo_id = _get_active_convo_id(user_id)
    if not convo_id:
        convo_id = "c-%s" % uuid.uuid4().hex[:8]
    title = "New Conversation"
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                title = content[:50]
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        title = block["text"][:50]
                        break
            break
    _conversations_col().update_one(
        {"user_id": user_id, "convo_id": convo_id},
        {"$set": {"messages": messages, "title": title, "active": True,
                  "updated_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True,
    )


def _clear_history(user_id):
    _conversations_col().update_many({"user_id": user_id, "active": True}, {"$set": {"active": False}})


def _list_conversations(user_id):
    return list(_conversations_col().find(
        {"user_id": user_id},
        {"_id": 0, "convo_id": 1, "title": 1, "active": 1, "updated_at": 1},
    ).sort("updated_at", -1).limit(20))


# ── Pydantic models ──────────────────────────────────────────────────────────

class GeminiChatRequest(BaseModel):
    message: str


class GeminiChatResponse(BaseModel):
    response: str
    function_calls_made: List[str] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/gemini-chat", response_model=GeminiChatResponse)
async def gemini_chat(request: GeminiChatRequest, user: ClerkUser = Depends(get_current_user)):
    """Send a message to the AI assistant (Claude Haiku)."""
    global _current_user_email
    _current_user_email = user.email

    client = _get_claude()

    # Load conversation history
    history = _load_history(user.user_id)

    # Build messages for Claude (only user/assistant turns)
    messages = []
    for msg in history:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add user message
    messages.append({"role": "user", "content": request.message})

    # Call Claude with tools
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
    except Exception as e:
        log.error("Claude API error: %s", e)
        raise HTTPException(status_code=502, detail="AI service error: %s" % str(e))

    # Handle tool calls in a loop
    function_calls_made = []
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations and response.stop_reason == "tool_use":
        iteration += 1

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Process each tool use block
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn_name = block.name
                fn_args = block.input
                log.info("Claude called tool: %s(%s)", fn_name, fn_args)
                function_calls_made.append(fn_name)

                if fn_name in FUNCTION_MAP:
                    try:
                        result = FUNCTION_MAP[fn_name](**fn_args)
                    except Exception as e:
                        log.error("Function %s error: %s", fn_name, e)
                        result = {"error": str(e)}
                else:
                    result = {"error": "Unknown function: %s" % fn_name}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

        # Send tool results back
        messages.append({"role": "user", "content": tool_results})

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            log.error("Claude error after tool response: %s", e)
            raise HTTPException(status_code=502, detail="AI service error: %s" % str(e))

    # Extract final text
    final_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    if not final_text:
        final_text = "I processed your request but couldn't generate a response. Please try again."

    # Save simplified history (just user/assistant text turns)
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": final_text})
    if len(history) > 100:
        history = history[-100:]
    _save_history(user.user_id, history)

    return GeminiChatResponse(response=final_text, function_calls_made=function_calls_made)


@router.get("/gemini-chat/history")
async def get_history(user: ClerkUser = Depends(get_current_user)):
    history = _load_history(user.user_id)
    return {"messages": history}


@router.delete("/gemini-chat/history")
async def clear_history(user: ClerkUser = Depends(get_current_user)):
    _clear_history(user.user_id)
    return {"success": True}


@router.get("/gemini-chat/conversations")
async def list_convos(user: ClerkUser = Depends(get_current_user)):
    return {"conversations": _list_conversations(user.user_id)}


@router.post("/gemini-chat/conversations/{convo_id}/activate")
async def activate_convo(convo_id: str, user: ClerkUser = Depends(get_current_user)):
    _conversations_col().update_many({"user_id": user.user_id, "active": True}, {"$set": {"active": False}})
    result = _conversations_col().update_one(
        {"user_id": user.user_id, "convo_id": convo_id}, {"$set": {"active": True}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True}


@router.get("/gemini-status")
async def gemini_status():
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    return {"status": "ok" if has_key else "no_api_key", "gemini_enabled": has_key, "model": MODEL}
