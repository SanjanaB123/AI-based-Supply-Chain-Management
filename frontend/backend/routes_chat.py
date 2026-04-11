import os
import json
import asyncio
import logging
from typing import TypedDict, Annotated, List

import torch
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException

from clerk_auth import get_current_user
from models import ClerkUser
from setfit import SetFitModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

# Force CPU for SetFit to avoid device mismatch errors on Mac
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Chat"])

# --- Configuration ---
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "config", "inventory_query_classifier")
LABEL_MAPPING_PATH = os.path.join(MODEL_PATH, "label_mapping.json")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
AGENTS_CONFIG_PATH = os.path.join(CONFIG_DIR, "agents.json")

# --- Models ---
class ChatRequest(BaseModel):
    message: str
    username: str

class ChatResponse(BaseModel):
    response: str
    agent: str

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user: str

# Global AI graph — set by init_ai(), used by endpoints
graph = None


async def init_ai():
    """Initialize AI/ML components. Called from app lifespan."""
    global graph

    # 1. Load SetFit classifier
    log.info(f"Loading SetFit model from {MODEL_PATH}...")
    classifier_model = SetFitModel.from_pretrained(MODEL_PATH)
    with open(LABEL_MAPPING_PATH, 'r') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    # 2. Connect to MCP server
    log.info(f"Connecting to MCP server at {MCP_SERVER_URL}...")
    mcp_client = MultiServerMCPClient({
        "inventory": {"url": MCP_SERVER_URL, "transport": "streamable_http"}
    })
    inventory_tools = await mcp_client.get_tools()
    log.info(f"Discovered {len(inventory_tools)} tools from MCP server.")

    # 3. Initialize LLM
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY not found. Chat functionality will be disabled.")
        return

    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=anthropic_api_key,
        temperature=0,
        max_tokens=4096,
    )

    # 4. Load agent configuration
    if not os.path.exists(AGENTS_CONFIG_PATH):
        log.warning(f"Agent config file not found: {AGENTS_CONFIG_PATH}")
        return

    with open(AGENTS_CONFIG_PATH) as f:
        agent_configs = json.load(f)["agents"]

    tool_map = {tool.name: tool for tool in inventory_tools}

    # Resolve prompt_file paths relative to config directory
    for config in agent_configs:
        config["prompt_file"] = os.path.join(CONFIG_DIR, config["prompt_file"])

    # Validate references up front
    for config in agent_configs:
        for tool_name in config["tools"]:
            if tool_name not in tool_map:
                raise ValueError(
                    f"Agent '{config['name']}' references unknown tool '{tool_name}'. "
                    f"Available: {list(tool_map.keys())}"
                )
        if not os.path.exists(config["prompt_file"]):
            raise FileNotFoundError(
                f"Agent '{config['name']}' prompt file not found: '{config['prompt_file']}'"
            )

    # Node factory
    def make_node(a):
        async def node(state: GraphState):
            return await a.ainvoke(state)
        return node

    def irrelevant_node(state: GraphState):
        return {
            "messages": [AIMessage(content="I'm sorry, I'm specialized in inventory management and supply chain forecasting. I can't help with that specific request. Feel free to ask about stock levels or sales summaries!")]
        }

    def router_fn(state: GraphState) -> str:
        query = state["messages"][-1].content
        prediction = classifier_model.predict([query])[0]
        result = label_map[prediction.item()]
        log.info(f"Router classified query as: {result}")
        return result

    # 5. Build graph
    workflow = StateGraph(GraphState)

    for config in agent_configs:
        agent_tools = [tool_map[name] for name in config["tools"]]
        with open(config["prompt_file"]) as f:
            prompt = f.read()
        agent = create_react_agent(llm, tools=agent_tools, prompt=prompt)
        workflow.add_node(config["name"], make_node(agent))
        workflow.add_edge(config["name"], END)
        log.info(f"Registered agent node: '{config['name']}' with tools: {config['tools']}")

    workflow.add_node("irrelevant", irrelevant_node)
    workflow.add_edge("irrelevant", END)

    route_map = {cfg["name"]: cfg["name"] for cfg in agent_configs}
    route_map["irrelevant"] = "irrelevant"
    workflow.add_conditional_edges("__start__", router_fn, route_map)

    graph = workflow.compile(checkpointer=MemorySaver())
    log.info("AI graph compiled successfully.")


# --- Endpoints ---

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _user: ClerkUser = Depends(get_current_user)):
    """AI chat endpoint for inventory management queries."""
    if not graph:
        raise HTTPException(status_code=503, detail="AI chat service not initialized")

    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "user": request.username,
    }
    config = {"configurable": {"thread_id": request.username}}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = await graph.ainvoke(inputs, config)
            break
        except Exception as e:
            if "529" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_secs = 2 ** attempt
                    log.warning(f"Anthropic overloaded (attempt {attempt+1}), retrying in {wait_secs}s...")
                    await asyncio.sleep(wait_secs)
                    continue
            raise

    final_message = result["messages"][-1].content
    return ChatResponse(response=final_message, agent="ai-assistant")


@router.get("/ai-status")
def ai_status(_user: ClerkUser = Depends(get_current_user)):
    """Check if AI chat service is available."""
    ai_ready = graph is not None
    return {
        "status": "online" if ai_ready else "degraded",
        "ai_enabled": ai_ready,
        "model_loaded": ai_ready,
        "mcp_server": MCP_SERVER_URL,
    }
