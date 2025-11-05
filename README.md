# Building Agentic AI Systems: A Complete Developer Guide

**Learn how to build intelligent, tool-using AI agents with LangChain and LangGraph**

---

## 📚 Table of Contents

1. [Introduction to Agentic AI](#introduction)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture)
4. [Step-by-Step Development](#development)
5. [Advanced Patterns](#advanced-patterns)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Real-World Example: Document Explorer Agent](#example)

---

## 🎯 Introduction to Agentic AI {#introduction}

### What is Agentic AI?

**Agentic AI** refers to AI systems that can:
- 🤔 **Reason** about problems and plan solutions
- 🔧 **Use tools** to gather information or perform actions
- 💭 **Remember** context across conversations
- 🎯 **Make decisions** about which actions to take
- 🔄 **Execute multi-step** workflows autonomously

### Traditional RAG vs Agentic AI

| Feature | Traditional RAG | Agentic AI |
|---------|----------------|------------|
| **Query Flow** | Fixed: Retrieve → Generate | Dynamic: Reason → Plan → Execute → Generate |
| **Tool Usage** | None | Multiple tools available |
| **Decision Making** | Predetermined logic | AI-driven decisions |
| **Multi-step** | Single pass | Can chain multiple actions |
| **Context** | Stateless | Stateful with memory |
| **Flexibility** | Limited | Highly adaptable |

### When to Use Agentic AI?

✅ **Use Agentic AI when you need:**
- Complex query understanding and reformulation
- Multi-step reasoning (e.g., "Compare X across all Y")
- Dynamic tool selection based on context
- Conversation memory and follow-up questions
- Adaptive behavior based on available data

❌ **Don't use Agentic AI when:**
- Simple, predictable workflows suffice
- Latency is critical (agents add overhead)
- Cost is a major concern (more LLM calls)
- You need 100% deterministic behavior

---

## 🧠 Core Concepts {#core-concepts}

### 1. State Management

**State** is the central data structure that flows through your agent.

```python
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    """Define what information flows through the agent"""
    messages: Annotated[List[BaseMessage], operator.add]  # Conversation history
    query: str                                            # Current query
    search_results: List[Dict]                           # Retrieved data
    answer: str                                          # Final answer
    tool_calls: List[str]                               # Tools used
    reasoning: str                                       # Agent's thought process
```

**Key Principles:**
- Use `TypedDict` for type safety
- `Annotated` with `operator.add` to accumulate values (e.g., messages)
- Keep state minimal but sufficient
- Include metadata for debugging (tool_calls, reasoning)

### 2. Tools

**Tools** are functions the agent can call to interact with external systems.

```python
from langchain_core.tools import tool

@tool
def search_documents(query: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Search for documents. The docstring is CRITICAL - it tells 
    the agent when and how to use this tool.
    
    Args:
        query: What to search for
        top_k: Number of results
        
    Returns:
        Search results with metadata
    """
    # Your implementation
    results = vector_database.search(query, limit=top_k)
    return {"results": results, "count": len(results)}
```

**Tool Design Best Practices:**
1. **Clear Docstrings**: The LLM reads these to decide when to use the tool
2. **Type Hints**: Help with validation and agent understanding
3. **Return Structured Data**: Always return Dict or Pydantic models
4. **Error Handling**: Return errors as data, don't raise exceptions
5. **Logging**: Log tool invocations for debugging

### 3. Nodes

**Nodes** are the processing steps in your agent workflow.

```python
def _agent_node(self, state: AgentState) -> AgentState:
    """
    The reasoning node - decides what to do next.
    Returns updated state, often with tool calls.
    """
    messages = state["messages"]
    
    # Add system prompt
    system_message = SystemMessage(content="You are a helpful assistant...")
    messages_with_system = [system_message] + messages
    
    # Invoke LLM with tools bound
    response = self.llm_with_tools.invoke(messages_with_system)
    
    # Update state
    return {**state, "messages": [response]}
```

**Common Node Types:**
1. **Agent Node**: LLM reasoning and tool selection
2. **Tool Node**: Execute selected tools
3. **Generator Node**: Create final output
4. **Router Node**: Conditional logic for flow control

### 4. Edges

**Edges** connect nodes and define workflow.

```python
# Unconditional edge: Always go from A to B
workflow.add_edge("tool_execution", "agent")

# Conditional edge: Choose next node based on logic
workflow.add_conditional_edges(
    "agent",
    should_continue,  # Function that returns next node name
    {
        "continue": "tools",
        "end": "generate_answer"
    }
)
```

### 5. Memory

**Memory** allows agents to remember conversations.

```python
from langgraph.checkpoint.memory import MemorySaver

# Initialize memory
memory = MemorySaver()

# Compile graph with memory
graph = workflow.compile(checkpointer=memory)

# Use with session ID
config = {"configurable": {"thread_id": "user123"}}
result = graph.invoke(initial_state, config)
```

---

## 🏗️ Architecture Overview {#architecture}

### The Agent Loop

```
┌─────────────────────────────────────────┐
│         User Query                      │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Agent Node (Reasoning)                  │
│  - Understand query                      │
│  - Decide on strategy                    │
│  - Select tools to use                   │
└──────────────┬───────────────────────────┘
               ↓
         ┌─────┴─────┐
         │ Condition │
         └─────┬─────┘
               ↓
    ┌──────────┴──────────┐
    ↓                     ↓
┌───────────┐      ┌─────────────┐
│Tool Node  │      │Generate     │
│Execute    │      │Answer Node  │
│Selected   │      │Create final │
│Tools      │      │response     │
└─────┬─────┘      └──────┬──────┘
      ↓                   ↓
      └───────→ Agent ────┘
              (loop back)
```

### Key Components

1. **StateGraph**: The workflow container
2. **Nodes**: Processing units (functions)
3. **Edges**: Connections between nodes
4. **Checkpointer**: Memory for persistence
5. **Tools**: External functions agent can call

---

## 🛠️ Step-by-Step Development {#development}

### Step 1: Setup Dependencies

```bash
pip install langchain langchain-openai langchain-core langgraph
```

### Step 2: Define Your State

```python
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator

class MyAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    results: List[Dict[str, Any]]
    answer: str
```

### Step 3: Create Tools

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> Dict[str, Any]:
    """
    Calculate mathematical expressions.
    Use this when user asks for calculations.
    
    Args:
        expression: Math expression like "2 + 2"
    """
    try:
        result = eval(expression)  # Use safe eval in production!
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool  
def search_web(query: str) -> Dict[str, Any]:
    """
    Search the web for information.
    Use when you need external knowledge.
    
    Args:
        query: Search query
    """
    # Your search implementation
    results = web_search_api.search(query)
    return {"success": True, "results": results}
```

### Step 4: Initialize LLM with Tools

```python
from langchain_openai import AzureChatOpenAI

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint="your-endpoint",
    api_key="your-key",
    deployment_name="gpt-4o-mini",
    temperature=0.2
)

# Create tool list
tools = [calculator, search_web]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)
```

### Step 5: Create Agent Node

```python
from langchain_core.messages import SystemMessage, HumanMessage

def agent_node(state: MyAgentState) -> MyAgentState:
    """Agent reasoning - decides what tools to use"""
    
    messages = state["messages"]
    
    # System prompt guides the agent
    system_prompt = SystemMessage(content="""
    You are a helpful assistant with access to tools.
    
    Available tools:
    - calculator: For math problems
    - search_web: For looking up information
    
    Think step-by-step:
    1. Understand what the user needs
    2. Decide which tools to use
    3. Call tools with appropriate parameters
    4. After getting results, provide a clear answer
    
    Be concise and accurate.
    """)
    
    # Add system message to conversation
    messages_with_system = [system_prompt] + messages
    
    # LLM decides whether to use tools
    response = llm_with_tools.invoke(messages_with_system)
    
    return {**state, "messages": [response]}
```

### Step 6: Create Tool Execution Node

```python
from langgraph.prebuilt import ToolNode

# ToolNode automatically executes tools from agent's decisions
tool_node = ToolNode(tools)

def execute_tools(state: MyAgentState) -> MyAgentState:
    """Execute tools selected by agent"""
    
    # ToolNode handles tool execution
    result = tool_node.invoke(state)
    
    # Extract and store results
    messages = result.get("messages", [])
    
    # Update state with tool outputs
    return {
        **state,
        "messages": messages
    }
```

### Step 7: Create Answer Generator Node

```python
from langchain_core.messages import AIMessage

def generate_answer(state: MyAgentState) -> MyAgentState:
    """Generate final answer based on tool results"""
    
    messages = state["messages"]
    query = state["query"]
    
    # Create final answer prompt
    answer_prompt = f"""
    Based on the tool results above, provide a final answer to:
    
    Question: {query}
    
    Your answer:
    """
    
    # Generate answer
    final_response = llm.invoke(messages + [HumanMessage(content=answer_prompt)])
    
    return {
        **state,
        "answer": final_response.content,
        "messages": [AIMessage(content=final_response.content)]
    }
```

### Step 8: Define Conditional Logic

```python
def should_continue(state: MyAgentState) -> str:
    """
    Decide whether to continue using tools or generate final answer.
    
    Returns:
        "continue" - Execute more tools
        "end" - Generate final answer
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If LLM made tool calls, continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    
    # Otherwise, generate answer
    return "end"
```

### Step 9: Build the Graph

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Create graph
workflow = StateGraph(MyAgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", execute_tools)
workflow.add_node("generate_answer", generate_answer)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "generate_answer"
    }
)

# Tools loop back to agent
workflow.add_edge("tools", "agent")

# Answer goes to END
workflow.add_edge("generate_answer", END)

# Compile with memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
```

### Step 10: Use the Agent

```python
def query_agent(question: str, session_id: str = "default"):
    """Query the agent"""
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "query": question,
        "results": [],
        "answer": ""
    }
    
    # Config with session for memory
    config = {"configurable": {"thread_id": session_id}}
    
    # Run agent
    result = graph.invoke(initial_state, config)
    
    return result["answer"]

# Test it!
answer = query_agent("What is 25 * 4 + 10?", session_id="user123")
print(answer)  # "The answer is 110"

# Follow-up (uses memory)
answer = query_agent("Now divide that by 2", session_id="user123")
print(answer)  # "The answer is 55"
```

---

## 🚀 Advanced Patterns {#advanced-patterns}

### Pattern 1: Dynamic Tool Selection

```python
def agent_node_with_context(state: MyAgentState) -> MyAgentState:
    """Agent that considers context for tool selection"""
    
    messages = state["messages"]
    available_data = state.get("available_data", {})
    
    # Customize system prompt based on context
    if available_data.get("has_database"):
        tools_desc = "You have access to: calculator, search_web, query_database"
    else:
        tools_desc = "You have access to: calculator, search_web"
    
    system_prompt = SystemMessage(content=f"""
    You are a helpful assistant.
    
    {tools_desc}
    
    Use tools strategically based on the query.
    """)
    
    messages_with_system = [system_prompt] + messages
    response = llm_with_tools.invoke(messages_with_system)
    
    return {**state, "messages": [response]}
```

### Pattern 2: Enforcing Constraints

```python
def agent_node_with_constraints(state: MyAgentState) -> MyAgentState:
    """Agent with hard constraints on behavior"""
    
    messages = state["messages"]
    document_set = state.get("document_set_name")
    
    system_content = "You are a helpful assistant."
    
    # Add constraint if document set specified
    if document_set:
        system_content += f"""
        
        **CRITICAL CONSTRAINT**: 
        You MUST ONLY search in the '{document_set}' document set.
        When calling search_documents, ALWAYS use document_set_name='{document_set}'.
        Do NOT search in any other document sets.
        """
    
    system_message = SystemMessage(content=system_content)
    messages_with_system = [system_message] + messages
    
    response = llm_with_tools.invoke(messages_with_system)
    return {**state, "messages": [response]}
```

### Pattern 3: Multi-Agent Collaboration

```python
class SpecialistAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    task: str
    specialist_results: Dict[str, Any]

def coordinator_node(state: SpecialistAgentState) -> SpecialistAgentState:
    """Coordinator decides which specialist to use"""
    
    task = state["task"]
    
    # Analyze task
    analysis = llm.invoke([HumanMessage(content=f"""
    Analyze this task and decide which specialist to use:
    Task: {task}
    
    Specialists:
    - data_analyst: For data analysis and statistics
    - researcher: For research and information gathering
    - writer: For content creation
    
    Which specialist should handle this? Respond with just the name.
    """)])
    
    specialist = analysis.content.strip().lower()
    
    return {
        **state,
        "specialist_selected": specialist
    }

def route_to_specialist(state: SpecialistAgentState) -> str:
    """Route to appropriate specialist"""
    return state.get("specialist_selected", "researcher")
```

### Pattern 4: Error Recovery

```python
def tool_node_with_retry(state: MyAgentState) -> MyAgentState:
    """Tool execution with automatic retry"""
    
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls if hasattr(last_message, "tool_calls") else []
    
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Try up to 3 times
        for attempt in range(3):
            try:
                result = execute_tool(tool_name, tool_args)
                results.append(result)
                break
            except Exception as e:
                if attempt == 2:  # Last attempt
                    results.append({
                        "error": f"Failed after 3 attempts: {str(e)}",
                        "tool": tool_name
                    })
                else:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
    
    return {
        **state,
        "tool_results": results
    }
```

### Pattern 5: Streaming Responses

```python
async def query_agent_streaming(question: str, session_id: str = "default"):
    """Query agent with streaming output"""
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "query": question,
        "answer": ""
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    # Stream events from the graph
    async for event in graph.astream_events(initial_state, config, version="v1"):
        kind = event["event"]
        
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content
        
        elif kind == "on_tool_start":
            tool_name = event["name"]
            yield f"\n[Using tool: {tool_name}]\n"
        
        elif kind == "on_tool_end":
            yield "\n[Tool execution complete]\n"
```

---

## 💡 Best Practices {#best-practices}

### 1. Tool Design

✅ **DO:**
- Write detailed docstrings (LLM reads them)
- Return structured data (Dict, Pydantic models)
- Include error information in return value
- Log all tool invocations
- Keep tools focused (single responsibility)

❌ **DON'T:**
- Raise exceptions from tools
- Return raw strings without structure
- Create tools with too many parameters (>5)
- Put business logic in tool functions
- Forget type hints

### 2. System Prompts

✅ **DO:**
- Be explicit about tool usage
- Provide examples of good behavior
- Set clear constraints
- Explain the agent's role
- Include formatting instructions

❌ **DON'T:**
- Make prompts too long (>1000 tokens)
- Use vague language
- Contradict yourself
- Forget to update prompts when adding tools
- Assume the LLM knows your system

### 3. State Management

✅ **DO:**
- Keep state minimal
- Use TypedDict for type safety
- Document each state field
- Initialize all fields
- Use Annotated for accumulation

❌ **DON'T:**
- Store large objects in state
- Use mutable defaults
- Mix concerns in state
- Forget to update state in nodes
- Pass sensitive data through state logs

### 4. Error Handling

```python
def safe_tool_execution(state: AgentState) -> AgentState:
    """Example of proper error handling"""
    
    try:
        # Try to execute tools
        result = tool_node.invoke(state)
        
        # Validate result
        if not result:
            raise ValueError("Empty result from tools")
        
        return {**state, **result}
    
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        
        # Return error as data, don't crash
        error_message = ToolMessage(
            content=f"Error: {str(e)}",
            tool_call_id="error"
        )
        
        return {
            **state,
            "messages": [error_message],
            "error": str(e)
        }
```

### 5. Testing

```python
import pytest
from unittest.mock import Mock, patch

def test_agent_node():
    """Test agent reasoning"""
    
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = "I need to use the calculator"
    mock_response.tool_calls = [
        {"name": "calculator", "args": {"expression": "2+2"}}
    ]
    
    # Mock LLM
    with patch.object(llm_with_tools, 'invoke', return_value=mock_response):
        state = {
            "messages": [HumanMessage(content="What is 2+2?")],
            "query": "What is 2+2?"
        }
        
        result = agent_node(state)
        
        assert len(result["messages"]) > 0
        assert result["messages"][0].tool_calls[0]["name"] == "calculator"

def test_tool_execution():
    """Test tool works correctly"""
    
    result = calculator("2 + 2")
    
    assert result["success"] == True
    assert result["result"] == 4
```

### 6. Monitoring and Logging

```python
import logging
from datetime import datetime

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monitored_agent_node(state: AgentState) -> AgentState:
    """Agent node with detailed monitoring"""
    
    start_time = datetime.now()
    query = state.get("query", "")
    
    logger.info(f"Agent processing query: {query[:100]}")
    
    try:
        # Process
        result = agent_node(state)
        
        # Log tool calls
        last_message = result["messages"][-1]
        if hasattr(last_message, "tool_calls"):
            tools_used = [tc["name"] for tc in last_message.tool_calls]
            logger.info(f"Agent selected tools: {tools_used}")
        
        # Log timing
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Agent processing took {duration:.2f}s")
        
        return result
    
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise
```

---

## 🔧 Troubleshooting {#troubleshooting}

### Problem 1: Agent Not Using Tools

**Symptoms:**
- Agent gives answers without calling tools
- Tools are ignored even when needed

**Solutions:**

1. **Check Tool Docstrings:**
```python
# ❌ BAD - No guidance
@tool
def search(q: str):
    return search_api(q)

# ✅ GOOD - Clear guidance
@tool
def search(query: str) -> Dict[str, Any]:
    """
    Search the database for information.
    
    Use this tool when:
    - User asks about specific data
    - You need to look up facts
    - Query requires database information
    
    Args:
        query: The search query string
    
    Returns:
        Search results with relevance scores
    """
    return search_api(query)
```

2. **Strengthen System Prompt:**
```python
system_prompt = """
You MUST use tools to answer questions. 
Do NOT make up information.

For ANY question about data:
1. FIRST call the search tool
2. THEN provide an answer based on results

Example:
User: "What is product X's price?"
Your response: [Call search tool with "product X price"]
"""
```

3. **Verify Tool Binding:**
```python
# Make sure tools are bound
print(llm_with_tools.tools)  # Should show your tools

# Test tool invocation
response = llm_with_tools.invoke([
    HumanMessage(content="Search for 'test'")
])
print(response.tool_calls)  # Should show tool calls
```

### Problem 2: Infinite Loops

**Symptoms:**
- Agent never reaches END
- Same tools called repeatedly
- Timeout errors

**Solutions:**

1. **Add Loop Counter:**
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    loop_count: int  # Add this

def should_continue(state: AgentState) -> str:
    loop_count = state.get("loop_count", 0)
    
    # Force stop after 5 iterations
    if loop_count >= 5:
        logger.warning("Max loops reached, forcing end")
        return "end"
    
    # Normal logic
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

def agent_node(state: AgentState) -> AgentState:
    # Increment counter
    loop_count = state.get("loop_count", 0) + 1
    
    # ... rest of logic ...
    
    return {
        **state,
        "loop_count": loop_count,
        "messages": [response]
    }
```

2. **Clear Termination Instructions:**
```python
system_prompt = """
After using tools and getting results, you MUST:
1. Stop using tools
2. Provide a final answer

Do NOT call the same tool twice unless you get an error.
"""
```

### Problem 3: Memory Not Working

**Symptoms:**
- Agent doesn't remember previous messages
- Context lost between queries

**Solutions:**

1. **Verify Session ID:**
```python
# ❌ BAD - Different session each time
result1 = graph.invoke(state, {"configurable": {"thread_id": "random1"}})
result2 = graph.invoke(state, {"configurable": {"thread_id": "random2"}})

# ✅ GOOD - Same session
session_id = "user123"
config = {"configurable": {"thread_id": session_id}}
result1 = graph.invoke(state1, config)
result2 = graph.invoke(state2, config)  # Remembers result1
```

2. **Check Memory Initialization:**
```python
from langgraph.checkpoint.memory import MemorySaver

# Create memory BEFORE compiling
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Not:
# graph = workflow.compile(checkpointer=MemorySaver())  # Creates new each time
```

3. **Inspect State:**
```python
# Check what's in memory
config = {"configurable": {"thread_id": "user123"}}
snapshot = graph.get_state(config)
print(snapshot.values.get("messages", []))
```

### Problem 4: Slow Performance

**Symptoms:**
- Long response times
- High API costs
- Timeouts

**Solutions:**

1. **Optimize Tool Calls:**
```python
# ❌ BAD - Retrieves too much
@tool
def get_all_data():
    """Get all database records"""
    return db.query("SELECT * FROM huge_table")

# ✅ GOOD - Targeted retrieval
@tool
def search_data(query: str, limit: int = 10):
    """Search specific records with limit"""
    return db.query(
        "SELECT * FROM table WHERE content LIKE ? LIMIT ?",
        query, limit
    )
```

2. **Reduce LLM Calls:**
```python
# Use smaller, faster models for tool selection
fast_llm = AzureChatOpenAI(deployment_name="gpt-4o-mini")
strong_llm = AzureChatOpenAI(deployment_name="gpt-4o")

def agent_node(state):
    # Use fast model for tool selection
    response = fast_llm_with_tools.invoke(messages)
    return {**state, "messages": [response]}

def generate_answer(state):
    # Use strong model for final answer
    response = strong_llm.invoke(messages)
    return {**state, "answer": response.content}
```

3. **Parallel Tool Execution:**
```python
import asyncio

async def parallel_tool_execution(tool_calls):
    """Execute multiple tools in parallel"""
    
    tasks = [
        execute_tool_async(tc["name"], tc["args"])
        for tc in tool_calls
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

---

## 📖 Real-World Example: Document Explorer Agent {#example}

Let's walk through the Document Explorer Agent implementation step-by-step.

### Architecture

```
User Query
    ↓
[Agent Node] - Reads query + document_set constraint
    ↓
Decision: Use tools?
    ↓                    ↓
[Tool Node]          [Generate Answer]
 ├─ search_documents      ↓
 ├─ get_document_sets    Create response
 └─ analyze_query        with citations
    ↓                         ↓
[Agent Node] ←──────────── END
 (loops back)
```

### Step 1: Define State

```python
class AgentState(TypedDict):
    """All data that flows through the agent"""
    messages: Annotated[List[BaseMessage], operator.add]
    document_set_name: Optional[str]  # Constraint from user
    query: str                        # Original question
    search_results: List[Dict]        # Retrieved documents
    answer: str                       # Final answer
    source_files: List[Dict]          # Citation metadata
    tool_calls: List[str]             # Tools used (for debugging)
    reasoning: str                    # Agent thought process
```

**Why this structure?**
- `messages`: LangChain requires this for conversation
- `document_set_name`: Constraint to filter search
- `search_results`: Temporarily store retrieved docs
- `source_files`: Track which files to cite
- `tool_calls`: Debugging and transparency

### Step 2: Create Specialized Tools

```python
@tool
def search_documents(
    query: str, 
    document_set_name: Optional[str] = None, 
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Search for relevant documents using vector similarity search.
    
    Use this tool when the user asks about information in documents.
    
    Args:
        query: What to search for
        document_set_name: Specific collection to search (IMPORTANT: use if specified)
        top_k: Number of results (3-10 for specific, 20-50 for broad)
    
    Returns:
        Search results with file names, content, and scores
    """
    try:
        # Ensure database is ready
        vector_service.ensure_index_exists()
        
        # Execute vector search
        results = vector_service.execute_similarity_query(
            query_text=query,
            document_set_name=document_set_name,
            top_k=top_k
        )
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        # Return error as data
        return {
            "success": False,
            "error": str(e),
            "results": []
        }
```

**Key Points:**
- Detailed docstring tells agent when to use it
- Parameters clearly documented
- Error handling returns data (doesn't raise)
- Logs execution for debugging

### Step 3: Agent Node with Constraints

```python
def _agent_node(self, state: AgentState) -> AgentState:
    """Agent decides what tools to use"""
    
    messages = state["messages"]
    document_set_name = state.get("document_set_name")
    
    # Base system prompt
    system_content = """You are an intelligent document assistant.

Your tools:
1. search_documents - Find relevant documents
2. get_document_sets - List available collections
3. analyze_query_intent - Understand complex queries

Your process:
1. Understand the query
2. Use tools to gather information
3. Stop tool use and provide a comprehensive answer

Guidelines:
- Use search_documents with appropriate top_k (3-10 for specific, 20-50 for broad)
- Always cite sources with file names and page numbers
- Be thorough but concise"""
    
    # ADD CONSTRAINT if document set specified
    if document_set_name:
        system_content += f"""

**IMPORTANT CONSTRAINT**: 
The user specified to search ONLY in '{document_set_name}'.
You MUST use document_set_name='{document_set_name}' when calling search_documents.
Do NOT search other document sets.
"""
    
    system_message = SystemMessage(content=system_content)
    messages_with_system = [system_message] + messages
    
    # LLM with tools bound decides next action
    response = self.llm_with_tools.invoke(messages_with_system)
    
    return {**state, "messages": [response]}
```

**Why this works:**
- Dynamic system prompt based on state
- Explicit constraint enforcement
- Clear instructions for tool usage
- Bold **IMPORTANT** and strong language (MUST, Do NOT)

### Step 4: Tool Execution Node

```python
def _tool_node(self, state: AgentState) -> AgentState:
    """Execute tools selected by agent"""
    
    # ToolNode automatically handles execution
    result = self.tool_node.invoke(state)
    
    # Extract tool calls for logging
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls if hasattr(last_message, "tool_calls") else []
    tool_call_names = [tc["name"] for tc in tool_calls]
    
    # Check for search results in tool outputs
    search_results = state.get("search_results", [])
    for msg in result.get("messages", []):
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            try:
                import json
                content_dict = json.loads(msg.content)
                if "results" in content_dict:
                    search_results = content_dict["results"]
            except:
                pass
    
    # Update tool calls tracking
    existing_calls = state.get("tool_calls", [])
    updated_calls = existing_calls + tool_call_names
    
    return {
        **state,
        "messages": result.get("messages", []),
        "search_results": search_results,
        "tool_calls": updated_calls
    }
```

**Key Points:**
- Uses LangGraph's ToolNode (handles complexity)
- Extracts results from tool outputs
- Tracks tool calls for transparency
- Preserves all state

### Step 5: Answer Generation with RAG

```python
def _generate_answer_node(self, state: AgentState) -> AgentState:
    """Create final answer with citations"""
    
    search_results = state.get("search_results", [])
    query = state.get("query", "")
    
    if search_results:
        # Prepare context from top 3 results
        context_parts = []
        source_files = []
        
        for idx, doc in enumerate(search_results[:3], 1):
            content = doc.get('content', '')[:8000]  # Limit size
            file_name = doc.get('file_name', 'Unknown')
            doc_set = doc.get('document_set_name', '')
            
            context_parts.append(
                f"[Source {idx}: {file_name}]\n{content}\n"
            )
            
            # Track unique sources
            if file_name not in [f['name'] for f in source_files]:
                source_files.append({
                    'name': file_name,
                    'document_set': doc_set
                })
        
        combined_context = "\n".join(context_parts)
        
        # RAG prompt
        answer_prompt = f"""Based on the following documents, answer the question.

Documents:
{combined_context}

Question: {query}

Instructions:
- Answer ONLY based on provided documents
- Cite file names and page numbers [PAGE X]
- Be detailed but concise
- If information is missing, state that clearly

Your Answer:"""
        
        # Generate answer
        answer_response = self.llm.invoke([
            HumanMessage(content=answer_prompt)
        ])
        answer = answer_response.content
        
        # Append source references
        if source_files:
            answer += "\n\n---\n\n📄 **Source Documents:**\n\n"
            for source in source_files:
                answer += f"- {source['name']}"
                if source['document_set']:
                    answer += f" (from {source['document_set']})"
                answer += "\n"
        
        state["answer"] = answer
        state["source_files"] = source_files
    else:
        # No results found
        answer = "I couldn't find relevant documents to answer your question."
        state["answer"] = answer
        state["source_files"] = []
    
    return {**state, "messages": [AIMessage(content=state["answer"])]}
```

**RAG Best Practices:**
- Limit context size (8000 chars per doc)
- Use top 3 results (balance between context and quality)
- Clear source attribution
- Explicit instructions to LLM
- Handle no-results case gracefully

### Step 6: Build the Graph

```python
def _create_graph(self) -> StateGraph:
    """Assemble the agent workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", self._agent_node)
    workflow.add_node("tools", self._tool_node)
    workflow.add_node("generate_answer", self._generate_answer_node)
    
    # Entry point
    workflow.set_entry_point("agent")
    
    # Conditional routing from agent
    workflow.add_conditional_edges(
        "agent",
        self._should_continue,
        {
            "continue": "tools",      # If tool calls exist
            "end": "generate_answer"  # If no tool calls
        }
    )
    
    # Tools loop back to agent
    workflow.add_edge("tools", "agent")
    
    # Answer goes to END
    workflow.add_edge("generate_answer", END)
    
    # Compile with memory
    return workflow.compile(checkpointer=self.memory)
```

### Step 7: Main Query Interface

```python
def query(
    self,
    query_text: str,
    document_set_name: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Main entry point"""
    
    try:
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query_text)],
            "document_set_name": document_set_name,  # Constraint
            "query": query_text,
            "search_results": [],
            "answer": "",
            "source_files": [],
            "tool_calls": [],
            "reasoning": ""
        }
        
        # Config for memory
        config = {"configurable": {"thread_id": session_id or "default"}}
        
        # Run agent
        final_state = self.graph.invoke(initial_state, config)
        
        # Return structured result
        return {
            "success": True,
            "Answer": final_state.get("answer", ""),
            "results": final_state.get("search_results", []),
            "tool_calls": final_state.get("tool_calls", []),
            "source_files": final_state.get("source_files", [])
        }
    
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "Answer": f"Error: {str(e)}"
        }
```

### What Makes This Agent Effective?

1. **Constrained Tool Use**: document_set_name enforcement
2. **Multi-Step Reasoning**: Can call multiple tools
3. **Memory**: Conversation context across queries
4. **Transparency**: Returns tool_calls and reasoning
5. **Error Handling**: Graceful degradation
6. **Citations**: RAG with source tracking
7. **Flexibility**: Adapts strategy based on query

---

## 🎓 Learning Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

### Tutorials
- [LangChain Quickstart](https://python.langchain.com/docs/get_started/quickstart)
- [Building Agents with LangGraph](https://langchain-ai.github.io/langgraph/tutorials/)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)

---

## 📝 Summary

### Key Takeaways

1. **Agentic AI = Reasoning + Tools + Memory**
   - Agents decide what tools to use dynamically
   - State flows through graph nodes
   - Memory enables multi-turn conversations

2. **Design Principles**
   - Clear tool docstrings (LLM reads them!)
   - Explicit system prompts with constraints
   - Structured state management
   - Error handling as data, not exceptions

3. **Common Patterns**
   - Agent → Tools → Agent loop
   - Conditional routing based on tool calls
   - RAG for answer generation
   - Session-based memory

4. **Optimization**
   - Limit context size
   - Use faster models for tool selection
   - Parallel tool execution
   - Add loop counters to prevent infinite loops

5. **Testing & Monitoring**
   - Unit test individual nodes
   - Mock LLM responses
   - Log tool invocations
   - Track performance metrics

### Next Steps

1. Start with a simple agent (1-2 tools)
2. Test tool docstrings thoroughly
3. Add constraints incrementally
4. Implement memory for multi-turn
5. Monitor and optimize performance
6. Expand to multi-agent systems

---

**Happy Building! 🚀**

*This guide covered the complete journey from basic concepts to production-ready agentic AI systems. Use it as a reference as you build your own intelligent agents.*

---

*Last Updated: November 5, 2025*
*Version: 1.0.0*
*Author: AI Infused Data Engineering Team*
