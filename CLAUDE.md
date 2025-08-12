# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph learning repository focused on building AI agents and chatbots using LangChain and LangGraph frameworks. The project demonstrates various agent patterns from simple state graphs to complex multi-agent systems.

## Development Setup

**Dependencies Management:**
- Uses `uv` as the package manager (uv.lock present)
- Python 3.12+ required
- Install dependencies: `uv sync` or `pip install -e .`

**Environment:**
- Requires `OPENAI_API_KEY` environment variable
- Uses dotenv for environment configuration

## Code Architecture

**Core Structure:**
- `src/langchain/` - Basic LangChain examples and simple chat implementations
- `src/langgraph/` - LangGraph-specific implementations divided into:
  - `exercise/` - Learning exercises progressing from simple agents to complex loops
  - `agents/` - Production-ready agent implementations (ReAct, RAG, Memory agents)

**Key Patterns:**
- All LangGraph agents use `StateGraph` with typed state classes inheriting from `TypedDict`
- Agents follow the node → edge → conditional edge pattern
- State management flows through graph nodes with immutable state updates
- Exercise progression: simple-agent → multiple-input → operations → edges → conditions → loops

**Agent Types Available:**
- **ReAct Agent** - Reasoning and Acting pattern implementation
- **RAG Agent** - Retrieval Augmented Generation with ChromaDB
- **Memory Agent** - Persistent conversation memory
- **Agent Bot** - Multi-purpose conversational agent
- **Drafter** - Document drafting and editing agent

**External Dependencies:**
- Uses ChromaDB for vector storage (chroma.sqlite3 database file)
- Integrates with OpenAI models via langchain-openai

## Common Commands

**Run Examples:**
- `python src/langgraph/exercise/01_simple-agent.py` - Basic agent
- `python src/langgraph/exercise/06_simple_loop.py` - Complex loop example
- `python src/main.py` - Test environment setup

**Development:**
- No specific test commands found - examine individual files to run agents
- Each agent file is self-contained and executable