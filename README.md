# AI Product Research Agent

A production-grade competitive intelligence agent using **LangGraph** (workflow), **LangChain** (LLM abstraction), **Ollama** (local LLM), and **OpenAPI**-spec external tools.

## Architecture

```
research_agent/
├── main.py                    # Entry point — run the agent
├── config/
│   └── settings.py            # All config (model, paths, thresholds)
├── agent/
│   ├── graph.py               # LangGraph state machine definition
│   ├── state.py               # AgentState TypedDict
│   └── nodes.py               # All graph node functions
├── tools/
│   ├── search_tool.py         # Web search (DuckDuckGo + OpenAPI spec)
│   └── scraper_tool.py        # Page content extraction
├── storage/
│   └── cache.py               # JSON-based findings cache
├── report/
│   └── generator.py           # PDF-ready HTML report builder
└── data/
    └── cache/                 # Auto-created — stores findings JSON
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Ollama with llama3
```bash
ollama pull llama3
ollama serve
```

### 3. Run the agent
```bash
python main.py
# Or with a custom query:
python main.py --query "Analyze Salesforce competitors and create a market report"
```

## Stack

| Layer | Library | Role |
|---|---|---|
| LLM abstraction | LangChain | Unified interface to Ollama |
| Workflow | LangGraph | Node graph, state, conditional routing |
| Local LLM | Ollama (llama3) | Inference — no API keys needed |
| Web search | DuckDuckGo Search | External tool via OpenAPI spec |
| Output | Jinja2 + HTML/CSS | PDF-ready market report |

## Output

The agent produces `data/report_<timestamp>.html` — a structured market report with:
- Executive summary
- Competitor profiles (pricing, features, positioning)
- Strengths / weaknesses table
- Strategic recommendations
