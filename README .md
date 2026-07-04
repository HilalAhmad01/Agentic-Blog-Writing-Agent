# Agentic Blog Writing Agent

# This is juts a small project to apply learnings nothing major :)

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/HilalAhmad01/Agentic-Blog-Writing-Agent.git
cd Agentic-Blog-Writing-Agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory and add your API key:

```env
OPENAI_API_KEY=your_openai_api_key
# or
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the agent

```bash
python main.py
```

## Project Structure

```text
Agentic-Blog-Writing-Agent/
├── main.py              # Entry point — runs the graph
├── graph/
│   ├── builder.py       # LangGraph workflow definition
│   ├── state.py         # Shared state schema
│   └── nodes.py         # Individual agent nodes (research, outline, write, review)
├── tools/                # Any tool functions used by agents
├── requirements.txt      # Dependencies
└── .env                  # API keys (not committed)
```

## Future Plans

This is just the starting point. Upcoming additions include:

- **Full Automation** — The agent will automatically pick topics, generate blogs on its own, and auto-post them to platforms (e.g., WordPress, Hashnode, Dev.to, etc.).
- **Scheduling** — Cron-based or event-driven blog generation without manual intervention.
- **Image Generation** — Auto-generate cover images for the blogs.
- **SEO Optimization** — Integrate SEO best practices into the writing agent.
