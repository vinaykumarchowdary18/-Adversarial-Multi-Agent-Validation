# AMAV — Adversarial Multi-Agent Validation

> Your personal AI that cross-validates itself. Four models debate every answer before you see it.

---

## How it works

```
Your question
      │
      ▼
[Tavily]  ←─ Live web evidence fetched first
      │
      ▼
[Proposer — Gemini 2.5 Flash]
  Writes the first structured draft
      │
      ├──────────────────────┐
      ▼                      ▼
[Critic A — Llama 3.3    [Critic B — DeepSeek v3
   via Groq]               via OpenRouter]
 Facts & logic            Completeness & clarity
 (independent)            (independent)
      │                      │
      └──────────┬───────────┘
                 ▼
      [Arbiter — GPT-4o-mini via GitHub]
       Reconciles verdicts, scores consensus
       Decides: finalize | revise | escalate
                 │
        (loops up to MAX_DEBATE_ROUNDS)
                 │
                 ▼
      ✅ Validated answer + confidence score
         + dissenting points logged
```

Each model is a different family (Google / Meta / DeepSeek / OpenAI) running on different infrastructure, ensuring no shared biases.

---

## File layout

```
amav/
├── main.py                      ← Entry point (CLI + interactive REPL)
│
├── core/
│   ├── __init__.py
│   ├── config.py                ← Loads & validates all .env keys
│   ├── models.py                ← Shared Pydantic data models
│   ├── logger.py                ← Rich-powered logging
│   └── orchestrator.py          ← Debate loop engine  ← YOU NEED THIS
│
├── agents/
│   ├── __init__.py
│   ├── _openai_compat.py        ← Shared async HTTP caller (Groq + OpenRouter + GitHub)
│   ├── proposer.py              ← Gemini 2.5 Flash
│   ├── critic_a.py              ← Llama 3.3-70b via Groq
│   ├── critic_b.py              ← DeepSeek v3 via OpenRouter
│   └── arbiter.py               ← GPT-4o-mini via GitHub Models  ← YOU NEED THIS
│
├── tools/
│   ├── __init__.py
│   ├── evidence.py              ← Tavily web search wrapper
│   └── list_github_models.py   ← Utility: discover models your token can access
│
├── outputs/                     ← Auto-created; stores JSON + Markdown reports
│
├── .env                         ← Your keys (never commit this)
├── .env.example                 ← Template
├── requirements.txt
└── README.md
```

---

## Quick setup

### 1. Clone / create folder

```bash
mkdir amav && cd amav
# drop all files into the structure above
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your `.env`

```bash
cp .env.example .env
# open .env and fill in your 5 keys
```

Key → Role mapping:

| .env key | Role | Provider |
|---|---|---|
| `GEMINI_API_KEY` | Proposer | Google AI Studio |
| `GROQ_API_KEY` | Critic A | Groq Cloud |
| `OPENROUTER_API_KEY` | Critic B | OpenRouter |
| `GITHUB_TOKEN` | Arbiter | GitHub Models |
| `TAVILY_API_KEY` | Evidence | Tavily |

> **GitHub token scope:** Your `GITHUB_TOKEN` must have the **Models: read** permission. Go to GitHub → Settings → Developer settings → Fine-grained tokens → create one with that scope.

### 4. Verify GitHub Models access (optional but recommended)

```bash
python tools/list_github_models.py --filter openai
```

This shows every OpenAI model your token can hit. If nothing shows, check the token scope.

### 5. Run

```bash
# Single question
python main.py "What are the most effective study techniques for a Master's-level research methods course?"

# Flag form
python main.py --task "Generate an SOP for applying to a Computer Science MSc at TU Delft"

# Interactive REPL (ask multiple questions in one session)
python main.py --interactive
```

---

## What you'll get

Every answer produces:

1. **Terminal output** — formatted answer + confidence badge
2. `outputs/<timestamp>_<slug>.md` — human-readable report with sources & dissents
3. `outputs/<timestamp>_<slug>.json` — full machine-readable record including all model verdicts

Example confidence levels:

| Consensus score | Meaning |
|---|---|
| ≥ 0.75 | ✅ High confidence — all critics broadly agreed |
| 0.50–0.74 | ⚠️ Medium — some disagreement, answer may need your verification |
| < 0.50 | 🔴 Low — significant unresolved critique; check sources manually |

---

## Use cases (personal AI for Masters + research)

| Task | Example prompt |
|---|---|
| SOP generation | `"Write a Statement of Purpose for a Data Science MSc at University of Edinburgh. Background: BE Computer Science, GPA 8.2, 1 year ML internship"` |
| Research synthesis | `"Summarise the current state of transformer-based protein folding models (2024-2025)"` |
| Literature review | `"What are the key papers on federated learning privacy guarantees from 2022 onwards?"` |
| University comparison | `"Compare MSc AI programmes at ETH Zurich, TU Delft, and KTH for a student interested in NLP"` |
| Daily task | `"Create a weekly study schedule for someone doing 3 courses + thesis proposal in 12 weeks"` |
| Documentation | `"Write a methodology section for a mixed-methods study on student AI tool usage"` |

---

## Tuning debate aggressiveness

In your `.env`:

```env
MAX_DEBATE_ROUNDS=2        # increase to 3 for high-stakes tasks (slower but more rigorous)
MIN_CONSENSUS_SCORE=0.75   # raise to 0.85 for maximum accuracy, lower to 0.65 for speed
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Missing required environment variables` | Check `.env` — all 5 keys must be set |
| `HTTP 401 from github` | Token missing **Models: read** scope |
| `HTTP 429` (rate limit) | Free OpenRouter tier: 50 req/day; add $5 credit to raise limit |
| `Unexpected Gemini response shape` | Model name wrong — run `gemini-2.5-flash` or check Google AI Studio |
| Tavily returns no results | Network issue or key expired; pipeline continues without evidence |

---

## Architecture notes

- **No shared state between critics** — Critic B explicitly has not seen Critic A's output (enforced by independent parallel async calls).
- **Model diversity by design** — Gemini (Google), Llama (Meta), DeepSeek (Chinese open-weight), GPT (OpenAI) — four different training paradigms, four different failure modes.
- **Transparency** — dissenting minority critique points are always logged even when overruled, so you can see what the system disagreed on.
- **Evidence-first** — Tavily fetches live web context before any model sees the task, grounding all 4 agents in current facts rather than training data alone.
