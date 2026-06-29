# SETUP.md — Getting Started on Windows

Quick guide to set up the lexical-consensus repository on your machine.

---

## Step 1 — Prerequisites

Install these if you don't have them already:

1. **Python 3.11 or higher** — https://www.python.org/downloads/
   - During install, check "Add Python to PATH"
2. **Git for Windows** — https://git-scm.com/download/win
3. **Visual Studio Code** (recommended editor) — https://code.visualstudio.com/
4. **Claude Code** — https://docs.claude.com/en/docs/agents-and-tools/claude-code/overview

Verify in PowerShell:
```powershell
python --version    # should show 3.11+
git --version       # should show 2.x
```

---

## Step 2 — Create the Repository Structure

Open PowerShell and navigate to where you want the project to live:

```powershell
cd C:\path\to\your\projects      # or wherever you keep your projects
```

Run the setup script (after copying `setup.ps1` from this session):

```powershell
.\setup.ps1
cd lexical-consensus
```

If PowerShell blocks the script, run this first (one-time only):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Step 3 — Copy the Source Files

From this Claude session, copy these files into the corresponding directories:

| File | Destination |
|---|---|
| `README.md` | `./` |
| `PROTOCOL.md` | `./` |
| `CLAUDE.md` | `./` |
| `SETUP.md` | `./` |
| `requirements.txt` | `./` |
| `.gitignore` | `./` |
| `setup.ps1` | `./scripts/` |
| `base_agent.py` | `./src/agents/` |
| `perception.py` | `./src/agents/` |
| `lexicon.py` | `./src/agents/` |
| `learner_agent.py` | `./src/agents/` |
| `ledger.py` | `./src/consensus/` |
| `artificial_vocab.py` | `./src/dataset/` |
| `experiment_summary.md` | `./docs/` |
| `diagnostic.py` | `./experiments/exp_001_baseline/` |
| `README.md` (experiment) | `./experiments/exp_001_baseline/` |

---

## Step 4 — Python Environment

Create a virtual environment so dependencies are isolated:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked, see Step 2 about ExecutionPolicy.

You should see `(.venv)` at the start of your prompt when active.

---

## Step 5 — Initialize Git

```powershell
git init
git add .
git commit -m "Initial structure — three-layer architecture with Carroll vocabulary"
```

---

## Step 6 — Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `lexical-consensus`
3. Description: "Empirical validation of language acquisition as a measure of machine intelligence"
4. Keep it Public if you want it citable, Private if you want time first
5. Do NOT initialize with README, .gitignore, or license (we have these)
6. Click "Create repository"

GitHub will show you the commands to connect:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/lexical-consensus.git
git branch -M main
git push -u origin main
```

---

## Step 7 — Open with Claude Code

From the repository directory:

```powershell
claude
```

Claude Code will read `CLAUDE.md` automatically and have full context of the
project. First conversation can be:

> "Read CLAUDE.md and tell me what the next concrete task is."

It should respond with: implement the diagnostic script in
`experiments/exp_001_baseline/diagnostic.py`.

---

## Step 8 — Verify Everything Works

Quick sanity check:

```powershell
python -c "from src.agents.perception import PerceptionLayer; p = PerceptionLayer(); print('OK')"
```

This will download DINOv2 the first time (about 100MB) and print "OK".
If it works, the architecture is correctly set up.

---

## Troubleshooting

**"python is not recognized"** — Python not in PATH. Reinstall Python and check the "Add to PATH" box.

**"pip install fails on torch"** — On Windows, PyTorch may need a specific install command. Visit https://pytorch.org/get-started/locally/ and copy the command for your system.

**"DINOv2 download fails"** — Check internet connection. If behind a firewall, may need to configure huggingface_hub mirror.

**"Out of memory with DINOv2"** — Use `dinov2-small` (already the default in `perception.py`). Not `large` or `giant`.

---

## What To Do Next

After setup, the first concrete development task is implementing
`diagnostic.py`. Open Claude Code in the project directory and start
the conversation there — it will guide the implementation step by step
based on the project context.
