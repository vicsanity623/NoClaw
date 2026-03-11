<p align="center">
  <img src="pyob.png" alt="PyOB" width="512" />
</p>
<div align="center">

# ∞ PyOuroBorus

### The Self-Healing, Symbolic Autonomous Code Architect

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-8E75B2?style=for-the-badge&logo=google-gemini&logoColor=white)](https://ai.google.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()
![Verify Codebase](https://github.com/vicsanity623/PyOB/actions/workflows/verify.yml/badge.svg)
![Marketplace](https://img.shields.io/badge/Marketplace-PyOB-green?logo=github)

**PYOB is a high-fidelity autonomous agent that performs surgical code modifications, cross-file dependency tracking, and self-healing verification — all without destroying your codebase.**

[Getting Started](#-getting-started) · [How It Works](#phase-1-initial-assessment--codebase-scan) · [Architecture](#system-overview) · [Documentation](#pyob--complete-technical-documentation)

---

</div>

---

This is the result of ongoing, **real-time self-evolution** autonomous loop running on GitHub Actions. As a sole developer, I have built the core engine, verification pipeline, and cloud deployment service (`pyob-bot`). 

Because the development process is so intertwined with automation, things sometimes move fast, and the README and DOCUMENTATION can lag behind the latest patch / cloud-verified commits (as seen in the most recent PRs).

**We welcome contributions!** If you have ideas on improving the LLM prompts (the core intelligence), streamlining the 5-layer verification, or fixing any unexpected edge cases, please:
1.  Fork this repository.
2.  Make your changes.
3.  Submit a **Pull Request**.

---
---

## 🧠 What is PyOB?

PyOB is NOT a personal AI CHAT assistant. PyOB is an **autonomous code review and feature engineering system** that continuously analyzes, patches, and evolves your codebase through a multi-stage verification pipeline. Unlike "black-box" coding assistants that rewrite entire files, PyOB operates with **surgical XML-based edits**, a **persistent symbolic dependency ledger**, and **automated GitHub integration** — ensuring your project is never left in a broken state. It **already knows its purpose and goals**.

### Key Differentiators

| Feature | Traditional AI Assistants | PYOB |
|---|---|---|
| **Edit Strategy** | Full file rewrites | Surgical `<SEARCH>/<REPLACE>` XML blocks |
| **Dependency Awareness** | None | Symbolic ledger (`SYMBOLS.json`) with ripple detection |
| **Error Recovery** | Manual | Context-aware self-healing with auto-rollback |
| **Verification** | None | 5-layer pipeline: XML matching → linting → Mypy → PIR → runtime test |
| **State Persistence** | Stateless | Hidden `.pyob/` vault: `MEMORY.md`, `ANALYSIS.md`, `HISTORY.md` |
| **API Resilience** | Single key, fails on rate limit | 10-key rotation with **Smart Sleep Backoff** and local fallback |

---

## ✨ Core Features

### 🔬 Surgical XML-Based Code Edits
Every proposed change uses exact `<SEARCH>/<REPLACE>` blocks. PyOB utilizes a **Multi-Strategy Matcher**. If any block fails to align, the entire patch is rejected and auto-regenerated to prevent partial, broken edits.

### 📚 GitHub "Librarian" & Cloud HUD
PyOB acts as a professional developer. After verifying a fix, it:
- Creates a unique feature branch.
- Commits as **`pyob-bot`** to keep your history clean.
- Opens a **Pull Request** for your review.
- Provides a **Live Public URL** (via Pinggy) to monitor the HUD from your mobile device.

### 🛡️ 5-Layer Verification Pipeline
1. **Atomic XML Matching** — Strict anchor validation with **Smart Indent Alignment**.
2. **Syntactic "Broom"** — `ruff format` and `ruff check --fix` automatically clear "trash" errors.
3. **Symbolic Ripple Engine** — Automatically detects and queues downstream dependencies for updates.
4. **Context-Aware Self-Healing (PIR)** — Feeds the *original goal + error* back to the AI for repair.
5. **Runtime Smoke Test** — Launches the app for 10 seconds to monitor for tracebacks.

### 🌀 Recursive Self-Evolution
PyOB can target its own source code to optimize its engine logic. 
- **Recursive Safety Pods**: Shelters working source code in an external backup directory (`~/Documents/PYOB_Backups`).
- **Verified Hot-Reboot**: The engine tests the importability of its new code before performing a live restart.

### 🤝 Human-in-the-Loop Governance
Interactive terminal checkpoints locally, or **Headless Auto-Approval** in the cloud.
- **`AUGMENT_PROMPT`** — Inject instructions into the AI's mental process.
- **`EDIT_CODE` / `EDIT_XML`** — Polish proposed changes in your terminal.
- **`FULL_DIFF`** — View the complete unified diff in a pager.

---

### 🚀 Getting Started

PyOB is a **GitHub Marketplace Action**. You can run it locally as a CLI or in the cloud as a service.

#### **Option 1: The Cloud Service (Recommended)**
Add PyOB to any repository by creating `.github/workflows/pyob.yml`:
```yaml
uses: vicsanity623/PyOB@main
with:
  gemini_keys: ${{ secrets.PYOB_GEMINI_KEYS }}
```

#### **Option 2: Running from Source (Developers)**
1. **Setup Environment**:
```bash
python3.12 -m venv build_env
source build_env/bin/activate
pip install -e .
```
2. **Validate**: `./check.sh`
3. **Launch**: `pyob .`
4. **Dashboard**: `observer.HTML` or `http://localhost:5000/`
---

### 🎯 Quick Start Workflow

1.  **Targeting:** Provide the path to the project you want PyOB to manage.
2.  **Dashboard:** Open **`http://localhost:5000`** to watch the "Observer" dashboard in real-time.
3.  **Approve:** When PyOB proposes a fix or feature, review the diff in your terminal and hit `ENTER`.
4.  **Self-Evolution:** To have PyOB improve itself, target its own root directory: `pyob .`
5.  **Verification:** The system runs a 4-layer pipeline (XML Match → Lint → Runtime Test → Downstream Ripple Check) to ensure the code is functional.
6.  **Auto-Locking:** Any dependencies PyOB installs during auto-repair are automatically locked into your `requirements.txt`.

### 🔄 The Autonomous Loop
PyOB will:
1. 🔍 **Bootstrap** — Generate `ANALYSIS.md` (project map) and `SYMBOLS.json` (symbolic ledger).
2. 🎯 **Target** — Intelligently select the next file to review based on history and symbolic ripples.
3. 🔬 **Analyze** — Scan for bugs, type errors, and architectural bloat (>800 lines).
4. 💡 **Propose** — Generate a `PEER_REVIEW.md` (fixes) or `FEATURE.md` (new logic).
5. ✅ **Verify** — Perform a "Dry Run" and runtime check before finalizing any edit.
6. 🔗 **Cascade** — Detect and queue downstream dependency impacts for immediate follow-up.
7. 💾 **Persist** — Compress and update `MEMORY.md` to maintain context for the next iteration.

---

## 🏗️ Architecture

### Modular Engine Design
PyOB is built using a **Mixin-based architecture** to separate concerns and prevent context bloat:

| Component | File | Role |
|---|---|---|
| **Entrance Controller** | `entrance.py` | Master loop, Symbolic targeting, and Recursive Forge management. |
| **Auto Reviewer** | `autoreviewer.py` | Orchestrates the 6-phase pipeline and feature implementation. |
| **Core Utilities** | `core_utils.py` | LLM streaming, Smart Python detection, and Cyberpunk Logging. |
| **Prompts & Memory** | `prompts_and_memory.py` | 8 specialized prompt templates and Transactional Memory logic. |
| **Structure Parser** | `structure_parser.py` | High-fidelity AST parsing for Python/JS signatures. |

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ENTRANCE CONTROLLER                         │
│                        (entrance.py)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐   │
│  │ Target   │  │ Symbolic │  │ GitHub   │  │ Remote Sync    │   │
│  │ Selector │  │ Engine   │  │ Librarian│  │ (Auto-Reboot)  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬─────────┘   │
│       │              │             │               │            │
│       ▼              ▼             ▼               ▼            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              TARGETED REVIEWER PIPELINE                 │    │
│  │          (Orchestrates Multi-File Services)             │    │
│  └─────────────────────┬───────────────────────────────────┘    │
│                        │                                        │
│       ┌────────────────┴────────────────┬────────────────┐      │
│       ▼                                 ▼                ▼      │
│ ┌──────────────┐                 ┌──────────────┐ ┌──────────────┐
│ │ CODE PARSER  │                 │  DASHBOARD   │ │  AUTO-HEAL   │
│ │ (ast_parser) │                 │  (Architect) │ │  (Runtime)   │
│ └──────────────┘                 └──────────────┘ └──────────────┘
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AUTO REVIEWER ENGINE                      │
│                        (autoreviewer.py)                        │
│  ┌───────────┐  ┌────────────────────────────────────────────┐  │
│  │  6-Phase  │  │          REVIEWER MIXINS                   │  │
│  │ Pipeline  │  │        (reviewer_mixins.py)                │  │
│  └─────┬─────┘  │ • Linter Fix Loop  • Runtime Verifier      │  │
│        │        │ • Feature Logic    • XML Matcher           │  │
│        │        └──────────────────────┬─────────────────────┘  │
│        │                               │                        │
│  ┌─────▼───────────────────────────────▼─────────────────────┐  │
│  │                    CORE UTILITIES MIXIN                   │  │
│  │                      (core_utils.py)                      │  │
│  │  • Gemini / Ollama Streaming   • Smart Key Rotation       │  │
│  │  • Headless Auto-Approval      • Workspace Backup/Restore │  │
│  └─────────────────────────────────────┬─────────────────────┘  │
│                                        │                        │
│  ┌─────────────────────────────────────▼─────────────────────┐  │
│  │                 PROMPTS & MEMORY MIXIN                    │  │
│  │                 (prompts_and_memory.py)                   │  │
│  │  • Template Management        • Rich Context Builder      │  │
│  │  • Transactional Memory Update • Aggressive Refactoring    │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Persistent State Files

| File | Purpose | Managed By |
|---|---|---|
| `ANALYSIS.md` | Recursive project map with file summaries and structural dropdowns | `entrance.py` |
| `SYMBOLS.json` | Dependency graph: definitions → files, references → call sites | `entrance.py` |
| `MEMORY.md` | Synthesized session memory; auto-refactored every 2 iterations | `prompts_and_memory.py` |
| `HISTORY.md` | Append-only ledger of every unified diff applied to the project | `entrance.py` |
| `PEER_REVIEW.md` | Generated bug fix proposals (created during Phase 1) | `autoreviewer.py` |
| `FEATURE.md` | Generated feature proposals (created during Phase 2) | `autoreviewer.py` |

---

## 🔄 Pipeline Phases

### Phase 1: Initial Assessment & Codebase Scan
Scans all supported files (`.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.sh`), runs linters, detects lazy code patterns (e.g., `typing.Any`), and generates surgical patch proposals.

### Phase 2: Feature Proposal
If no bugs were found in Phase 1, the AI analyzes a randomly selected file and proposes one interactive, user-facing feature with a `<SNIPPET>` code block.

### Phase 3: Downstream Cascade Check
After any modification, PyOB runs `mypy` across the workspace. If type errors surface in dependent files, it generates **cascade fixes** using the `PCF.md` prompt template.

### Phase 4: Runtime Verification
Identifies the project entry point (`if __name__ == "__main__":` or `main.py`/`app.py`), launches it for 10 seconds, and monitors `stdout`/`stderr` for crashes. Auto-installs missing pip packages on `ModuleNotFoundError`.

### Phase 5: Memory Update
Synthesizes all session actions into `MEMORY.md` using the `UM.md` prompt template, preserving architectural decisions and dependency mappings.

### Phase 6: Memory Refactoring (Every 3rd Iteration)
Aggressively summarizes `MEMORY.md` to prevent context bloat, consolidating repetitive logs into a concise knowledge base.

---

## 📋 Prompt Templates

PyOB uses 8 specialized prompt templates, auto-generated as `.md` files in the target directory:

| Template | Full Name | Purpose |
|---|---|---|
| `PP.md` | Patch Prompt | Code review and bug fix generation |
| `PF.md` | Propose Feature | Interactive feature proposal |
| `IF.md` | Implement Feature | Surgical feature implementation |
| `ALF.md` | Auto Linter Fix | Syntax error repair |
| `FRE.md` | Fix Runtime Error | Runtime crash diagnosis and repair |
| `PIR.md` | Post-Implementation Repair | Context-aware error recovery (knows the original goal) |
| `PCF.md` | Propose Cascade Fix | Cross-file dependency repair |
| `UM.md` | Update Memory | Memory synthesis and consolidation |
| `RM.md` | Refactor Memory | Aggressive memory summarization |

---

## ⚙️ Configuration

### API Keys
Gemini API keys are configured in `core_utils.py` in the `GEMINI_API_KEYS` list. Multiple keys enable automatic rotation and rate-limit resilience.

### Models
| Setting | Default | Location |
|---|---|---|
| Gemini Model | `gemini-2.5-flash` | `core_utils.py` → `GEMINI_MODEL` |
| Local Model | `qwen3-coder:30b` | `core_utils.py` → `LOCAL_MODEL` |
| Temperature | `0.1` | `core_utils.py` → `stream_gemini()` / `stream_ollama()` |

### Ignored Paths
PyOB automatically skips certain directories and files to avoid self-modification and virtual environments:

<details>
<summary><b>Ignored Directories</b></summary>

`.git`, `autovenv`, `venv`, `.venv`, `code`, `.mypy_cache`, `.ruff_cache`, `patch_test`, `env`, `__pycache__`, `node_modules`, `.vscode`, `.idea`, `other_dir`

</details>

<details>
<summary><b>Ignored Files</b></summary>

`core_utils.py`, `prompts_and_memory.py`, `autoreviewer.py`, `entrance.py`, all prompt templates (`ALF.md`, `FRE.md`, etc.), `sw.js`, `manifest.json`, `package-lock.json`, `auto.py`, `any_other_file_to_ignore.filetype`

</details>

### Supported File Types
`.py` · `.js` · `.ts` · `.html` · `.css` · `.json` · `.sh`

---

## 🧪 How the XML Edit Engine Works

PyOB's edit engine is a multi-strategy matcher that ensures reliable code modifications:

```
1. Exact Match        →  Direct string replacement
2. Stripped Match     →  Leading/trailing whitespace tolerance
3. Normalized Match   →  Ignores comments and collapses whitespace
4. Regex Fuzzy Match  →  Line-by-line regex matching with indent tolerance
5. Robust Line Match  →  Stripped line-by-line content comparison
```

If all 5 strategies fail for any `<SEARCH>` block, the **entire multi-block edit is rejected** and the AI is asked to regenerate.

### Smart Indent Alignment
The engine detects the base indentation of both the `<SEARCH>` and `<REPLACE>` blocks, then re-aligns the replacement to match the source file's indentation style — preventing whitespace corruption.

---

## 🛡️ Safety Mechanisms

| Mechanism | Description |
|---|---|
| **Atomic Rollback** | Snapshots the entire workspace before every edit; instantly restores from backup if any verification layer fails. |
| **Librarian Isolation** | Edits are never pushed directly to `main`. The "Librarian" creates unique feature branches and Pull Requests for safe review. |
| **Remote Sync Guard** | Automatically fetches and merges updates from `origin/main` before every loop to prevent working on stale code. |
| **Headless Auto-Approval** | Detects CI environments (GitHub Actions) and auto-proceeds through checkpoints while maintaining full verification. |
| **Smart Sleep Backoff** | Dynamically calculates the exact seconds until the next Gemini API key is available, preventing "API Spam" and idle waste. |
| **Import Preservation** | AST-based logic that ensures the AI doesn't accidentally drop crucial imports during surgical refactors. |
| **Circular Safety** | Prevents "Brain Loops" by using `TYPE_CHECKING` guards during cross-module initialization. |

---

## 📺 The Architect HUD (Observer Dashboard)

PyOB includes a built-in, real-time **Cyberpunk HUD**. While the engine runs in the terminal (locally or in the cloud), you can monitor the "Ouroboros" through a state-of-the-art web interface.

- **Responsive SOTA UI**: A modern, glassmorphic HUD designed for both desktop and mobile monitoring.
- **Manual Target Override**: Use the HUD to force the engine to target a specific file for the next iteration.
- **Pending Patch Preview**: Real-time visualization of AI thoughts and proposed patches before they are committed.
- **Symbolic Ledger Stats**: Live tracking of the dependency graph size and symbolic ripple counts.
- **Cloud Tunneling**: When running on GitHub Actions, PyOB automatically generates a secure **Live Tunnel URL** (via Pinggy) so you can monitor your bot's progress from your phone.
- **URL**: `http://localhost:5000` (Local) or a dynamic `.pinggy.link` (Cloud).

---

## 📖 Full Documentation

For in-depth technical documentation covering the verification pipeline, symbolic dependency management, prompt engineering, and more, see the **[Technical Documentation](docs/DOCUMENTATION.md)**.

---

## 📁 Project Structure

```text
PyOB/
├── pyproject.toml           # ⚙️ Package config, dependencies, and CLI entry point
├── check.sh                 # 🧪 5-layer validation suite (Ruff + Mypy + Pytest)
├── Dockerfile               # 🐳 Cloud runtime environment for GitHub Actions
├── action.yml               # 🏗️ GitHub Marketplace Action manifest
├── src/
│   └── pyob/
│       ├── pyob_launcher.py     # 🚀 Main entry point & environment setup
│       ├── entrance.py          # 🧠 Master loop & GitHub Librarian
│       ├── autoreviewer.py      # 🔧 Pipeline orchestration
│       ├── reviewer_mixins.py   # 🛠️ Implementation logic for linters & healing
│       ├── pyob_code_parser.py  # 🔬 AST/Regex structure analysis
│       ├── pyob_dashboard.py    # 📺 Architect HUD (Observer UI)
│       ├── core_utils.py        # ⚙️ LLM streaming & XML edit engine
│       └── prompts_and_memory.py # 📝 Template & persistence management
└── tests/                   # 🧪 Engine unit tests
```

### Generated Files (The `.pyob/` Data Vault)

To keep your project root clean, PyOB now stores all its persistent metadata and prompt templates in a hidden directory.

```text
your-project/
├── .pyob/                   # 📂 Hidden Metadata Vault
│   ├── ANALYSIS.md          # 🗺️ Intelligent Project Map & Summaries
│   ├── SYMBOLS.json         # 🔗 Symbolic Dependency Ledger
│   ├── MEMORY.md            # 🧠 Long-term AI Logic State
│   ├── HISTORY.md           # 📜 Truncated Unified Diff Ledger
│   ├── observer.html        # 📺 Local Dashboard HUD source
│   └── [PP.md, IF.md...]    # 📋 8 Specialized Prompt Engines
├── PEER_REVIEW.md           # 🔍 Active Bug Fix Proposals
├── FEATURE.md               # 💡 Active Feature Proposals
├── requirements.txt         # 🔒 Auto-locked dependencies
└── [your source files]
```

---

<div align="center">

**Built with surgical precision.** 🦅

</div>

# PyOB — Complete Technical Documentation

> **Version**: 0.3.1 · **Last Updated**: March 2026
> **Architecture**: Python 3.12+ · Mixin-Based Package · GitHub Marketplace Action

---

## Table of Contents

1. [System Philosophy](#1-system-philosophy-constrained-surgical-autonomy)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Reference](#3-module-reference)
   - 3.1 [pyob_launcher.py — Entry Point](#31-pyob_launcherpy--main-cli-entry-point)
   - 3.2 [entrance.py — The Entrance Controller](#32-entrancepy--the-entrance-controller)
   - 3.3 [autoreviewer.py — The Pipeline Orchestrator](#33-autoreviewerpy--the-auto-reviewer)
   - 3.4 [reviewer_mixins.py — Implementation Muscles](#34-reviewer_mixinspy--engine-implementations)
   - 3.5 [pyob_code_parser.py — Structural Analysis](#35-pyob_code_parserpy--structural-analysis)
   - 3.6 [pyob_dashboard.py — SOTA Architect HUD](#36-pyob_dashboardpy--the-architect-hud)
   - 3.7 [core_utils.py — Cloud-Aware Foundation](#37-core_utilspy--core-utilities-mixin)
4. [The Verification & Healing Pipeline](#4-the-verification--healing-pipeline)
5. [Symbolic Dependency Management](#5-symbolic-dependency-management)
6. [The XML Edit Engine](#6-the-xml-edit-engine)
7. [The GitHub Librarian](#7-the-github-librarian-integration)
8. [Headless & Cloud Autonomy](#8-headless--cloud-autonomy)
9. [LLM Backend & Smart Sleep Backoff](#9-llm-backend--resilience)
10. [Persistence & State Vault (.pyob/)](#10-persistence--state-management)
11. [Safety & Rollback Mechanisms](#11-safety--rollback-mechanisms)
12. [Marketplace & Docker Infrastructure](#12-marketplace--docker-infrastructure)
13. [Internal Constants & Rulesets](#13-internal-constants--defaults)
14. [Operational Workflow](#14-operational-workflow)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. System Philosophy: Constrained Surgical Autonomy

PyOB is an autonomous agent built on **constrained agency**. Unlike chat-based assistants that require constant prompting, PyOB is a self-driven engine that operates within a strict "Safety Cage" defined by:

1. **Surgical Patching** — Patches are applied via `<SEARCH>/<REPLACE>` blocks limited to 2-5 line anchors.
2. **Atomic Commits** — Changes are isolated in unique Git branches and submitted as PRs via the Librarian.
3. **Multi-Step Verification** — Every edit must pass a 5-layer gate (XML match → Linter → Mypy → PIR → Smoke Test).
4. **Self-Evolution** — The engine is recursive; it can identify its own logic flaws and refactor its source code.

---

## 2. Architecture Overview

### Modular Package Structure
PyOB has transitioned from a script collection to a standardized Python package located in `src/pyob/`.

```text
CoreUtilsMixin (core_utils.py)
├── Provides: Smart Sleep, Headless Approval, XML Engine, LLM Streaming
│
PromptsAndMemoryMixin (prompts_and_memory.py)
├── Provides: Rule-based Templates (Rule 7: No src. imports), CRUD Memory
│
ValidationMixin + FeatureOperationsMixin (reviewer_mixins.py)
├── Provides: Ruff/Mypy validation, Runtime Auto-Heal, XML Implementation
│
AutoReviewer(All Mixins) (autoreviewer.py)
├── Provides: 6-Phase orchestrator logic
│
EntranceController (entrance.py)
├── Provides: Master loop, Remote Sync, Librarian PR logic, Reboot Flag
```

### System Data Flow (Cloud Service Mode)
```
[User / Schedule] ─▶ [GitHub Action] ─▶ [Docker Container]
                                              │
                    ┌─────────────────────────┴────────────────────────┐
                    │      ENTRANCE CONTROLLER (Master Loop)           │
                    │ 1. Sync Remote Main  2. Pick Target  3. Backup   │
                    └───────────┬────────────────────────────┬─────────┘
                                ▼                            ▼
                    ┌──────────────────────┐      ┌────────────────────┐
                    │   AUTO REVIEWER      │      │    LIBRARIAN       │
                    │ (6-Phase Pipeline)   │      │ (Branch/Commit/PR) │
                    └───────────┬──────────┘      └────────────────────┘
                                ▼
                    ┌──────────────────────────────────────────────────┐
                    │           VERIFICATION & HEALING                 │
                    │ [Ruff --fix] ─▶ [Mypy] ─▶ [10s Smoke Test]      │
                    └──────────────────────────────────────────────────┘
```

---

## 3. Module Reference

### 3.1 `pyob_launcher.py` — Main CLI Entry Point
The environment bootstrapper. It configures the runtime, handles macOS terminal re-launching, and detects "Headless" environments.

| Method | Description |
|---|---|
| `load_config` | Pulls keys from `~/.pyob_config` (Local) or `os.environ` (Cloud). Detects non-TTY to skip prompts. |
| `ensure_terminal` | macOS-specific logic to force PyOB into a visible Terminal window for DMG users. |
| `main` | Entry point. Detects macOS app bundle paths and ignores them to ensure clean targeting. |

### 3.2 `entrance.py` — The Entrance Controller
The master orchestrator. Manages symbolic targeting, Git lifecycle, and Hot-Reboots.

| Method | Description |
|---|---|
| `run_master_loop` | Infinite loop with `sync_with_remote` check. Manages the `self_evolved_flag`. |
| `sync_with_remote` | Fetches `origin/main`. If behind, performs a merge. Triggers reboot if engine files change. |
| `handle_git_librarian` | Creates branch `pyob-evolution-vX-timestamp`, commits as `pyob-bot`, and opens PR. |
| `reboot_pyob` | **Verified Hot-Reboot:** Tests if new code is importable before calling `os.execv` to restart. |

### 3.3 `autoreviewer.py` — The Auto Reviewer
The high-level pipeline orchestrator. Ties together the specialized mixins into the 6-phase autonomous cycle.

### 3.4 `reviewer_mixins.py` — Engine Implementations
Separates "Muscle" from "Brain."

- **`ValidationMixin`**: Runs `ruff format`, then `ruff check --fix`. If errors remain, it triggers the PIR loop.
- **`FeatureOperationsMixin`**: The heavy-duty XML matcher. Interprets AI proposals and writes them to `PEER_REVIEW.md`.

### 3.5 `pyob_code_parser.py` — Structural Analysis
A high-fidelity analysis tool that uses **AST (Python)** and **Regex (JS/CSS)** to map the project architecture. It generates the `<details>` dropdowns seen in `ANALYSIS.md`.

### 3.6 `pyob_dashboard.py` — The Architect HUD
A `BaseHTTPRequestHandler` that serves the SOTA Cyberpunk HUD. Features glassmorphism, responsive mobile layout, and real-time AJAX stats updates.

---

## 4. The Verification & Healing Pipeline

PyOB follows a "Proactive Defense" model to ensure code stability.

### Layer 1: Atomic XML Match
Edits are binary: either every block in a response matches perfectly, or the entire iteration is discarded.

### Layer 2: Syntactic "Broom"
1. **`ruff format`**: Normalizes all whitespace.
2. **`ruff check --fix`**: Automatically clears unused imports and variables without costing AI tokens.
3. **Remaining Errors**: Grouped by file and fed into the AI for surgical repair.

### Layer 3: Runtime Smoke Test
- Locates the entry point via `_find_entry_file`.
- Launches the process for 10 seconds.
- **Auto-Dependency Locking**: If a `ModuleNotFoundError` is detected, PyOB runs `pip install` and immediately updates `requirements.txt`.

---

## 5. Symbolic Dependency Management

### `SYMBOLS.json` Ledger
PyOB maintains a mapping of **Definitions** (where a function/class is born) to **References** (where it is used).

### Symbolic Ripple Engine
1. When a file is edited, the engine identifies changed symbols.
2. It looks up the ledger to see if those symbols are "Definitions."
3. It finds all "References" in other files.
4. **Cascade Queue:** Impacted files are prioritized for the next iteration, with the original change-diff provided as mandatory context.

---

## 6. The XML Edit Engine

### Multi-Strategy Matching
`apply_xml_edits` attempts 5 strategies per block:
1. **Exact** (Literal)
2. **Stripped** (Newline tolerance)
3. **Normalized** (Comment/Whitespace stripping)
4. **Regex Fuzzy** (Indentation tolerance)
5. **Robust Line Match** (Content-only line comparison)

### Smart Indent Alignment
The engine detects the target line's indentation and re-aligns the AI's `<REPLACE>` block to match, preventing the "Unexpected Indentation" errors common in Python agents.

---

## 7. The GitHub Librarian Integration

PyOB acts as a professional developer through the **Librarian** module:

- **Isolated Branches:** Every change is pushed to a unique branch.
- **Bot Identity:** Commits are attributed to `pyob-bot` using the `BOT_GITHUB_TOKEN`.
- **Automated PRs:** Uses the GitHub CLI (`gh`) to open Pull Requests targeting `main`.
- **PR Body:** Includes the AI's `<THOUGHT>` process as the PR description for human review.

---

## 8. Headless & Cloud Autonomy

PyOB detects when it is running in **GitHub Actions** (via the `GITHUB_ACTIONS=true` env var):

- **Auto-Approval:** Bypasses "Press ENTER to apply" prompts.
- **Non-TTY Safety:** Skips all `termios` and `input()` calls to prevent `EOFError` or `ioctl` crashes.
- **Cloud Tunneling:** Starts a background **Pinggy** tunnel to provide a public URL for the dashboard HUD.

---

## 9. LLM Backend & Smart Sleep Backoff

### Multi-Key Key Rotation
PyOB rotates through a pool of up to 10 Gemini API keys. Keys that hit a `429 Rate Limit` are quarantined for 20 minutes.

### Smart Sleep Backoff
When all keys are rate-limited, the engine calculates:
`sleep_duration = min(key_cooldowns) - current_time`
The bot "naps" for the exact number of seconds until the first key is available, ensuring zero waste of Cloud Runner minutes.

---

## 10. Persistence & State Management (.pyob/)

All project metadata is stored in the hidden `.pyob/` vault to prevent root directory clutter.

| File | Purpose |
|---|---|
| `.pyob/ANALYSIS.md` | Persistent map used by the AI to select targets. |
| `.pyob/SYMBOLS.json` | The symbolic dependency graph. |
| `.pyob/MEMORY.md` | Transactional AI memory; refactored every 2 iterations to prevent context bloat. |
| `.pyob/HISTORY.md` | Detailed ledger of applied patches. |

---

## 11. Safety & Rollback Mechanisms

- **External Safety Pods:** Before editing an "Engine File" (like `entrance.py`), PyOB shelters a copy of the current source in `~/Documents/PYOB_Backups/`.
- **Workspace Backup:** Every iteration starts with an in-memory snapshot of the entire project.
- **Atomic Rollback:** If any verification layer (Linter, Mypy, or Runtime) fails 3 times, the entire workspace is restored to the backup.

---

## 12. Marketplace & Docker Infrastructure

### Marketplace Action
PyOB is a containerized GitHub Action (`action.yml`). It uses a `Dockerfile` based on `python:3.12-slim` with `git`, `curl`, and `gh` pre-installed.

### Docker Environment
The Docker container maps the user's repository to `/github/workspace`, allowing PyOB to operate on the files as if it were a local CLI tool.

---

## 13. Internal Constants & Rulesets

### Mandatory Import Rule (Rule 7)
The AI is strictly prohibited from using the `src.` prefix in imports.
- **Correct:** `from pyob.core_utils import ...`
- **Incorrect:** `from src.pyob.core_utils import ...`

### Indentation Guard (Rule 6)
Deletions must leave a placeholder comment (e.g., `# [Logic moved to new module]`) to maintain Python's indentation integrity.

---

## 14. Operational Workflow

1. **Remote Sync:** Pull latest merges from GitHub.
2. **Genesis / Update:** Build or refresh `ANALYSIS.md` and `SYMBOLS.json`.
3. **Targeting:** Select file via AI or the `Cascade Queue`.
4. **Pipeline:** Scan → Propose → Verify → Auto-Heal.
5. **Librarian:** Push Branch → Open PR.
6. **Self-Evolution:** If engine changed, verify importability and Hot-Reboot.

---

## 15. Troubleshooting

### `ModuleNotFoundError: No module named 'src'`
**Cause:** AI incorrectly added `src.` to an import statement.
**Fix:** Remove the `src.` prefix. Ensure `pyproject.toml` is installed via `pip install -e .`.

### `EOFError: EOF when reading a line`
**Cause:** PyOB tried to call `input()` in a cloud environment.
**Fix:** Ensure `sys.stdin.isatty()` checks are present in the launcher.

### `termios.error: Inappropriate ioctl for device`
**Cause:** `get_user_approval` tried to manipulate a non-existent keyboard.
**Fix:** The "Headless Auto-Approval" logic in `core_utils.py` handles this.

---
> **PyOB** — The engine that builds itself, with surgical precision. 🦅

---

## Star History

<a href="https://www.star-history.com/?repos=vicsanity623%2FPyOB&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=vicsanity623/PyOB&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=vicsanity623/PyOB&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=vicsanity623/PyOB&type=date&legend=top-left" />
 </picture>
</a>
