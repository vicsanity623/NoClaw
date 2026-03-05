<p align="center">
  <img src="no-claw.png" alt="NoClaw" width="200" />
</p>
<div align="center">

# 🐙 NoClaw

### The Self-Healing, Symbolic Autonomous Code Architect

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-8E75B2?style=for-the-badge&logo=google-gemini&logoColor=white)](https://ai.google.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

**NoClaw is a high-fidelity autonomous agent that performs surgical code modifications, cross-file dependency tracking, and self-healing verification — all without destroying your codebase.**

[Getting Started](#-getting-started) · [How It Works](#phase-1-initial-assessment--codebase-scan) · [Architecture](#system-overview) · [Documentation](#noclaw--complete-technical-documentation)

---

</div>

## 🧠 What is NoClaw?

NoClaw is an **autonomous code review and feature engineering system** that continuously analyzes, patches, and evolves your codebase through a multi-stage verification pipeline. Unlike "black-box" coding assistants that rewrite entire files, NoClaw operates with **surgical XML-based edits**, a **persistent symbolic dependency ledger**, and **human-in-the-loop checkpoints** — ensuring your project is never left in a broken state.

### Key Differentiators

| Feature | Traditional AI Assistants | NoClaw |
|---|---|---|
| **Edit Strategy** | Full file rewrites | Surgical `<SEARCH>/<REPLACE>` XML blocks |
| **Dependency Awareness** | None | Symbolic ledger (`SYMBOLS.json`) with ripple detection |
| **Error Recovery** | Manual | Context-aware self-healing with auto-rollback |
| **Verification** | None | 4-layer pipeline: XML matching → linting → PIR → runtime test |
| **State Persistence** | Stateless | `MEMORY.md`, `ANALYSIS.md`, `HISTORY.md`, `SYMBOLS.json` |
| **API Resilience** | Single key, fails on rate limit | Multi-key rotation with automatic local LLM fallback |

---

## ✨ Core Features

### 🔬 Surgical XML-Based Code Edits
Every proposed change uses exact `<SEARCH>/<REPLACE>` blocks matched against the source. If even one anchor fails to align, the **entire patch is rejected** and auto-regenerated — never leaving partial, broken edits.

### 🔗 Symbolic Ripple Engine
NoClaw maintains a live dependency graph (`SYMBOLS.json`). When a function signature changes in `logic.py`, every file that references it is automatically queued for synchronized updates via the **cascade system**.

### 🛡️ 4-Layer Verification Pipeline
1. **Atomic XML Matching** — Strict `<SEARCH>` anchor validation
2. **Syntactic Validation** — `ruff` (Python), `node --check` (JS), brace-balancing (CSS)
3. **Context-Aware Self-Healing (PIR)** — Feeds the *original goal + error + broken code* back to the LLM
4. **10-Second Runtime Smoke Test** — Launches your app and monitors for crashes/tracebacks

### 🤝 Human-in-the-Loop Governance
Interactive terminal checkpoints at every stage:
- **`AUGMENT_PROMPT`** — Inject additional instructions into the AI prompt
- **`EDIT_PROMPT`** — Open the full prompt in your editor for manual refinement
- **`EDIT_XML`** — Refine the AI's proposed XML edit blocks before application
- **`EDIT_CODE`** — Polish the AI's proposed code in your terminal editor (Nano/Vim)
- **`FULL_DIFF`** — View the complete unified diff in a pager before deciding
- **`REGENERATE`** — Reject the proposal and ask the AI to try again
- **`SKIP`** — Cancel the current operation entirely

### 🔄 Hybrid LLM Backend with Auto-Failover
- **Primary**: Gemini 2.5 Flash with multi-key rotation and 429-aware cooldowns
- **Fallback**: Local Ollama (`qwen3-coder:30b`) activates automatically when all API keys are rate-limited
- **Progress Spinner**: Real-time token estimation with animated progress bar during inference

---

## 🚀 Getting Started

NoClaw can be used either as a pre-compiled standalone application (Recommended) or by running the source code directly.

### 📋 Prerequisites

Regardless of installation method, NoClaw utilizes external tools for code verification and local LLM fallback.

| Requirement | Purpose | Required? |
|---|---|---|
| **[Ollama](https://ollama.ai)** | Local model server (fallback) | ⚡ Recommended |
| **`ruff`** | Python linting & formatting | ⚡ Recommended |
| **`mypy`** | Static type checking | ⚡ Recommended |
| **Node.js** | JavaScript syntax validation | Optional |

---

### 📦 Option 1: Standalone Installation (End-Users)

Download the latest pre-built binaries from the **[Releases Page](https://github.com/vicsanity623/NoClaw/releases)**.

#### **macOS (.dmg)**
1. **Download and Mount:** Open `NoClaw-v0.1.0.dmg`.
2. **Install:** Drag the **NoClaw** icon into your `/Applications` folder.
3. **Launch:** Open NoClaw via Spotlight (`Cmd + Space` > "NoClaw").
4. **Setup:** A Terminal window will open automatically. Follow the prompts to enter your Gemini API keys and select your AI models. These settings are saved to `~/.noclaw_config`.

#### **Windows (.exe)**
1. **Download:** Save `NoClaw.exe` to a known directory.
2. **Launch:** Double-click the executable or run it via PowerShell/CMD.
3. **Setup:** Follow the on-screen prompts to configure your API keys and model preferences.

---

### 🛠️ Option 2: Running from Source (Developers)

Use this method if you wish to modify NoClaw or contribute to its development.

#### **1. Environment Setup**
```bash
# Clone the repository
git clone https://github.com/vicsanity623/NoClaw.git
cd NoClaw

# Install Python dependencies (Python 3.10+ required)
pip install requests ollama

# Install linting tools for verification pipeline
pip install ruff mypy
```

#### **2. Local Model Preparation**
If you intend to use the local fallback feature, pull the recommended model:
```bash
ollama pull qwen3-coder:30b
```

#### **3. Execution**
Run the launcher directly. On the first run, you will be prompted to configure your API keys and model settings.
```bash
python noclaw_launcher.py
```

---

### 🎯 Quick Start Workflow

Once launched and configured, NoClaw follows this operational flow:

1.  **Project Targeting:** Enter the full path to the project you wish to analyze.
2.  **Bootstrap:** NoClaw generates `ANALYSIS.md` (project map) and `SYMBOLS.json` (dependency graph).
3.  **Autonomous Loop:** The system targets files based on dependency ripple logic.
4.  **Interactive Checkpoints:** NoClaw will pause for your approval before applying any surgical XML edits.
    -   Type `FULL_DIFF` to see the proposed changes.
    -   Type `EDIT_XML` or `EDIT_CODE` to refine the output.
    -   Hit `ENTER` to commit the patch.
5.  **Verification:** The system runs the 4-layer pipeline (XML Match → Lint → PIR → Runtime Test) to ensure the code is functional.
6.  **Persistence:** Your `MEMORY.md` and `HISTORY.md` are updated to maintain context for the next iteration.

NoClaw will:
1. 🔍 **Bootstrap** — Generate `ANALYSIS.md` (project map) and `SYMBOLS.json` (dependency graph)
2. 🎯 **Target** — Intelligently select the next file to review based on history and dependencies
3. 🔬 **Analyze** — Scan for bugs, lint errors, and architectural gaps
4. 💡 **Propose** — Generate a `PEER_REVIEW.md` (bug fixes) or `FEATURE.md` (new features)
5. ⏸️ **Checkpoint** — Wait for your approval before applying any changes
6. ✅ **Verify** — Run the 4-layer verification pipeline and auto-heal if needed
7. 🔗 **Cascade** — Detect and queue downstream dependency impacts
8. 💾 **Persist** — Update `MEMORY.md` and `HISTORY.md` with session context
9. 🔁 **Iterate** — Loop back to step 2 with a 2-minute cooldown

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ENTRANCE CONTROLLER                         │
│                      (entrance.py)                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Target   │  │ Analysis │  │ Symbolic │  │   History    │   │
│  │ Selector │  │ Builder  │  │ Ripple   │  │   Tracker    │   │
│  │          │  │          │  │ Engine   │  │              │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       │              │             │               │           │
│       ▼              ▼             ▼               ▼           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TARGETED REVIEWER                          │   │
│  │          (Scoped AutoReviewer Instance)                  │   │
│  └─────────────────────┬───────────────────────────────────┘   │
│                        │                                       │
│  ┌─────────────────────▼───────────────────────────────────┐   │
│  │            FINAL VERIFICATION & HEALING                  │   │
│  │         (10s Runtime Test + Auto-Rollback)               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AUTO REVIEWER                             │
│                     (autoreviewer.py)                            │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐   │
│  │  6-Phase  │  │ XML Edit │  │  Linter   │  │  Runtime   │   │
│  │ Pipeline  │  │  Engine  │  │ Fix Loop  │  │ Verifier   │   │
│  └─────┬─────┘  └────┬─────┘  └─────┬─────┘  └─────┬──────┘   │
│        │              │              │              │           │
│  ┌─────▼──────────────▼──────────────▼──────────────▼──────┐   │
│  │               CORE UTILITIES MIXIN                       │   │
│  │                 (core_utils.py)                           │   │
│  │  • Gemini Streaming  • Ollama Streaming  • Key Rotation  │   │
│  │  • User Approval     • Workspace Backup  • XML Parser    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            PROMPTS & MEMORY MIXIN                        │   │
│  │            (prompts_and_memory.py)                        │   │
│  │  • Template Management    • Rich Context Builder         │   │
│  │  • Memory Update/Refactor • Impactful History Extraction │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Role |
|---|---|---|
| **Entrance Controller** | `entrance.py` | Orchestrates the master loop, symbolic targeting, ripple detection, and final verification |
| **Auto Reviewer** | `autoreviewer.py` | Executes the 6-phase pipeline: scan → analyze → propose → implement → verify → persist |
| **Core Utilities** | `core_utils.py` | LLM streaming, API key management, XML edit engine, workspace backup/restore |
| **Prompts & Memory** | `prompts_and_memory.py` | Prompt template management, memory persistence, and context enrichment |

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
After any modification, NoClaw runs `mypy` across the workspace. If type errors surface in dependent files, it generates **cascade fixes** using the `PCF.md` prompt template.

### Phase 4: Runtime Verification
Identifies the project entry point (`if __name__ == "__main__":` or `main.py`/`app.py`), launches it for 10 seconds, and monitors `stdout`/`stderr` for crashes. Auto-installs missing pip packages on `ModuleNotFoundError`.

### Phase 5: Memory Update
Synthesizes all session actions into `MEMORY.md` using the `UM.md` prompt template, preserving architectural decisions and dependency mappings.

### Phase 6: Memory Refactoring (Every 3rd Iteration)
Aggressively summarizes `MEMORY.md` to prevent context bloat, consolidating repetitive logs into a concise knowledge base.

---

## 📋 Prompt Templates

NoClaw uses 8 specialized prompt templates, auto-generated as `.md` files in the target directory:

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
NoClaw automatically skips certain directories and files to avoid self-modification and virtual environments:

<details>
<summary><b>Ignored Directories</b></summary>

`.git`, `autovenv`, `venv`, `.venv`, `code`, `.mypy_cache`, `.ruff_cache`, `patch_test`, `env`, `__pycache__`, `node_modules`, `.vscode`, `.idea`

</details>

<details>
<summary><b>Ignored Files</b></summary>

`core_utils.py`, `prompts_and_memory.py`, `autoreviewer.py`, `entrance.py`, all prompt templates (`ALF.md`, `FRE.md`, etc.), `sw.js`, `manifest.json`, `package-lock.json`, `auto.py`

</details>

### Supported File Types
`.py` · `.js` · `.ts` · `.html` · `.css` · `.json` · `.sh`

---

## 🧪 How the XML Edit Engine Works

NoClaw's edit engine is a multi-strategy matcher that ensures reliable code modifications:

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
| **Workspace Backup** | Full in-memory snapshot before every modification attempt |
| **Atomic Rollback** | Restores the entire workspace if verification fails |
| **Import Preservation** | AST-based import retention ensures no imports are accidentally deleted |
| **Cascaded Healing** | Downstream type errors trigger automatic synchronized repairs |
| **Rate-Limit Quarantine** | 429'd API keys get a 20-minute timeout; system auto-falls back to local LLM |
| **Timeout Protection** | All user prompts have configurable timeouts (default: 220s) with auto-proceed |

---

## 📖 Full Documentation

For in-depth technical documentation covering the verification pipeline, symbolic dependency management, prompt engineering, and more, see the **[Technical Documentation](docs/DOCUMENTATION.md)**.

---

## 📁 Project Structure

```
NoClaw/
├── entrance.py              # 🧠 Entrance Controller — master loop & symbolic orchestration
├── autoreviewer.py          # 🔧 Auto Reviewer — 6-phase pipeline & edit engine
├── core_utils.py            # ⚙️ Core Utilities — LLM streaming, XML parser, key rotation
├── prompts_and_memory.py    # 📝 Prompts & Memory — template management & persistence
├── docs/
│   └── DOCUMENTATION.md     # 📖 Full technical documentation
└── README.md                # 📄 This file
```

### Generated Files (in your target project)

```
your-project/
├── ANALYSIS.md              # 🗺️ Auto-generated project map
├── SYMBOLS.json             # 🔗 Dependency graph
├── MEMORY.md                # 🧠 Persistent session memory
├── HISTORY.md               # 📜 Change history ledger
├── PEER_REVIEW.md           # 🔍 Pending bug fix proposals (temporary)
├── FEATURE.md               # 💡 Pending feature proposals (temporary)
├── FAILED_PEER_REVIEW.md    # ❌ Rolled-back bug fixes (for debugging)
├── FAILED_FEATURE.md        # ❌ Rolled-back features (for debugging)
├── PP.md, PF.md, IF.md ...  # 📋 Prompt templates (auto-generated)
└── [your source files]
```

---

<div align="center">

**Built with surgical precision.** 🦅

</div>

# NoClaw — Complete Technical Documentation

> **Version**: 1.0 · **Last Updated**: March 2026
> **Architecture**: Python 3.10+ · Gemini 2.5 Flash / Ollama Local LLM

---

## Table of Contents

1. [System Philosophy](#1-system-philosophy-constrained-surgical-autonomy)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Reference](#3-module-reference)
   - 3.1 [entrance.py — The Entrance Controller](#31-entrancepy--the-entrance-controller)
   - 3.2 [autoreviewer.py — The Auto Reviewer](#32-autoreviewerpy--the-auto-reviewer)
   - 3.3 [core_utils.py — Core Utilities Mixin](#33-core_utilspy--core-utilities-mixin)
   - 3.4 [prompts_and_memory.py — Prompts & Memory Mixin](#34-prompts_and_memorypy--prompts--memory-mixin)
4. [The Verification & Healing Pipeline](#4-the-verification--healing-pipeline)
5. [Symbolic Dependency Management](#5-symbolic-dependency-management)
6. [The XML Edit Engine](#6-the-xml-edit-engine)
7. [Prompt Template System](#7-prompt-template-system)
8. [Human-in-the-Loop Bridging](#8-human-in-the-loop-bridging)
9. [LLM Backend & Resilience](#9-llm-backend--resilience)
10. [Persistence & State Management](#10-persistence--state-management)
11. [Safety & Rollback Mechanisms](#11-safety--rollback-mechanisms)
12. [Configuration Reference](#12-configuration-reference)
13. [Internal Constants & Defaults](#13-internal-constants--defaults)
14. [Operational Workflow](#14-operational-workflow)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. System Philosophy: Constrained Surgical Autonomy

NoClaw is built on the principle of **constrained agency**. Rather than giving an AI free reign to rewrite files, NoClaw forces every modification through:

1. **Surgical XML blocks** — Small, verifiable `<SEARCH>/<REPLACE>` patches instead of full file rewrites
2. **Symbolic verification** — A persistent dependency ledger that tracks the global impact of every change
3. **Multi-layer healing** — Four independent verification layers that catch errors at different levels (syntax, type, runtime)
4. **Human checkpoints** — Interactive approval gates at every critical decision point

This design eliminates the **"hallucination-deletion" spiral** common in autonomous coding agents, where an AI hallucinates a bug, deletes working code to "fix" it, then cascades errors throughout the project.

### Design Principles

| Principle | Implementation |
|---|---|
| **Never leave broken state** | Atomic workspace backup/restore before every modification |
| **Verify, don't trust** | Every AI output is validated before disk write |
| **Surgical over wholesale** | `<SEARCH>` blocks must be 2-5 lines; no full-file rewrites |
| **Context over repetition** | PIR protocol feeds the *original goal* back on failure |
| **Human sovereignty** | Every change requires explicit or timeout-based approval |

---

## 2. Architecture Overview

### Class Hierarchy

```
CoreUtilsMixin (core_utils.py)
├── Provides: LLM streaming, XML edit engine, key rotation,
│             user approval, workspace backup/restore,
│             entry file detection, import preservation
│
PromptsAndMemoryMixin (prompts_and_memory.py)
├── Provides: Prompt template management, memory CRUD,
│             rich context building, history extraction
│
AutoReviewer(CoreUtilsMixin, PromptsAndMemoryMixin) (autoreviewer.py)
├── Provides: 6-phase review pipeline, file analysis,
│             feature proposal/implementation, PR generation,
│             linter fix loops, runtime verification,
│             downstream cascade checks
│
├── TargetedReviewer(AutoReviewer) (entrance.py)
│   └── Overrides scan_directory() to target a single file
│
└── EntranceController (entrance.py)
    ├── Owns: AutoReviewer instance (self.llm_engine)
    ├── Provides: Master loop, symbolic targeting, ripple detection,
    │             analysis/ledger management, structure parsing,
    │             final verification & healing
    └── Entry Point: __main__ → run_master_loop()
```

### Data Flow

```
User runs: python entrance.py /path/to/project
                    │
                    ▼
        ┌──────────────────────┐
        │  EntranceController  │
        │  __init__()          │
        │  • Sets target_dir   │
        │  • Creates AutoReviewer │
        │  • Loads SYMBOLS.json│
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  run_master_loop()   │◄─────────────────────────────┐
        │  1. Bootstrap if     │                              │
        │     ANALYSIS.md      │                              │
        │     missing          │                              │
        │  2. Call execute_    │                              │
        │     targeted_        │                              │
        │     iteration()      │                              │
        └──────────┬───────────┘                              │
                   │                                          │
                   ▼                                          │
        ┌──────────────────────┐                              │
        │  execute_targeted_   │                              │
        │  iteration()         │                              │
        │  1. Backup workspace │                              │
        │  2. Pick target file │                              │
        │  3. Create Targeted  │                              │
        │     Reviewer         │                              │
        │  4. Run pipeline     │                              │
        │  5. Update analysis  │                              │
        │  6. Detect ripples   │                              │
        │  7. Final verify     │                              │
        └──────────┬───────────┘                              │
                   │                                          │
                   ▼                                          │
        ┌──────────────────────┐                              │
        │  AutoReviewer.       │                              │
        │  run_pipeline()      │                              │
        │  Phase 1: Scan/Fix   │                              │
        │  Phase 2: Propose    │                              │
        │  Phase 3: Cascade    │                              │
        │  Phase 4: Runtime    │                              │
        │  Phase 5: Memory     │                              │
        │  Phase 6: Refactor   │                              │
        └──────────┬───────────┘                              │
                   │                                          │
                   ▼                                          │
        ┌──────────────────────┐                              │
        │  120s cooldown       │──────────────────────────────┘
        └──────────────────────┘
```

---

## 3. Module Reference

### 3.1 `entrance.py` — The Entrance Controller

The top-level orchestrator that manages symbolic targeting, dependency tracking, and final runtime verification.

#### Classes

##### `TargetedReviewer(AutoReviewer)`
A scoped subclass of `AutoReviewer` that overrides `scan_directory()` to operate on exactly one file.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(target_dir: str, target_file: str)` | Sets the forced target file |
| `scan_directory` | `() → list[str]` | Returns only `[self.forced_target_file]` if it exists |

##### `EntranceController`
The master controller that owns the main event loop.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(target_dir: str)` | Initializes paths, creates `AutoReviewer`, loads `SYMBOLS.json` |
| `run_master_loop` | `()` | Infinite loop: bootstrap → target → iterate → cooldown (120s) |
| `execute_targeted_iteration` | `(iteration: int)` | Single iteration: backup → pick target → run pipeline → verify → cascade |
| `_run_final_verification_and_heal` | `(backup_state: dict) → bool` | Launches app for 10s; auto-heals up to 3 times; rolls back on failure |
| `detect_symbolic_ripples` | `(old, new, source_file) → list` | Finds files referencing symbols defined in the modified file |
| `pick_target_file` | `() → str` | Uses LLM to intelligently select next file based on `ANALYSIS.md` and `HISTORY.md` |
| `build_initial_analysis` | `()` | Genesis scan: builds `ANALYSIS.md` and `SYMBOLS.json` from scratch |
| `update_analysis_for_single_file` | `(target_abs_path, rel_path)` | Updates one file's section in `ANALYSIS.md` |
| `update_ledger_for_file` | `(rel_path, code)` | Parses definitions (AST for Python, regex for JS/TS) and references |
| `generate_structure_dropdowns` | `(filepath, code) → str` | Generates HTML `<details>` dropdowns for imports, classes, functions, constants |
| `append_to_history` | `(rel_path, old_code, new_code)` | Appends truncated unified diff to `HISTORY.md` |
| `load_ledger` | `() → dict` | Loads `SYMBOLS.json` or returns empty schema |
| `save_ledger` | `()` | Writes `SYMBOLS.json` to disk |

**Internal Parsers:**

| Method | Language | Extracts |
|---|---|---|
| `_parse_python` | Python | Imports, classes, functions (with args), uppercase constants |
| `_parse_javascript` | JS/TS | Imports, classes, functions (3 patterns including arrows), constants/entities |
| `_parse_html` | HTML | Script sources, stylesheet links, element IDs |
| `_parse_css` | CSS | Class selectors (first 50) |

---

### 3.2 `autoreviewer.py` — The Auto Reviewer

The core review and modification engine. Inherits from both `CoreUtilsMixin` and `PromptsAndMemoryMixin`.

#### Class: `AutoReviewer(CoreUtilsMixin, PromptsAndMemoryMixin)`

##### Initialization

| Attribute | Type | Description |
|---|---|---|
| `target_dir` | `str` | Absolute path to the project being reviewed |
| `pr_file` | `str` | Path to `PEER_REVIEW.md` |
| `feature_file` | `str` | Path to `FEATURE.md` |
| `failed_pr_file` | `str` | Path to `FAILED_PEER_REVIEW.md` |
| `failed_feature_file` | `str` | Path to `FAILED_FEATURE.md` |
| `memory_file` | `str` | Path to `MEMORY.md` |
| `analysis_path` | `str` | Path to `ANALYSIS.md` |
| `history_path` | `str` | Path to `HISTORY.md` |
| `symbols_path` | `str` | Path to `SYMBOLS.json` |
| `memory` | `str` | Loaded content of `MEMORY.md` |
| `session_context` | `list[str]` | Running log of actions in the current session |
| `key_cooldowns` | `dict` | Maps API keys to their cooldown expiry timestamps |

##### Methods

| Method | Signature | Description |
|---|---|---|
| `get_language_info` | `(filepath) → tuple[str, str]` | Returns `(language_name, language_tag)` for syntax highlighting |
| `scan_for_lazy_code` | `(filepath, content) → list[str]` | AST walker that flags `Any` type hints |
| `run_linters` | `(filepath) → tuple[str, str]` | Runs `ruff check` and `mypy` on a single file |
| `build_patch_prompt` | `(lang_name, lang_tag, content, ruff_out, mypy_out, custom_issues) → str` | Assembles the `PP.md` prompt with all context |
| `get_valid_edit` | `(prompt, source_code, require_edit, target_filepath) → tuple[str, str, str]` | **Core edit loop**: streams LLM → validates XML → shows diff → gets approval |
| `run_linter_fix_loop` | `(context_of_change) → bool` | Runs ruff/node/CSS checks; auto-fixes up to 3 times per language |
| `run_and_verify_app` | `(context_of_change) → bool` | Launches entry file for 10s; auto-fixes crashes up to 3 times |
| `analyze_file` | `(filepath, current_index, total_files)` | Phase 1 per-file analysis: lint → scan → patch prompt → AI review |
| `scan_directory` | `() → list[str]` | Walks `target_dir` finding supported files, skipping ignored paths |
| `propose_feature` | `(target_path)` | Phase 2: generates a feature proposal with `<SNIPPET>` block |
| `implement_feature` | `(feature_content) → bool` | Applies an approved feature from `FEATURE.md` into the source |
| `implement_pr` | `(pr_content) → bool` | Applies all approved patches from `PEER_REVIEW.md` |
| `check_downstream_breakages` | `(target_path, rel_path) → bool` | Phase 3: runs workspace-wide `mypy` to detect cascading errors |
| `propose_cascade_fix` | `(mypy_errors, trigger_file) → bool` | Generates and applies a fix for downstream type errors |
| `write_pr` | `(filepath, explanation, llm_response)` | Appends a patch proposal to `PEER_REVIEW.md` |
| `run_pipeline` | `(current_iteration)` | **Master pipeline**: Phase 1–6 with approval checkpoints |

##### `get_valid_edit()` — The Core Edit Loop

This is the most complex method in NoClaw. It handles:

1. **Pre-LLM Checkpoint**: User can `EDIT_PROMPT`, `AUGMENT_PROMPT`, or `SKIP`
2. **Key Rotation**: Cycles through available Gemini keys; falls back to Ollama
3. **429 Handling**: Rate-limited keys get 20-minute quarantine
4. **XML Validation**: Calls `apply_xml_edits()` and rejects partial failures
5. **Diff Display**: Shows colorized unified diff (green=added, red=removed, blue=hunks)
6. **Post-LLM Checkpoint**: User can `APPLY`, `FULL_DIFF`, `EDIT_CODE`, `EDIT_XML`, `REGENERATE`, or `SKIP`

```
get_valid_edit() Flow:
┌─────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
│ Pre-LLM     │──▶│ Stream   │──▶│ Validate  │──▶│ Show     │
│ Checkpoint  │   │ LLM      │   │ XML Edits │   │ Diff     │
│ (User)      │   │ Response │   │ (5-layer) │   │ (color)  │
└─────────────┘   └──────────┘   └───────────┘   └────┬─────┘
                       ▲              │ Fail           │
                       └──────────────┘                ▼
                                              ┌──────────────┐
                                              │ Post-LLM     │
                                              │ Checkpoint    │
                                              │ (User)        │
                                              └──────────────┘
```

---

### 3.3 `core_utils.py` — Core Utilities Mixin

Provides foundational infrastructure shared across all components.

#### Constants

| Constant | Value | Description |
|---|---|---|
| `GEMINI_API_KEYS` | `list[str]` | Pool of Gemini API keys for rotation |
| `GEMINI_MODEL` | `"gemini-2.5-flash"` | Primary cloud LLM model |
| `LOCAL_MODEL` | `"qwen3-coder:30b"` | Fallback local Ollama model |
| `PR_FILE_NAME` | `"PEER_REVIEW.md"` | Bug fix proposal filename |
| `FEATURE_FILE_NAME` | `"FEATURE.md"` | Feature proposal filename |
| `FAILED_PR_FILE_NAME` | `"FAILED_PEER_REVIEW.md"` | Rolled-back PR filename |
| `FAILED_FEATURE_FILE_NAME` | `"FAILED_FEATURE.md"` | Rolled-back feature filename |
| `MEMORY_FILE_NAME` | `"MEMORY.md"` | Persistent memory filename |
| `ANALYSIS_FILE` | `"ANALYSIS.md"` | Project analysis filename |
| `HISTORY_FILE` | `"HISTORY.md"` | Change history filename |
| `SYMBOLS_FILE` | `"SYMBOLS.json"` | Dependency graph filename |
| `IGNORE_DIRS` | `set` | Directories excluded from scanning |
| `IGNORE_FILES` | `set` | Files excluded from scanning (includes NoClaw's own source files) |
| `SUPPORTED_EXTENSIONS` | `set` | `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.sh` |

#### Class: `CoreUtilsMixin`

| Method | Signature | Description |
|---|---|---|
| `get_user_approval` | `(prompt_text, timeout=220) → str` | Non-blocking terminal input with countdown timer; supports Windows (`msvcrt`) and Unix (`tty`/`termios`/`select`) |
| `_launch_external_code_editor` | `(initial_content, file_suffix=".py") → str` | Opens proposed code in `$EDITOR` (default: `nano`) for manual refinement |
| `_edit_prompt_with_external_editor` | `(initial_prompt) → str` | Opens a prompt in `$EDITOR` for manual editing |
| `_get_user_prompt_augmentation` | `(initial_text="") → str` | Opens a temp `.txt` file for quick instruction injection |
| `backup_workspace` | `() → dict` | Snapshots all supported files into an in-memory dictionary |
| `restore_workspace` | `(state: dict)` | Writes all files in the snapshot back to disk |
| `load_memory` | `() → str` | Reads `MEMORY.md` content or returns empty string |
| `stream_gemini` | `(prompt, api_key, on_chunk) → str` | Streams Gemini API response via SSE; returns `ERROR_CODE_XXX` on failure |
| `stream_ollama` | `(prompt, on_chunk) → str` | Streams Ollama local model response |
| `_stream_single_llm` | `(prompt, key=None, context="") → str` | Unified LLM streamer with animated progress spinner |
| `get_valid_llm_response` | `(prompt, validator, context="") → str` | Loops LLM calls until `validator(response)` returns `True` |
| `ensure_imports_retained` | `(orig_code, new_code, filepath) → str` | AST-based comparison that prepends any imports dropped during editing |
| `apply_xml_edits` | `(source_code, llm_response) → tuple[str, str, bool]` | **5-strategy XML edit engine** (see Section 6) |
| `_find_entry_file` | `() → str \| None` | Searches for `if __name__ == "__main__":`, then `main.py`/`app.py` |

---

### 3.4 `prompts_and_memory.py` — Prompts & Memory Mixin

Manages the prompt template lifecycle and persistent memory.

#### Class: `PromptsAndMemoryMixin`

| Method | Signature | Description |
|---|---|---|
| `_ensure_prompt_files` | `()` | Writes all 8 prompt templates to the target directory on every initialization |
| `load_prompt` | `(filename, **kwargs) → str` | Loads a template and performs `{key}` → `value` substitution |
| `_get_impactful_history` | `() → str` | Extracts the 3 most recent `HISTORY.md` entries as a summary |
| `_get_rich_context` | `() → str` | Builds a comprehensive context block from `ANALYSIS.md` header + recent history + memory |
| `update_memory` | `()` | Synthesizes session actions into `MEMORY.md` via the `UM.md` template |
| `refactor_memory` | `()` | Aggressively summarizes `MEMORY.md` via the `RM.md` template to prevent bloat |

---

## 4. The Verification & Healing Pipeline

This is the most critical logic path in NoClaw, ensuring codebase integrity through four distinct layers.

### Layer 1: Atomic XML Matching

Edits are atomic. If the AI proposes five `<EDIT>` blocks and the system fails to find the exact `<SEARCH>` anchor for the fifth one, the **entire multi-block patch is rejected**. The system then triggers an automatic regeneration attempt rather than applying a partial (broken) fix.

**Key behavior:**
- The `apply_xml_edits()` method returns a boolean `all_edits_succeeded`
- If `False`, `get_valid_edit()` increments the `attempts` counter and loops
- No partial edits are written to disk

### Layer 2: Syntactic Validation (Linter Loop)

Immediately after file modification via `run_linter_fix_loop()`:

| Language | Validator | Error Handling |
|---|---|---|
| **Python** | `ruff format` → `ruff check` | Groups errors by file; AI auto-fixes up to 3 times per file |
| **JavaScript** | `node --check` | Per-file validation; AI auto-fixes up to 3 times |
| **CSS** | Brace counting (`{` vs `}`) | Reports unbalanced braces; no AI auto-fix |

### Layer 3: Context-Aware Self-Healing (PIR)

If Layer 2 or Layer 4 detects an error, NoClaw initiates a **Post-Implementation Repair (PIR)**.

| Fixer Type | Context Provided |
|---|---|
| **Standard Fixer** (`ALF.md`, `FRE.md`) | Error text + broken code |
| **PIR Fixer** (`PIR.md`) | Original feature request + error text + broken code |

The PIR advantage: When the AI knows *what it was trying to do* (e.g., "I duplicated a function while trying to add timezone support"), it can make a logically correct repair instead of a blind syntax fix.

### Layer 4: Runtime Verification

Controlled by both `autoreviewer.py` (`run_and_verify_app`) and `entrance.py` (`_run_final_verification_and_heal`):

1. Identifies the project's entry point (searches for `if __name__ == "__main__":`, then `main.py`/`app.py`)
2. Launches the app with `subprocess.Popen` and monitors for 10 seconds
3. Checks `stderr` for crash keywords: `Traceback`, `Exception:`, `Error:`, `NameError:`, `AttributeError:`
4. On `ModuleNotFoundError`: auto-installs the missing package via `pip install`
5. On crash: feeds the traceback to `_fix_runtime_errors()` which identifies the most likely culprit file from the traceback path
6. Retries up to 3 times before performing a full workspace rollback

**Return codes considered non-crash:** `None`, `0`, `15`, `-15`, `137`, `-9`, `1` (process signals)

---

## 5. Symbolic Dependency Management

NoClaw tracks the "Global Impact" of code changes via `SYMBOLS.json`.

### Schema

```json
{
  "definitions": {
    "MyClass": "models/user.py",
    "calculate_total": "utils/math.py",
    "initApp": "static/app.js"
  },
  "references": {
    "main.py": ["MyClass", "calculate_total", "initApp"],
    "views/dashboard.py": ["MyClass", "calculate_total"],
    "static/app.js": ["initApp"]
  }
}
```

### Ledger Generation

During the **Genesis Scan** (`build_initial_analysis()`), the controller parses every file:

| Language | Definition Extraction | Reference Extraction |
|---|---|---|
| **Python** | AST: `FunctionDef`, `ClassDef` names | Regex: `[a-zA-Z0-9_$]{4,}` followed by `(` or `.` |
| **JS/TS** | Regex: `function`, `class`, `const`/`var`/`let` declarations | Same regex pattern as Python |

### Symbolic Ripple Detection

When a file containing a **definition** is edited, `detect_symbolic_ripples()`:

1. Computes the unified diff between old and new content
2. Extracts all identifiers (4+ chars) from added/removed lines
3. Checks if any extracted identifier is a **definition** owned by the source file
4. Finds all other files that **reference** those identifiers
5. Adds impacted files to the `cascade_queue` for automatic review in subsequent iterations

### Cascade Queue

The `cascade_queue` is a FIFO list maintained by `EntranceController`:
- When a ripple is detected, impacted files are appended (deduplicated)
- Each cascade target also receives the triggering diff as `cascade_diffs[rel_path]`
- On the next iteration, cascade files take priority over LLM-selected targets
- The cascade reviewer's `session_context` includes the dependency change diff

---

## 6. The XML Edit Engine

### Edit Format

```xml
<THOUGHT>
Explanation of what this edit does and why...
</THOUGHT>
<EDIT>
<SEARCH>
exact lines to find in source
</SEARCH>
<REPLACE>
new replacement lines
</REPLACE>
</EDIT>
```

### 5-Strategy Matching Pipeline

The `apply_xml_edits()` method in `core_utils.py` attempts to match each `<SEARCH>` block using progressively fuzzier strategies:

#### Strategy 1: Exact String Match
```python
if raw_search in new_code:
    new_code = new_code.replace(raw_search, raw_replace, 1)
```
Direct substring replacement. Fastest and most reliable.

#### Strategy 2: Stripped Match
```python
clean_search = raw_search.strip("\n")
if clean_search in new_code:
    new_code = new_code.replace(clean_search, clean_replace, 1)
```
Tolerates leading/trailing newlines added by the LLM.

#### Strategy 3: Normalized Match
```python
def normalize(t):
    t = re.sub(r"#.*", "", t)          # Strip comments
    return re.sub(r"\s+", " ", t).strip()  # Collapse whitespace

# Slides a window over source lines looking for normalized match
```
Ignores comments and whitespace differences. Matches by normalizing both search and source into single-space strings.

#### Strategy 4: Regex Fuzzy Match
```python
# Builds a regex from each search line:
# ^[ \t]*{escaped_line}[ \t]*\n+
# Allows flexible indentation
```
Constructs a multiline regex that tolerates indentation differences between the AI's output and the actual source.

#### Strategy 5: Robust Line-by-Line Match
```python
# Strips each line and checks if search_line is contained in code_line
for i in range(len(code_lines) - len(search_lines) + 1):
    match = all(sline in code_lines[i+j].strip() for j, sline in enumerate(search_lines_stripped))
```
Most forgiving strategy — checks if each stripped search line appears as a substring within the corresponding source line.

### Smart Indent Alignment

Before replacement, the engine:

1. Detects the base indentation of the `<SEARCH>` block (first non-empty line)
2. Detects the base indentation of the `<REPLACE>` block (first non-empty line)
3. Strips the replace block's base indent
4. Prepends the search block's base indent to every non-empty line

This prevents indentation corruption when the AI outputs code at a different indentation level than the source.

### Atomic Failure Mode

If **any** of the `<EDIT>` blocks fails all 5 strategies:
- `all_edits_succeeded` is set to `False`
- `get_valid_edit()` detects this and increments the attempt counter
- The AI is asked to regenerate the entire response
- **No partial edits are written to disk**

---

## 7. Prompt Template System

### Template Architecture

All 8 templates are defined as Python strings in `prompts_and_memory.py` → `_ensure_prompt_files()` and written to the target directory as `.md` files on every initialization. This ensures templates are always fresh and match the current NoClaw version.

### Template Variable Substitution

Templates use `{variable_name}` placeholders. The `load_prompt()` method performs simple string replacement:

```python
for key, value in kwargs.items():
    template = template.replace(f"{{{key}}}", str(value))
```

### Template Reference

#### `PP.md` — Patch Prompt (Code Review)
**Variables:** `memory_section`, `ruff_section`, `mypy_section`, `custom_issues_section`, `lang_tag`, `content`
**Purpose:** Analyzes code for bugs, syntax errors, and architectural gaps. Strict rules: 2-5 line `<SEARCH>` blocks, no hallucinated bugs, no new features.

#### `PF.md` — Propose Feature
**Variables:** `memory_section`, `lang_tag`, `content`, `rel_path`
**Purpose:** Suggests one interactive feature. Must output `<THOUGHT>` + `<SNIPPET>` blocks. Checks for orphaned logic that needs UI connections.

#### `IF.md` — Implement Feature
**Variables:** `memory_section`, `feature_content`, `lang_name`, `lang_tag`, `source_code`, `rel_path`
**Purpose:** Surgically implements an approved feature. Respects function signatures from `ANALYSIS.md`. Uses multiple `<EDIT>` blocks (imports, `__init__`, logic).

#### `ALF.md` — Auto Linter Fix
**Variables:** `rel_path`, `err_text`, `code`
**Purpose:** Fixes syntax errors from linter validation. Minimal context — just the error and the code.

#### `FRE.md` — Fix Runtime Error
**Variables:** `memory_section`, `logs`, `rel_path`, `code`
**Purpose:** Diagnoses and fixes runtime crashes from traceback logs.

#### `PIR.md` — Post-Implementation Repair
**Variables:** `context_of_change`, `err_text`, `rel_path`, `code`
**Purpose:** Context-aware error recovery. Receives the *original goal* that caused the breakage, enabling intelligent repair.

#### `PCF.md` — Propose Cascade Fix
**Variables:** `memory_section`, `trigger_file`, `rel_broken_path`, `mypy_errors`, `broken_code`
**Purpose:** Fixes downstream type errors caused by changes in a dependency file.

#### `UM.md` — Update Memory
**Variables:** `current_memory`, `session_summary`
**Purpose:** Synthesizes session actions into `MEMORY.md`. Merges rather than appends.

#### `RM.md` — Refactor Memory
**Variables:** `current_memory`
**Purpose:** Aggressively summarizes bloated `MEMORY.md`. Consolidates repeated entries.

---

## 8. Human-in-the-Loop Bridging

NoClaw allows for "Supervised Autonomy" through interactive terminal checkpoints.

### Pre-LLM Checkpoints

Before any LLM call in `get_valid_edit()`:

| Command | Action |
|---|---|
| `ENTER` (empty) | Send prompt as-is |
| `EDIT_PROMPT` | Opens full prompt in `$EDITOR` for manual refinement |
| `AUGMENT_PROMPT` | Opens a blank file to add quick instructions (appended to prompt) |
| `SKIP` | Cancel the operation entirely |

### Post-LLM Checkpoints

After the AI generates proposed changes:

| Command | Action |
|---|---|
| `ENTER` (empty) | Apply the proposed change to disk |
| `FULL_DIFF` | View the complete unified diff in a pager (`$PAGER` or `less -R`) |
| `EDIT_CODE` | Open the proposed code in `$EDITOR`; save to apply your refinements |
| `EDIT_XML` | Open the raw AI XML response in `$EDITOR`; re-parse after editing |
| `REGENERATE` | Reject the proposal; increment attempts and ask AI again |
| `SKIP` | Cancel and keep original code |

### Timeout Behavior

All checkpoints have a configurable timeout (default: **220 seconds**). If the timeout expires without user input, the system defaults to `"PROCEED"` — auto-applying the change to maintain autonomous operation during unattended sessions.

### Terminal Input Implementation

The `get_user_approval()` method provides a real-time countdown display:

```
⏳ 185s remaining | You: FULL_DIFF
```

- **Unix**: Uses `tty.setcbreak()` + `select.select()` for non-blocking character-by-character input with 100ms polling
- **Windows**: Uses `msvcrt.kbhit()` + `msvcrt.getwch()` for the same behavior

---

## 9. LLM Backend & Resilience

### Gemini Streaming

- **Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse`
- **Protocol**: Server-Sent Events (SSE)
- **Temperature**: `0.1` (near-deterministic)
- **Timeout**: 220 seconds
- **Real-time output**: Chunks are printed to stdout as they arrive

### Ollama Streaming

- **Library**: `ollama` Python package
- **Model**: `qwen3-coder:30b`
- **Context window**: 32,000 tokens
- **Temperature**: `0.1`
- **Fallback condition**: All Gemini API keys exhausted or rate-limited

### Key Rotation Strategy

```
Available Keys Pool:
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ K1  │  │ K2  │  │ K3  │  │ K4  │  │ K5  │
│ ✅  │  │ ⏳  │  │ ✅  │  │ ✅  │  │ ⏳  │
└─────┘  └─────┘  └─────┘  └─────┘  └─────┘
  ↑                  ↑        ↑
  Available          Available Available

Selection: key = available_keys[attempts % len(available_keys)]
```

1. On each attempt, select from the pool of non-cooled-down keys using modular rotation
2. On `HTTP 429`: quarantine the key for **20 minutes** (`key_cooldowns[key] = time.time() + 1200`)
3. If **all keys** are quarantined: seamlessly switch to local Ollama
4. Keys are automatically reinstated when their cooldown expires

### Progress Spinner

During LLM inference, a background thread displays:

```
⠹ Reading [game.py] ~1250 ctx... [████████████░░░░░░░░░░░░░] 48.0% (5.2s)
```

- Estimates progress based on `input_tokens / 12.0` seconds expected
- Transitions to `"100% - AI Inference..."` when estimate is exceeded
- Clears line and shows `"🤖 AI Output (Gemini ...abc1):"` when first chunk arrives

---

## 10. Persistence & State Management

### `ANALYSIS.md` — Project Map

Generated during the genesis scan and updated after every file modification.

**Structure:**
```markdown
# 🧠 Project Analysis

**Project Summary:**
[AI-generated 2-sentence project description]

---

## 📂 File Directory

### `models/user.py`
**Summary:** [AI-generated one-sentence description]

<details><summary>Imports (5)</summary>...</details>
<details><summary>Classes/Structures (2)</summary>...</details>
<details><summary>Logic/Functions (8)</summary>...</details>
<details><summary>Entities/Constants (3)</summary>...</details>

---
```

**Purpose:** Allows the AI to "see" the entire project architecture without reading every file into the context window. Used by `pick_target_file()` to make intelligent targeting decisions.

### `MEMORY.md` — Persistent Session Memory

Updated at the end of every pipeline iteration (Phase 5) and aggressively refactored every 2nd iteration (Phase 6).

**Key behaviors:**
- The `UM.md` template instructs the AI to **merge** recent actions into the existing memory rather than appending using `mem_str` to set a memory cap @ `if len(mem_str) > 1500:`
- The `RM.md` template consolidates repeated entries and removes redundant logs
- Memory content is injected into prompts via `_get_rich_context()` as `### Logic Memory:`
- Maximum memory size is kept manageable through periodic refactoring

### `HISTORY.md` — Change Ledger

Append-only log of every unified diff applied to the project.

**Structure:**
```markdown
## 2026-03-04 12:30:45 - `game.py`
```diff
--- Original
+++ Proposed
@@ -10,3 +10,5 @@
...
```
---
```

**Truncation:** Diffs longer than 20 lines are truncated to first 5 + last 5 lines with a `[TRUNCATED FOR MEMORY]` marker.

### `SYMBOLS.json` — Dependency Graph

See [Section 5: Symbolic Dependency Management](#5-symbolic-dependency-management).

---

## 11. Safety & Rollback Mechanisms

### Workspace Backup/Restore

Before every modification attempt, `backup_workspace()` creates an in-memory snapshot:

```python
state = {}
for root, dirs, files in os.walk(self.target_dir):
    # Skips IGNORE_DIRS
    for file in files:
        if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            state[path] = file_content
```

If verification fails, `restore_workspace(state)` writes all files back to their backed-up content.

### Import Preservation

The `ensure_imports_retained()` method prevents the AI from accidentally dropping imports:

1. Parses both original and new code with `ast.parse()`
2. Extracts all `Import` and `ImportFrom` nodes from the original
3. Checks if each original import exists in the new code
4. Prepends any missing imports to the new code

This runs automatically during `implement_feature()` for Python files.

### Failed Proposal Archives

When a PR or feature implementation fails and the workspace is rolled back:
- `PEER_REVIEW.md` → renamed to `FAILED_PEER_REVIEW.md`
- `FEATURE.md` → renamed to `FAILED_FEATURE.md`

This preserves the failed proposal for debugging while clearing the active queue.

### Cascade Queue Deduplication

Files are only added to the cascade queue if they're not already present:
```python
if r not in self.cascade_queue:
    self.cascade_queue.append(r)
```

---

## 12. Configuration Reference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EDITOR` | `nano` | Terminal editor for prompt/code editing |
| `PAGER` | `less -R` | Pager for viewing full diffs |

### Modifiable Constants (in `core_utils.py`)

| Constant | Default | Description |
|---|---|---|
| `GEMINI_API_KEYS` | 5 keys | API key pool for rotation |
| `GEMINI_MODEL` | `"gemini-2.5-flash"` | Gemini model identifier |
| `LOCAL_MODEL` | `"qwen3-coder:30b"` | Ollama model identifier |
| `IGNORE_DIRS` | 12 directories | Directories excluded from scanning |
| `IGNORE_FILES` | 14 files | Files excluded from scanning |
| `SUPPORTED_EXTENSIONS` | 7 extensions | File types NoClaw can review |

### Timing Constants

| Constant | Value | Location |
|---|---|---|
| User approval timeout | 220 seconds | `get_user_approval()` |
| Key quarantine duration | 1200 seconds (20 min) | `get_valid_edit()` |
| API request timeout | 220 seconds | `stream_gemini()` |
| Master loop cooldown | 120 seconds | `run_master_loop()` |
| Runtime test duration | 10 seconds | `run_and_verify_app()` |
| Runtime process kill grace | 2 seconds | `run_and_verify_app()` |
| Memory refactor interval | Every 2 iterations | `run_pipeline()` |

---

## 13. Internal Constants & Defaults

### File Name Constants

| Constant | Value |
|---|---|
| `PR_FILE_NAME` | `"PEER_REVIEW.md"` |
| `FEATURE_FILE_NAME` | `"FEATURE.md"` |
| `FAILED_PR_FILE_NAME` | `"FAILED_PEER_REVIEW.md"` |
| `FAILED_FEATURE_FILE_NAME` | `"FAILED_FEATURE.md"` |
| `MEMORY_FILE_NAME` | `"MEMORY.md"` |
| `ANALYSIS_FILE` | `"ANALYSIS.md"` |
| `HISTORY_FILE` | `"HISTORY.md"` |
| `SYMBOLS_FILE` | `"SYMBOLS.json"` |

### Language Mapping

| Extension | Language Name | Tag |
|---|---|---|
| `.py` | Python | `python` |
| `.js` | JavaScript | `javascript` |
| `.ts` | TypeScript | `typescript` |
| `.html` | HTML | `html` |
| `.css` | CSS | `css` |
| `.json` | JSON | `json` |
| `.sh` | Bash | `bash` |
| `.md` | Markdown | `markdown` |

### Lazy Code Detection (AST)

The `scan_for_lazy_code()` method flags:
- `ast.Name` nodes where `node.id == "Any"` — bare `Any` type hint usage
- `ast.Attribute` nodes where `node.attr == "Any"` — `typing.Any` usage

---

## 14. Operational Workflow

### First Run (Genesis)

```
1. EntranceController.__init__()
   └── Creates AutoReviewer, loads empty ledger

2. run_master_loop()
   └── Checks for ANALYSIS.md → Not found

3. build_initial_analysis()
   ├── Scans all supported files
   ├── For each file:
   │   ├── Parses structure (AST/regex) → generates dropdowns
   │   ├── Updates SYMBOLS.json with definitions and references
   │   └── Asks LLM for one-sentence summary
   ├── Writes ANALYSIS.md
   └── Saves SYMBOLS.json

4. execute_targeted_iteration(1)
   ├── Backup workspace
   ├── pick_target_file() → LLM selects from ANALYSIS.md
   ├── Create TargetedReviewer for selected file
   ├── run_pipeline(1)
   │   ├── Phase 1: analyze_file() → scan, lint, review
   │   ├── Phase 2: propose_feature() → if no bugs found
   │   ├── User checkpoint → APPLY / SKIP
   │   ├── implement_pr() or implement_feature()
   │   ├── Phase 3: check_downstream_breakages()
   │   ├── Phase 4: run_and_verify_app()
   │   └── Phase 5: update_memory()
   ├── Update ANALYSIS.md for modified file
   ├── Update SYMBOLS.json
   ├── Detect symbolic ripples → queue cascades
   └── Final verification with healing

5. 120-second cooldown → loop back to step 4
```

### Subsequent Runs

If `ANALYSIS.md` already exists, step 3 is skipped. The system resumes the targeted iteration loop immediately.

### Cascade Flow

```
Iteration N: Modified function `calculate()` in `math.py`
  └── Ripple detected: `main.py` references `calculate`
      └── Added to cascade_queue

Iteration N+1: cascade_queue is not empty
  └── Pops `main.py` from queue
  └── Session context includes: "CRITICAL SYMBOLIC RIPPLE: ..."
  └── TargetedReviewer scans `main.py` with cascade context
```

---

## 15. Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---|---|---|
| `Warning: 'ollama' package not found` | Ollama Python package not installed | `pip install ollama` |
| All keys rate-limited, no Ollama | Both backends unavailable | Install Ollama and pull `qwen3-coder:30b` |
| `ruff` / `mypy` not found | Linting tools not installed | `pip install ruff mypy` (NoClaw will skip these checks gracefully) |
| `Node.js not installed` | JS validation unavailable | Install Node.js (NoClaw will skip JS checks) |
| Edits keep failing to match | AI generating incorrect `<SEARCH>` blocks | System auto-retries; if persistent, use `EDIT_XML` to fix manually |
| App crashes during runtime test | Feature implementation introduced a bug | System auto-heals up to 3 times; then rolls back |
| Memory growing too large | Many iterations without refactoring | Memory auto-refactors every 2 iterations; can manually delete `MEMORY.md` |
| `FAILED_PEER_REVIEW.md` appears | PR implementation failed and was rolled back | Review the failed file; issues will be re-detected on next scan |

### Logging

NoClaw uses Python's built-in `logging` module at the `INFO` level:

```
2026-03-04 12:30:45,123 | [1/5] Scanning game.py (Python) - Reading 245 lines into AI context...
```

All output includes timestamps for debugging timing-related issues.

---

> **NoClaw** — Surgical precision, never destructive. 🦅

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=vicsanity623/NoClaw&type=date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=vicsanity623/NoClaw&type=date" />
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=vicsanity623/NoClaw&type=date" />
</picture>
