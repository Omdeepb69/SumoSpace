# SumoSpace Autonomous Intelligence Training & Small-Model Optimization Master Plan

We are now shifting SumoSpace from being only an autonomous agent framework into becoming a fully specialized autonomous intelligence ecosystem optimized for local and mid-sized models.

The mission is to create a deeply SumoSpace-native model stack capable of:

* autonomous reasoning
* tool usage
* planning
* execution
* debugging
* recovery
* committee collaboration
* environment awareness
* long-horizon task completion

while running efficiently on:

* consumer GPUs
* local workstations
* laptops
* edge devices
* eventually mobile hardware

This is NOT generic instruction tuning.

We are designing a specialized autonomous systems model trained specifically to understand:

* SumoSpace architecture
* SumoSpace workflows
* SumoSpace execution philosophy
* dynamic tool systems
* real autonomous execution traces
* recovery and correction behavior
* adaptive reasoning under changing environments

The system must remain robust even as the framework evolves over time.

---

# PRIMARY OBJECTIVE

Create a production-grade autonomous model ecosystem for SumoSpace that:

* performs reliably on small models
* minimizes hallucinations
* survives framework evolution
* adapts to changing toolsets
* handles custom user tools
* reasons over environments
* executes long-running workflows
* recovers from failures autonomously

---

# PHASE 1 — FOUNDATION ARCHITECTURE

## Goal

Build the core training and runtime architecture before any large-scale dataset generation begins.

---

## 1. MODEL STRATEGY

**CRITICAL REVISION**: While the long-term vision requires specialized models (Planner, Executor, Critic), **we will start by training ONE single generalist model** (e.g., Qwen2.5-Coder 7B). 

Training a single model first is essential to prove the dataset, architecture, and training pipeline work before introducing the massive complexity of multi-model distillation.

### A. The Foundation Generalist

Responsibilities:
* tool use
* planning
* execution
* repair
* environment-aware decision making

Best Choice:
**Alibaba Cloud Qwen2.5-Coder 7B**

Reason:
* strongest tool-use priors
* strong XML handling
* excellent code capabilities
* efficient quantization
* MUCH easier to iterate on than the 14B variant

*(Note: Specialized Planner, Executor, and Critic models will be distilled from this generalist in later phases.)*

---

# PHASE 2 — DATASET ENGINEERING (MOST IMPORTANT)

This is the actual moat.

The dataset determines whether SumoSpace becomes:

* another wrapper
  OR
* a real autonomous execution intelligence system.

---

# CORE DATASET PRINCIPLES

The dataset MUST teach:

* reasoning
* adaptation
* execution philosophy
* environmental awareness
* tool semantics
* recovery behavior
* generalized autonomy

NOT memorization.

The model must understand:

* why a tool exists
* when to use it
* when NOT to use it
* how to adapt if it changes

---

# CRITICAL REQUIREMENT — FRAMEWORK EVOLUTION RESILIENCE

The dataset MUST intentionally simulate framework evolution.

Future SumoSpace versions may:

* rename tools
* change parameter names
* modify schemas
* alter prompts
* change workflows
* add/remove agents
* introduce new environments
* modify execution order

The model must NOT break because:
write_file became save_file.

The model should infer:
“This tool writes files”
from:

* descriptions
* environment metadata
* examples
* execution traces

The model must learn semantics instead of memorizing token patterns.

---

# CRITICAL COMPONENT — STATE REPRESENTATION

Autonomous intelligence heavily depends on the CURRENT WORLD STATE. Every trace sample in the dataset MUST include a structured environment state header:

```json
{
  "environment": {
    "os": "linux",
    "internet": false,
    "gpu": true,
    "cwd": "/workspace/project",
    "available_tools": ["read_file", "write_file", "shell"],
    "token_budget_remaining": 12000,
    "previous_failures": ["shell_timeout"],
    "execution_mode": "sandboxed"
  }
}
```
Without this state injection, models cannot learn adaptive reasoning.

---

# DATASET TYPES

## 1. Multi-Turn Autonomous Traces

Traces must be MULTI-TURN. Single-turn traces are not enough. The model must learn that "the world changes after actions" through:
* state progression
* memory updates
* failure accumulation
* adaptive retries

---

## 2. Failed Traces & NEGATIVE REWARD STATES

VERY IMPORTANT. Real autonomous systems fail constantly. We must train on malformed calls, wrong assumptions, and loop recovery.

Crucially, failures must include a `<reflection>` block explaining *why* it failed, teaching the model to infer corrections:

**GOOD TRACE EXAMPLE:**
```xml
<tool>write_file</tool>
<error>missing parameter</error>
<reflection>
The previous tool call failed because the content parameter was omitted.
The write_file tool requires:
- path
- content

The next attempt must include the full file body.
</reflection>
```
This is MASSIVE for small-model robustness.

---

## 3. Tool Generalization Data

The dataset MUST include:

* built-in tools
* dynamically injected tools
* user-created custom tools
* partially documented tools
* tools with changed schemas
* renamed tools
* incomplete descriptions

The model should learn:
“How to infer tool behavior from context.”

---

# EXAMPLE: XML DSL

For autonomous systems, JSON is horrible. The dataset will use an XML-based Domain Specific Language (DSL). This is easier for small models, easier to tokenize, stream, and repair:

```xml
<thought>
Need to inspect the repository structure first.
</thought>
<call tool="list_directory">
.
</call>
```

Also, generate tool variations to prevent token memorization:
* `<call tool="save_document">`
* `<call tool="patch_script">`

The model must infer: “These all perform filesystem writing.”

---

# 4. Environment Awareness Dataset

The model must understand:

* OS differences
* GPU availability
* internet access
* filesystem permissions
* memory constraints
* missing dependencies
* containerization
* sandbox restrictions
* execution budgets

Example:
If internet_access = false:
the model should stop attempting web search.

---

# 5. Multi-Agent Committee Data

Train:

* planner disagreements
* critic interventions
* recovery loops
* consensus building
* plan revision
* safety overrides

The model should understand:

* collaborative reasoning
* self-correction
* delegated cognition

---

# PHASE 3 — SYNTHETIC DATA GENERATION

We should NOT manually write datasets.

Instead:
build autonomous dataset factories.

---

# DATA GENERATION PIPELINE

## Step 1 — Frontier Model Teacher Generation

Use:

* OpenAI GPT-4o
* Anthropic Claude
* Google Gemini

to generate:

* perfect execution traces
* recovery loops
* critiques
* tool usage examples
* adaptive workflows

---

## Step 2 — Automatic Mutation Engine

Programmatically mutate traces:

* rename tools
* alter schemas
* inject failures
* corrupt outputs
* remove context
* reorder tools
* create ambiguities

This teaches robustness.

---

## Step 3 — Self-Play Simulation

Create:

* planner vs critic
* executor vs validator
* recovery competitions
* tool hallucination repair

This creates emergent behavior datasets.

---

## Step 4 — Real User Telemetry

Eventually:
anonymized real SumoSpace execution traces become training data.

This is the long-term moat.

---

# PHASE 4 — TRAINING STRATEGY

---

# Stage 1 — Supervised Fine-Tuning (SFT)

Train on:

* high-quality traces
* successful workflows
* corrected failures
* committee reasoning

Goal:
teach baseline SumoSpace behavior.

Use:
QLoRA first.

Reason:

* cheap
* scalable
* fast iteration

---

# Stage 2 — Preference Optimization

Use:

* DPO
* ORPO
* SimPO

Train:
good plans vs bad plans.

Teach:

* reliability
* efficiency
* safety
* minimal hallucination

---

# Stage 3 — Tool-Use Reinforcement

Reward:

* successful execution
* valid edits
* passing tests
* successful recovery
* minimal retries

Penalize:

* hallucinations
* loops
* invalid tool calls
* wasted steps

---

# Stage 4 — Long-Horizon Curriculum

Train progressively:
1-step tasks
→
3-step tasks
→
10-step tasks
→
multi-file repo modifications
→
persistent autonomous sessions

---

# PHASE 5 — BENCHMARKING

Traditional benchmarks are insufficient.

We need:
execution benchmarks.

---

# EVALUATION METRICS

Measure:

* task completion
* retry intelligence
* recovery capability
* hallucination rate
* tool correctness
* execution efficiency
* planning quality
* adaptability
* schema robustness
* environment adaptation
* rollback correctness
* long-horizon reliability

---

# PHASE 6 — DEPLOYMENT

Support:

* GGUF
* EXL2
* AWQ
* GPTQ
* ONNX
* mobile quantization

Target:

* Ollama
* llama.cpp
* vLLM
* edge runtimes
* mobile runtimes

---

# FINAL VISION

The end goal is NOT:
“another coding assistant.”

The end goal is:
a production-grade autonomous execution intelligence runtime capable of:

* software engineering
* enterprise automation
* robotics cognition
* persistent research
* autonomous workflows
* distributed agent collaboration

The system should feel less like:
“an LLM wrapper”

and more like:
“an operating system for autonomous intelligence.”
