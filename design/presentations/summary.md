# Sovereign Inferencing on Isambard AI - Pilot summary

AGENTIC AI = INFERENCING SERVICE + AI HARNESS
INFERENCING SERVICE = AI MODEL + COMPUTE

## 1. Pilot Study Aims

The aim of this pilot was to establish whether we could give University staff and researchers direct, self-hosted access to open-weight AI models running on our own Isambard AI HPC infrastructure — and connect that access to modern "agentic" AI harnesses, such as claude code, copilot-cli, vscode, OpenCode, pi, i.e. tools that can read, write, and act on a user's own files, code, and data directly. The pilot replaces the backend inferencing service provided by Anthropic, OpenAI or other commercial providers, but re-uses their front end harnesses. There are several reasons we wanted to do this:

* data sovereignty (sensitive or unpublished research data never has to leave University-controlled infrastructure, cannot be used for model training)
* cost control (unmetered open-weight models on our own GPUs, rather than per-token commercial API spend)
* capability (recent open-weight models when combined with harnesses are now close enough to commercial frontier models to be genuinely useful for real research and engineering work, not just a cheaper fallback)
* research independence (AI assisted research priorities are currently being decided by companies like Anthropic, rather than universities)

## 2. Example Use Case

A researcher or research software engineer working with data that can't leave the University (clinical data, unpublished results, commercially sensitive collaborations) wants to use an AI coding/research assistant to help run a literature review, analyse data, debug an analysis pipeline, or develop software — but can't point a commercial AI agent at that data. With this service, they instead connect an AI harness at a model running on Isambard. The resulting AI agent can read their code and data, run analyses, iterate on fixes, and write results back — all within the University's own control. This pilot's own engineering work is itself a live example of the pattern: a large share of the deployment, debugging, and documentation work behind this pilot was done collaboratively with an AI agent directly reviewing logs, and creating code — not as a chatbot answering questions about it from the side.

## 3. Technical Implementation

We built a command-line tool (`ivllm`) that lets a user request an AI model, submits the corresponding job to Isambard AI via SLURM (spanning multiple GPU nodes where needed, coordinated with Ray), and — once the model is serving — opens a secure SSH tunnel back to the user's own machine, exposing a standard OpenAI-compatible API inferencing service on `localhost`. Because the API is a standard interface, any existing AI agent harness (Claude Code, copilot-cli and others) can be pointed at it with no special integration work. The standard OpenAI API could also provide the backend for a wide range of pre-existing or custom built applications (potentially including chatbots). `ivllm` manages the full job lifecycle (startup, health checks, idle shutdown, cancellation, diagnostics capture) and shares installed model runtimes and downloaded model weights across the whole project team, so the setup cost is paid once, not per user.

## 4. Degree of Success

We have successfully deployed and debugged working configurations for a range of current large open-weight models ranging from relatively small (~30B parameter) to moderately large (~780B parameter) and including DeepSeek, Qwen, GLM, MiniMax, and Nemotron model families. `ivllm` is in active day-to-day use by a pilot group including University of Bristol academics, government partners, external academic collaborators, and shows that open-weight models on Isambard are a genuinely useful substitute for a commercial AI subscription without data leaving University control. The pilot group is in the process of being expanded to other UoB research groups in Robotics and Physics.

## 5. Remaining Issues

There are some questions which we have not answered and which would need more long term technical involvement and resourcing:

* We don't have a clear idea of how well this will scale, or how many people it can support.
* Open weight models are being released all the time and configuring new models is an ongoing task.
* The tooling today is still aimed at a technically confident pilot group rather than a general University audience.
* `ivllm` works with Isambard AI and one inference engine (vLLM); a full service would need a decision on whether/how to generalise beyond that.
* We provide access to one AI model at a time for a user, but a better long term solution would be a gateway providing access to multiple models.
* Access control is done by project membership on Isambard. This would need to be reviewed in wider use.
* Safe use of Agentic AI means using the agent in a sandbox. There are various solutions for this but it need to be addressed strategically.
* Isambard AI is setup for batch processing, for AI training. Inferencing is an interactive service, and this tension needs to be resolved.
