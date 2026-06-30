# AMAVKUB — Containerized Kubernetes Deployment of AMAV

> AMAV (Adversarial Multi-Agent Validation) runs four independent LLMs through a structured debate loop — a Proposer drafts, two Critics evaluate in parallel, an Arbiter rules on consensus. This repo packages that system as a Docker container and deploys it as a Kubernetes batch Job, with a production-hardening pass to handle real upstream API failures gracefully.

---

## What this is

AMAVKUB is a containerized, Kubernetes-orchestrated deployment of the AMAV multi-agent debate system. It takes the original AMAV codebase and adds:

- A **Dockerfile** packaging the app and its dependencies into a runnable image
- A **Kubernetes Job manifest** (not a Deployment — AMAV is a one-shot batch task, not a long-running service)
- **Secret/ConfigMap separation** for the five required API credentials versus tunable runtime parameters
- **Retry-with-exponential-backoff logic** (via `tenacity`) added to both LLM-calling paths, closing a real gap discovered during deployment testing where a single transient upstream error (HTTP 429/5xx) would crash the entire debate mid-run

## Architecture
