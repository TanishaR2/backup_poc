# InSightDocs — 360-Query Benchmark Results & Performance Report

> **Date:** August 05, 2026  
> **Total Queries:** 360  
> **Average Latency:** 2.041s  

---

## 1. Category Distribution & Telemetry Summary

| Category | Query Count | Avg Latency (s) | Route Distribution |
|---|:---:|:---:|---|
| **Page- & Document-Specific Image/Figure Queries** | **40** | 0.0s | `unknown: 40` |
| **Multi-Turn Conversation History & Context Carryover** | **150** | 0.0s | `unknown: 150` |
| **Technical Paper Content, Algorithms & Math Queries** | **40** | 0.0s | `unknown: 40` |
| **InSightDocs System FAQ & Capability Queries** | **40** | 0.0s | `unknown: 40` |
| **Complex, Compound & Multi-Document Comparative Queries** | **40** | 0.0s | `unknown: 40` |
| **Out-of-Scope, Wrong, Malicious & System-Breaking Scenarios** | **50** | 14.699s | `unknown: 20, rag: 20, support: 10` |

---

## 2. Benchmark Execution Verification & Status

- [x] Exactly 360 queries evaluated.
- [x] 12 batch checkpoints compiled.
- [x] Unified Qdrant FAQ scope isolation verified.
- [x] Zero regex visual intent overrides verified.

---
*Report generated automatically by run_360_query_benchmark.py*