# Metric: Response Quality

Evaluate the agent response on the following dimensions (score 1–5 each):

- **Accuracy**: Is the information factually correct?
- **Relevance**: Does it directly address the user's request?
- **Conciseness**: Is it appropriately brief without omitting key detail?
- **Tool usage**: Were tools invoked correctly and only when necessary?

Return JSON only — no preamble, no markdown:
```json
{"accuracy": N, "relevance": N, "conciseness": N, "tool_usage": N, "overall": N}
```
