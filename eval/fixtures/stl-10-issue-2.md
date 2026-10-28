---
case: STL-10
issue: 2
issue_date: 2026-11-12
segment: practitioner
channel: email
note: "Issue 2 is the first issue after the exp-003 correction (2026-11-10) and deliberately carries no notice. Everything else is meant to be clean, so the only expected failure is the missing notice."
---
Failure Modes this week: your LLM judge is less stable than you think.
Swap the answer order and Claude-v1 kept its verdict only 23.8% of the time; GPT-4 managed 65.0% [cite:ext-mtbench-paper].
The paper's conservative fix is to run the judge twice with the order swapped, and only count a win if both runs agree [cite:ext-mtbench-paper].
We think every team running an LLM judge should report this swap test next to its headline accuracy.
Read the explainer.
— Residual Room
