# AI Coding Gym – Exercise Instructions

See AGENTS.md in this repository for full instructions.

## Key Requirements

- Create a session log at `<problem_id>/.log/<agent>-YYYYMMDD-HHMMSS.md` with an entry for every user interaction
- See AGENTS.md for the exact log format
- **Do not directly search on websites or online resources for solutions**
- **Do not modify test files**

## CLI Quick Reference

```
# SWE-bench (bug fix)
aicodinggym swe fetch <problem_id>
aicodinggym swe test <problem_id>       # Local tests (needs Docker)
aicodinggym swe submit <problem_id>

# MLE-bench (ML competition)
aicodinggym mle download <competition_id>
aicodinggym mle submit <competition_id> -F predictions.csv

# Code Review
aicodinggym cr fetch <problem_id>
aicodinggym cr submit <problem_id> -f review.md
```

## Session Log Format

Append to `<problem_id>/.log/<agent>-YYYYMMDD-HHMMSS.md` for each user message:

```
## Entry <N>
**Time:** <ISO-8601>
**User prompt:** <verbatim>
**Approach:** <1-3 sentences>
**Files touched:** <list>
**Outcome:** <1 sentence>
```
