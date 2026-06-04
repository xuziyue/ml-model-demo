# Code Review Basic

This skill performs branch-aware code review by first collecting source and destination branches, then reviewing the diff between them.

## Use cases

- API contract and error handling review
- Test coverage and regression checks
- Backward compatibility and migration risk review

## Required workflow

1. Always ask user for source branch and destination branch.
2. Run `git diff <destination>...<source>` to collect changed code.
3. Review only changed lines and nearby context.
4. Output findings by severity with concrete suggestions.

## Prompt spec

See `SKILL.md` for the exact skill behavior and output format.
