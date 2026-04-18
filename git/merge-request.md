# Merge Request

A merge request (MR) — or pull request (PR) — is the primary mechanism for reviewing, discussing, and integrating code changes. In ML projects, MRs cover not just code but also config changes, data pipeline updates, and model experiments.

---

## MR Structure

### Keep MRs small and focused

Each MR should address a single concern. Large MRs are harder to review, slower to merge, and more likely to introduce bugs.

| MR size | Lines changed | Review quality |
|---------|--------------|----------------|
| Small | < 200 | Thorough, fast turnaround |
| Medium | 200–500 | Acceptable, may need multiple passes |
| Large | > 500 | Superficial, high risk of missed issues |

If a feature requires > 500 lines, break it into a chain of smaller MRs that build on each other.

### One concern per MR

- ✅ `feat: add batch inference endpoint` — one feature
- ✅ `fix: correct null handling in preprocessing` — one fix
- ❌ `feat: add batch endpoint + refactor config loader + update dependencies` — too many concerns

---

## MR Title and Description

### Title

Use the same conventional commit format as your commit messages:

```
feat(serving): add health check endpoint
fix(training): correct data split ratio calculation
chore: bump scikit-learn to 1.6.1
```

### Description template

```markdown
## What
Brief description of what this MR does.

## Why
The problem or requirement this addresses.

## Changes
- Added/modified/removed X
- Updated config for Y

## Testing
How this was tested (unit tests, manual verification, notebook output).

## Checklist
- [ ] Code follows project style guidelines
- [ ] Pre-commit hooks pass locally
- [ ] Unit tests added/updated
- [ ] Documentation updated
- [ ] No PII or credentials in the diff
- [ ] Config changes are environment-aware
```

---

## Branch Rules

### Source and target branches

| MR type | Source branch | Target branch |
|---------|-------------|---------------|
| Feature | `feature/*` | `staging` or `main` |
| Bug fix | `fix/*` | `staging` or `main` |
| Hotfix | `hotfix/*` | `main` (then backport to `staging`) |
| Release | `staging` | `main` |

### Branch protection

- `main` and `staging` should be protected — no direct pushes
- Require at least one approval before merge
- Require CI pipeline to pass before merge is allowed
- Require branch to be up-to-date with the target before merge

---

## Code Review Guidelines

### For authors

- Self-review the diff before requesting review — catch obvious issues yourself
- Add inline comments on complex logic to guide the reviewer
- Keep the MR description up to date if the scope changes during review
- Respond to every comment — resolve or explain why you disagree
- Avoid force-pushing during review — it makes it harder to track incremental changes

### For reviewers

- Review within 24 hours — don't block the team
- Focus on:

| Priority | What to check |
|----------|--------------|
| High | Correctness, security, PII exposure, credential leaks |
| Medium | Edge cases, error handling, test coverage |
| Low | Naming, style (should be caught by linters), minor readability |

- Use clear prefixes in comments:
  - `blocker:` — must be fixed before merge
  - `suggestion:` — optional improvement
  - `question:` — need clarification, not necessarily a change
  - `nit:` — minor style preference, non-blocking
- Approve when satisfied — don't hold MRs for trivial nits

---

## ML-Specific Review Checklist

ML changes require additional scrutiny beyond standard code review:

| Change type | What to verify |
|-------------|---------------|
| Feature engineering | No data leakage, correct aggregation logic, handles nulls |
| Model config | Hyperparameters are reasonable, no accidental overfit settings |
| Training pipeline | Reproducibility (seed, data snapshot), evaluation metrics logged |
| Serving code | Input validation, error handling, latency impact |
| Data schema | Backward compatible, documented, validated with schema checks |
| Config change | Correct environment scoping, no hardcoded values |

---

## Merge Strategy

| Strategy | When to use |
|----------|------------|
| Squash merge | Feature branches — collapses all commits into one clean commit on the target |
| Merge commit | Release branches — preserves full history for auditability |
| Rebase | Small, linear changes — keeps history flat (avoid on shared branches) |

- Prefer **squash merge** for feature and fix branches — keeps the target branch history clean and readable
- The squash commit message should follow conventional commit format
- Delete the source branch after merge

---

## CI Integration

Configure CI to run automatically on every MR:

```yaml
# GitLab CI example
lint:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'

unit_tests:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
```

At minimum, run these stages on every MR:
- lint & format
- unit tests
- security / PII scan

Expensive stages (training, deployment) should only run on the target branch after merge.

---

## Tips

- Use MR templates — configure a default template in your repository so every MR follows the same structure
- Link to issues/tickets — reference `Closes #123` in the description for automatic issue tracking
- Use draft/WIP MRs for early feedback — prefix with `Draft:` to signal it is not ready for final review
- Review your own MR first — the diff view often reveals issues you missed in the editor
- Avoid approving your own MR — always require at least one other reviewer
