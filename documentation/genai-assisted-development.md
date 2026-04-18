# GenAI-Assisted Development

GenAI tools like Amazon Q Developer can accelerate code review, documentation writing, and knowledge discovery — but only when used with clear, specific prompts. Vague prompts produce vague output. The techniques below make the difference between a useful response and a generic one.

---

## Amazon Q Developer in the IDE

Amazon Q Developer runs directly in your IDE (VS Code, JetBrains) and has access to your open files, workspace context, and cursor position. This context awareness is what makes it more useful than a generic chat interface for development tasks.

### Key capabilities

| Capability | How to trigger |
|-----------|---------------|
| Chat with file context | `@filename` in the chat panel to include a specific file |
| Chat with folder context | `@foldername` to include all files in a directory |
| Workspace-aware chat | `@workspace` to include relevant files automatically |
| Inline code completion | `Alt+C` / `Option+C` to trigger manually |
| Code review | `/review` slash command |
| Test generation | `/test` slash command |
| Documentation generation | `/docs` slash command |
| Agentic coding | `/dev` slash command for multi-file changes |

### Saved prompts

Store reusable prompt templates in `~/.aws/amazonq/prompts/` and reference them with `@prompt_name`. This is useful for team-wide standards — everyone uses the same review checklist or docstring template.

```
~/.aws/amazonq/prompts/
├── review-checklist.md     # @review-checklist
├── numpy-docstring.md      # @numpy-docstring
└── security-review.md      # @security-review
```

### Workspace rules

Place rules in `[workspace_root]/.amazonq/rules/` — they are automatically included in every chat and inline completion request. Use them to encode project-specific conventions so the AI always follows your standards without being reminded.

```
.amazonq/rules/
├── guidelines.md    # coding standards, naming conventions
├── structure.md     # project layout
└── tech.md          # tech stack and library versions
```

---

## Code Review with GenAI

### What GenAI reviews well

- Security issues — hardcoded credentials, SQL injection, missing input validation
- Logic errors — off-by-one, null handling, incorrect conditions
- Missing error handling — unhandled exceptions, missing fallbacks
- Code style and naming — inconsistent conventions, unclear names
- Documentation gaps — missing docstrings, outdated comments
- Performance anti-patterns — Python loops over arrays, repeated computation

### What still needs human review

- Business logic correctness — the AI does not know your domain rules
- Data leakage — requires understanding of the full pipeline context
- Architecture decisions — trade-offs that depend on team and system constraints
- Security in context — the AI can flag patterns but cannot assess your threat model

### Effective review prompts

Be specific about what you want reviewed and what standards to apply:

```
Review @model.py for:
- Missing input validation before the predict() call
- Any place where exceptions are silently swallowed
- NumPy-style docstrings on all public methods
- Python 3.12 type hints — no typing.Optional or typing.List
```

```
Review the changes in @handler.py against our coding standards in @guidelines.md.
Focus on error handling and logging patterns.
```

```
Check @feature_pipeline.py for data leakage — specifically:
- Any feature computed using data after the label timestamp
- Any scaler or encoder fitted on the full dataset before splitting
```

### Iterative review

Use follow-up prompts to drill into specific findings:

```
You flagged a potential null handling issue on line 42.
Show me what the fix should look like following our fail-fast pattern.
```

```
The function _compute_discount has no docstring.
Generate a NumPy-style docstring for it based on the implementation.
```

### Review a diff before committing

```
I'm about to commit these changes to the feature store client.
Review only the modified lines in @feature_store.py for:
- Breaking changes to the public API
- Missing error handling on the new retry logic
- Any hardcoded values that should come from config
```

---

## Documentation with GenAI

### Generate a module docstring

Provide the file and ask for the specific format:

```
Generate a module-level docstring for @ladder.py following the NumPy format in @readme.md.
Include the Functions/Routines section listing all public functions,
a Notes section explaining the business context, and a See Also section.
```

### Generate a function docstring

Point at the function and specify the standard:

```
Generate a NumPy-style docstring for the _compute_psi function in @metrics.py.
Use Python 3.12 type hints. Include Parameters, Returns, Raises, Notes, and Examples sections.
The Examples section should use doctest syntax.
```

### Fill in missing docstrings across a module

```
@validation.py is missing docstrings on several functions.
Generate NumPy-style docstrings for every function that does not have one.
Follow the conventions in @documentation/readme.md.
```

### Update outdated documentation

```
The docstring on the split() method in @sampler.py describes the old behaviour.
The function signature has changed — it now accepts a stratify parameter.
Update the docstring to match the current implementation.
```

### Generate inline comments for complex logic

```
The _enforce_monotonic_constraints function in @ladder.py has no inline comments.
Add comments explaining why each step is necessary, not what it does.
Focus on the business rules and constraints, not the mechanics.
```

### Generate a Markdown guide from code

```
Read @training.py and generate a Markdown guide explaining the training pipeline.
Structure it as: overview → data flow diagram → step-by-step explanation → configuration.
Write it for a new team member who understands Python but not this codebase.
```

---

## Prompt Engineering Skills

### 1. Provide context explicitly

The AI cannot read your mind. Tell it what file, what function, what standard, and what outcome you want.

```
# ❌ Vague
Review my code.

# ✅ Specific
Review @handler.py for missing error handling.
Our standard is in @guidelines.md — every exception must be logged with an ERROR prefix
before being re-raised. Flag any place this is not followed.
```

### 2. Reference your own standards

Point the AI at your existing documentation so it follows your conventions, not generic ones:

```
Generate a docstring for this function following the NumPy format defined in @documentation/readme.md.
Use Python 3.12 type hints as shown in the examples there.
```

### 3. Constrain the scope

Broad requests produce broad answers. Narrow the scope to get actionable output:

```
# ❌ Too broad
Review the whole project.

# ✅ Scoped
Review only the predict() method in @model.py.
Check for: input validation, error handling, and logging completeness.
```

### 4. Ask for reasoning, not just findings

Ask the AI to explain why something is a problem — this helps you learn and verify the finding is real:

```
You flagged line 87 as a potential data leakage issue.
Explain why this is a problem and what the correct fix is.
```

### 5. Use chain prompting for complex tasks

Break large tasks into steps rather than asking for everything at once:

```
Step 1: Read @pipeline.py and list all public functions that are missing docstrings.
Step 2: For each function you listed, generate a NumPy-style docstring.
Step 3: Show me the complete updated file with all docstrings added.
```

### 6. Ask for alternatives

When reviewing architecture or design decisions, ask for trade-offs:

```
I'm using a Redis cache with a 5-minute TTL for prediction scores.
What are the trade-offs of this approach?
What alternative caching strategies should I consider for this use case?
```

### 7. Validate generated code

Never accept generated code without reviewing it. Ask the AI to explain what it generated:

```
You generated a new _validate_features method.
Walk me through what it does line by line and explain any assumptions it makes.
```

### 8. Use the `/review` command for structured reviews

The `/review` slash command runs a structured code review scan covering security, quality, and best practices. Use it as a first pass before a human review:

```
/review
```

Then follow up with targeted prompts based on the findings.

---

## Documentation Workflow

A practical workflow for keeping documentation current using GenAI:

```
1. Write the code
2. /review — catch issues before documenting broken code
3. Ask Q to generate docstrings for new functions
4. Review and edit generated docstrings — add business context the AI cannot know
5. Ask Q to update the module-level docstring to reflect new functions
6. Ask Q to check if any existing Markdown guides need updating
7. Run mkdocs serve to verify rendered output
```

### Keeping docs in sync with code changes

When you modify a function, ask Q to check if the documentation is still accurate:

```
I changed the signature of compute_features() in @features.py.
It now accepts a weights parameter. Check:
1. Is the docstring still accurate?
2. Are there any Markdown guides in @documentation/ that reference this function?
3. What needs to be updated?
```

---

## What GenAI Cannot Replace

- **Domain knowledge** — the AI does not know your business rules, data contracts, or stakeholder requirements
- **Final judgement on correctness** — generated code and docs must be reviewed by someone who understands the system
- **Security sign-off** — AI can flag patterns but cannot assess your specific threat model or compliance requirements
- **Architecture decisions** — trade-offs depend on team capability, system constraints, and organisational context
- **Keeping secrets secret** — never paste credentials, PII, or sensitive data into any AI tool

---

## Rules

- Always review AI-generated code before committing — treat it as a first draft, not a final answer
- Use `@filename` to give the AI the exact context it needs — do not describe the code, show it
- Store team standards in `.amazonq/rules/` — this ensures every AI interaction follows your conventions automatically
- Ask for reasoning when the AI flags an issue — a finding you cannot explain is a finding you cannot fix correctly
- Use saved prompts (`~/.aws/amazonq/prompts/`) for repeated tasks like docstring generation — consistency matters
- Never paste credentials, API keys, or PII into the chat — the AI does not need them and they should not leave your environment
