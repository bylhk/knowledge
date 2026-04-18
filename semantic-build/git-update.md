# Git Update with Semantic Build

This guide covers how to automate version bumping, Git tagging, and changelog generation using semantic build tools. See [readme.md](readme.md) for the full semantic versioning overview.

---

## Overview

The semantic build workflow derives the next version from your commit messages:

```
commit messages → determine bump type → update version ID → create Git tag → generate CHANGELOG.md
```

No manual version editing required — the tooling reads your conventional commits and does everything.

---

## Option 1: Commitizen (Python-native)

Best for Python/ML projects. Install with:

```bash
pip install commitizen
```

### Configuration

Add to `pyproject.toml`:

```toml
[tool.commitizen]
name = "cz_conventional_commits"
version = "1.0.0"
version_files = [
    "pyproject.toml:version",
    "src/__version__.py:__version__",
    "setup.cfg:version",
]
tag_format = "v$version"
changelog_file = "CHANGELOG.md"
update_changelog_on_bump = true
```

`version_files` tells commitizen which files contain the version string — it updates all of them in one go.

### Usage

```bash
# Bump version, update files, create tag, generate changelog — all in one command
cz bump --changelog

# Dry run — see what would happen without making changes
cz bump --dry-run

# Force a specific bump type
cz bump --increment MINOR

# Generate changelog without bumping
cz changelog
```

### What happens when you run `cz bump --changelog`

1. Reads commits since the last tag
2. Determines the bump type (`feat:` → MINOR, `fix:` → PATCH, `feat!:` → MAJOR)
3. Updates the version string in all `version_files`
4. Appends a new section to `CHANGELOG.md` with grouped commits
5. Creates a Git commit with the version bump
6. Creates a Git tag (e.g. `v1.2.0`)

---

## Option 2: semantic-release (Node.js / CI-focused)

Best for CI-driven releases. Install with:

```bash
npm install -D semantic-release
```

### Configuration

Add to `package.json` or `.releaserc.json`:

```json
{
  "branches": ["main"],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/changelog",
    ["@semantic-release/exec", {
      "prepareCmd": "sed -i 's/^version = .*/version = \"${nextRelease.version}\"/' pyproject.toml"
    }],
    "@semantic-release/git",
    "@semantic-release/gitlab"
  ]
}
```

### Usage

```bash
# Run in CI — analyses commits, bumps, tags, publishes
npx semantic-release
```

This is typically called from a CI stage rather than locally.

---

## Option 3: Manual (when automation is not set up)

If you need to do it manually, follow this exact order:

```bash
# 1. Update version ID in source files
# pyproject.toml, __version__.py, setup.cfg, etc.

# 2. Update CHANGELOG.md
# Add a new section at the top:
# ## [1.2.0] - 2025-01-15
# ### Added
# - New batch inference endpoint
# ### Fixed
# - Null handling in feature preprocessing

# 3. Commit the version bump
git add -A
git commit -m "chore(release): v1.2.0"

# 4. Create the Git tag
git tag -a v1.2.0 -m "Release v1.2.0"

# 5. Push commit and tag
git push origin main --follow-tags
```

Manual releases are error-prone — automate with commitizen or semantic-release whenever possible.

---

## CHANGELOG.md Format

The auto-generated changelog follows the [Keep a Changelog](https://keepachangelog.com) convention:

```markdown
# Changelog

## [1.2.0] - 2025-01-15

### Added
- Batch inference endpoint (`feat: add batch inference endpoint`)
- Staging deployment rule (`ci: add staging deployment rule`)

### Fixed
- Null handling in feature preprocessing (`fix: correct null handling`)

## [1.1.0] - 2025-01-02

### Added
- Model drift alerting

### Changed
- Updated xgboost to 2.1.0
```

Commits are grouped by type: `feat:` → Added, `fix:` → Fixed, `refactor:` → Changed, `chore:` → Changed.

---

## CI Integration

Run the version bump automatically in the build stage of your pipeline:

```yaml
# GitLab CI example
semantic_build:
  stage: build
  script:
    - pip install commitizen
    - cz bump --changelog --yes
    - git push origin main --follow-tags
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
```

```yaml
# GitHub Actions example
- name: Bump version
  run: |
    pip install commitizen
    cz bump --changelog --yes
    git push --follow-tags
```

Ensure the CI runner has push permissions (deploy key or token with write access).

---

## Summary

| Step | Commitizen | semantic-release | Manual |
|------|-----------|-----------------|--------|
| Update version ID | ✅ automatic | ✅ automatic | ✏️ edit files |
| Create Git tag | ✅ automatic | ✅ automatic | `git tag -a` |
| Generate CHANGELOG.md | ✅ automatic | ✅ automatic | ✏️ write manually |
| Single command | `cz bump --changelog` | `npx semantic-release` | 5 steps |
| Best for | Python projects, local + CI | CI-only workflows | One-off releases |
