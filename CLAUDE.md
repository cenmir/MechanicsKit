# MechanicsKit — Claude notes

## Release policy (pre-1.0)

The 0.x series moves the **minor** slot slowly on purpose so the version
number stays meaningful until the 1.0 API commitment.

- **PATCH (0.X.Y → 0.X.Y+1)** — bug fixes, docs, performance, and small
  additive refinements to existing functions (new kwargs, new optional
  behavior). Most changes land here.
- **MINOR (0.X.0 → 0.X+1.0)** — new public functions / verbs, or
  meaningful reshuffles of existing ones.
- **MAJOR (X.0.0)** — breaking changes. Reserved for 1.0.

**Rule of thumb:** if the change only *adds an option* to something that
already exists → patch. If it adds a thing the user can import or call
that didn't exist before → minor.

Bump **before `git push`**, not on every commit. Keep `pyproject.toml`
and `mechanicskit/__init__.py` in lockstep, and add a `## [x.y.z]`
section to `CHANGELOG.md`.

A `pre-push` hook in `.git/hooks/pre-push` reprints this policy.
