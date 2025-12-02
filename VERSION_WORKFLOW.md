# Version Update Workflow

This document describes when and how to update MatKit's version number.

## Quick Reference

```bash
# Update version in matkit/__init__.py
__version__ = '0.1.2'  # Increment appropriately

# Update CHANGELOG.md
# Move items from [Unreleased] to new version section

# Commit and tag
git add .
git commit -m "Release v0.1.2"
git tag -a v0.1.2 -m "Version 0.1.2"
git push && git push --tags
```

## Semantic Versioning (MAJOR.MINOR.PATCH)

### PATCH Version (0.1.X → 0.1.X+1)
**When:** Bug fixes, documentation updates, performance improvements

**Examples:**
- Fixed color interpolation bug in `patch()`
- Fixed typo in docstring
- Improved performance of mesh iteration
- Updated README with clearer examples

**What to do:**
1. Increment patch number in `matkit/__init__.py`
2. Move changes from `[Unreleased]` to new `[0.1.X]` section in CHANGELOG
3. Commit with message: `"Release v0.1.X - Bug fixes"`

### MINOR Version (0.X.0 → 0.X+1.0)
**When:** New features, new functions, backward-compatible additions

**Examples:**
- Added new element type (HEXA20, WEDGE15)
- Added new function (`extract_boundary()`, `plot_mesh()`)
- Added new OneArray methods
- Added SymPy integration

**What to do:**
1. Increment minor number, reset patch to 0 in `matkit/__init__.py`
2. Move changes from `[Unreleased]` to new `[0.X.0]` section in CHANGELOG
3. Commit with message: `"Release v0.X.0 - New features"`

### MAJOR Version (X.0.0 → X+1.0.0)
**When:** Breaking changes that require users to modify their code

**Examples:**
- Changed mesh API (renamed methods, changed parameters)
- Removed deprecated functions
- Changed OneArray indexing behavior
- Changed default parameters that affect existing code

**What to do:**
1. Think carefully! Is this really necessary?
2. Document migration guide for users
3. Increment major number, reset minor and patch to 0
4. Update CHANGELOG with migration notes
5. Commit with message: `"Release vX.0.0 - BREAKING CHANGES"`

## Automated Reminders

**Claude will automatically remind you to bump the version when:**
- A bug is fixed → "📌 Version bump needed: This bug fix warrants v0.1.X (patch)"
- A feature is added → "📌 Version bump needed: This feature warrants v0.X.0 (minor)"
- Breaking change is made → "📌 Version bump needed: This breaking change warrants vX.0.0 (major)"

## When NOT to Update Version

- Every single commit (too granular)
- Work in progress (not ready for release)
- Internal refactoring that doesn't affect users
- Changes to test files only

## Best Practice

1. **During development**: Work in `[Unreleased]` section of CHANGELOG
2. **When ready to release**: Move unreleased items to versioned section and bump version
3. **Frequency**: Update version every few meaningful changes, not every commit

## Example CHANGELOG Workflow

### During development:
```markdown
## [Unreleased]

### Fixed
- Fixed interpolation bug in patch()
```

### When releasing v0.1.2:
```markdown
## [Unreleased]

### Added

### Changed

### Fixed

## [0.1.2] - 2025-12-02

### Fixed
- Fixed interpolation bug in patch()
```

Update `__version__ = '0.1.2'` in `matkit/__init__.py`
