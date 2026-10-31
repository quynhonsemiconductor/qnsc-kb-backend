"""The release this code belongs to.

release-please rewrites the marked line when it merges a release PR (`extra-files` in
release-please-config.json), so the number the API reports moves with pyproject.toml.
`tests/unit/test_build_info.py` fails if the two ever disagree.

A release number says which RELEASE is running, not which merge: dozens of merges land
between two releases. `src/core/build_info.py` adds the commit, which is what tells one
merged PR from the next.
"""

__version__ = "0.1.2"  # x-release-please-version
