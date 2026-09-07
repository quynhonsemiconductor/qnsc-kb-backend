import os

# BEFORE src.core.config is imported anywhere, because `settings = Settings()` runs at
# import time and reads the process environment exactly once. The Settings default for
# ENVIRONMENT is `production` — deliberately, so a deployment that forgets the variable
# fails closed instead of booting with the committed development secrets — and under that
# default the suite's module-level `settings` would demand externally supplied secrets,
# HTTPS origins and R2 credentials that no test has.
#
# setdefault, not assignment: `ENVIRONMENT=production poetry run pytest` still exercises
# the production branch, which is what the hardening tests need.
os.environ.setdefault("ENVIRONMENT", "development")
