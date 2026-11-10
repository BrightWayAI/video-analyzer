import os

# Hosted mode must be on before the server module is imported: it decides
# which tools get registered.
os.environ["VIDEO_ANALYZER_HOSTED"] = "1"
os.environ.setdefault("MCPRUSH_TOKEN", "mgw_test_token")
os.environ.setdefault("PUBLIC_BASE_URL", "https://files.example.test")
