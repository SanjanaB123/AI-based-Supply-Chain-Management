"""
conftest.py — loaded by pytest before any test modules.
Pre-mocks missing dependencies so imports don't fail during collection.
"""
import sys
from unittest.mock import MagicMock

# Mock google.cloud.storage (not installed locally)
_mock_storage = MagicMock()
_mock_google_cloud = MagicMock()
_mock_google_cloud.storage = _mock_storage
sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = _mock_google_cloud
sys.modules["google.cloud.storage"] = _mock_storage

# Mock pymongo (not installed locally)
_mock_pymongo = MagicMock()
sys.modules["pymongo"] = _mock_pymongo
sys.modules["pymongo.errors"] = _mock_pymongo.errors

# Mock airflow (not installed locally)
for mod in [
    "airflow", "airflow.sdk", "airflow.operators",
    "airflow.operators.python", "airflow.providers",
    "airflow.providers.smtp", "airflow.providers.smtp.operators",
    "airflow.providers.smtp.operators.smtp", "airflow.exceptions",
]:
    sys.modules[mod] = MagicMock()

# Make AirflowException a real exception class so `raise` works
sys.modules["airflow.exceptions"].AirflowException = type(
    "AirflowException", (Exception,), {}
)
