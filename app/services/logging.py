import os
import json
import logging
import contextvars
from logging.handlers import RotatingFileHandler

# Context variable to hold the request ID for tracing
request_id_var = contextvars.ContextVar("request_id", default="")

class CorrelationFilter(logging.Filter):
    """
    Filter that injects the current request correlation ID into each log record.
    """
    def filter(self, record):
        record.request_id = request_id_var.get("")
        return True

class PrettyFormatter(logging.Formatter):
    """
    Standard console formatter for human-readable pretty logs:
    [LEVEL] [request_id=xxx] logger_name | message
    """
    def format(self, record):
        req_id = getattr(record, "request_id", "") or "no-request-id"
        log_fmt = f"[{record.levelname}] [request_id={req_id}] {record.name} | {record.getMessage()}"
        if record.exc_info:
            log_fmt += "\n" + self.formatException(record.exc_info)
        return log_fmt

class JSONFormatter(logging.Formatter):
    """
    Structured console formatter for machine-parseable single-line JSON logs.
    """
    def format(self, record):
        req_id = getattr(record, "request_id", "") or "no-request-id"
        # Attempt to parse message as JSON if it is a JSON string (e.g. validator logging)
        msg = record.getMessage()
        try:
            msg_data = json.loads(msg)
        except Exception:
            msg_data = msg

        import datetime
        dt = datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc)
        log_entry = {
            "timestamp": dt.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "request_id": req_id,
            "logger": record.name,
            "message": msg_data
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

def setup_logging():
    """
    Configures standard python logging system based on environmental variables.
    """
    log_format = os.getenv("LOG_FORMAT", "pretty").lower()
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clean up any existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        
    console_handler = logging.StreamHandler()
    console_handler.addFilter(CorrelationFilter())
    
    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(PrettyFormatter())
        
    root_logger.addHandler(console_handler)

# Configure dedicated audit logger
os.makedirs("logs", exist_ok=True)
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # Avoid cluttering console stdout

# Rotating File Handler: 10MB per file, max 10 files
audit_handler = RotatingFileHandler(
    "logs/audit.log",
    maxBytes=10*1024*1024,
    backupCount=10,
    encoding="utf-8"
)
# Formatter writes log records exactly as-is (they will be JSON serialized strings)
audit_handler.setFormatter(logging.Formatter("%(message)s"))
audit_logger.addHandler(audit_handler)

def log_audit_event(event_type: str, request_id: str, patient_id: str, details: dict):
    """
    Helper function to securely write standard audit-safe metadata to logs/audit.log.
    Raw patient messages are strictly excluded to ensure compliance.
    """
    import time
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": request_id or "no-request-id",
        "patient_id": patient_id or "unknown",
        "event_type": event_type,
    }
    # Safely merge details, ignoring any raw text variables like 'message'
    for k, v in details.items():
        if k.lower() not in ["message", "patient_message", "symptoms_text"]:
            event[k] = v
            
    audit_logger.info(json.dumps(event))
