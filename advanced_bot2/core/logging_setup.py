# logging_setup.py

import logging
import json
import sys
from logging.handlers import RotatingFileHandler
import requests

################################################################################
# 1) JSON Formatter
################################################################################
class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "funcName": record.funcName,
            "message": record.getMessage()
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data)

################################################################################
# 2) Cloud / Slack / Splunk / GCP / AWS => placeholder
################################################################################
class BaseHTTPHandler(logging.Handler):
    """
    Basit bir HTTP handler iskeleti. 
    Her log satırında HTTP POST yapar (senkron).
    Yüksek hacimde yavaşlatma riski var => pratikte asenkron veya shipping agent tercih edilir.
    """
    def __init__(self, endpoint, token=None, level=logging.INFO):
        super().__init__(level)
        self.endpoint = endpoint
        self.token = token

    def emit(self, record):
        msg = self.format(record)
        try:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            requests.post(self.endpoint, headers=headers, data=msg, timeout=3)
        except Exception:
            pass

class AWSCloudWatchHandler(BaseHTTPHandler):
    """
    Örnek => normalde watchtower kütüphanesi tavsiye edilir,
    ama placeholder: endpoint => self.endpoint='https://logs.amazonaws.com/...'
    token => IAM credential vs.
    """
    pass

class GCPLoggingHandler(BaseHTTPHandler):
    """
    GCP => endpoint='https://logging.googleapis.com/v2/entries:write', 
    token => google Oauth token
    """
    pass

class SlackHandler(BaseHTTPHandler):
    """
    Slack => endpoint='https://hooks.slack.com/services/xxx/yyy'
    token => genelde Slack incoming webhook'ta token yoktur, endpoint yeterli.
    """
    pass

class SplunkHandler(BaseHTTPHandler):
    """
    Splunk HEC => endpoint='https://splunk.example.com:8088/services/collector'
    token => 'Splunk <hec_token>'
    """
    pass



################################################################################
# 4) configure_logger => tek fonksiyon
################################################################################
def configure_logger(config: dict):
    logger = logging.getLogger("BotLogger")
    logger.setLevel(logging.INFO)

    # 1) Console Handler
    if config.get("log_to_console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)

    # 2) Dosya (Rotating)
    if config.get("log_to_file", True):
        file_path= config.get("log_file_path","bot.log")
        max_bytes= config.get("log_max_bytes",1_000_000)
        backup_count= config.get("log_backup_count",3)
        file_handler= RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    # 3) AWS
    if config.get("aws_logging", False):
        endpoint= config.get("aws_endpoint","")
        token= config.get("aws_token","")
        aws_handler= AWSCloudWatchHandler(endpoint, token=token)
        aws_handler.setFormatter(JsonFormatter())
        logger.addHandler(aws_handler)

    # 4) GCP
    if config.get("gcp_logging", False):
        endpoint= config.get("gcp_endpoint","")
        token= config.get("gcp_token","")
        gcp_handler= GCPLoggingHandler(endpoint, token=token)
        gcp_handler.setFormatter(JsonFormatter())
        logger.addHandler(gcp_handler)

    # 5) Slack
    if config.get("slack_logging", False):
        endpoint= config.get("slack_webhook_url","")
        slack_handler= SlackHandler(endpoint)
        slack_handler.setFormatter(JsonFormatter())
        logger.addHandler(slack_handler)

    # 6) Splunk
    if config.get("splunk_logging", False):
        endpoint= config.get("splunk_hec_endpoint","")
        token= config.get("splunk_hec_token","")
        sp_handler= SplunkHandler(endpoint, token=token)
        sp_handler.setFormatter(JsonFormatter())
        logger.addHandler(sp_handler)

    return logger


def log(msg, lvl="info"):
    logger = logging.getLogger("BotLogger")
    if lvl=="info":
        logger.info(msg)
    elif lvl=="warning":
        logger.warning(msg)
    elif lvl=="error":
        logger.error(msg)
    else:
        logger.debug(msg)

