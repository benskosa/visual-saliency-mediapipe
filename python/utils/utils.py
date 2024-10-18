from datetime import datetime, timezone as tz, timedelta
from typing import Tuple, Dict, Any
import socket

def current_time() -> float:
    """
    Get the current time in UTC.
    :return: the current time in UTC  
    """
    return datetime.now(tz.utc).timestamp()

def self_ip() -> str:
    """
    Get the IP address of the current machine.
    :return: the IP address of the current machine
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP