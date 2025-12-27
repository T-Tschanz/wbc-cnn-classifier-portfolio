
import psutil

def monitor_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)
