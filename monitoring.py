from prometheus_client import start_http_server

def start_monitoring():
    start_http_server(8000)
