import http.server
import os
import socketserver
import threading
import time
import webbrowser

PORT = 8000

# Force the server to run in the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def start_server():
    Handler = http.server.SimpleHTTPRequestHandler

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True

    # Allow port reuse to avoid "Address already in use" errors if you restart quickly
    socketserver.TCPServer.allow_reuse_address = True

    with ThreadedHTTPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    time.sleep(0.5)

    # Explicitly target index.html instead of the root directory
    webbrowser.open(f"http://localhost:{PORT}/index.html")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server.")
