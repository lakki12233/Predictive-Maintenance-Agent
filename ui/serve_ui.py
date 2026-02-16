#!/usr/bin/env python3
"""Simple HTTP server to serve the test UI"""
import http.server
import socketserver
import os

PORT = 8080
# Serve from parent directory to access sample files and test resources
DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"🌐 Serving UI at http://localhost:{PORT}/ui/ui.html")
        print(f"📁 Directory: {DIRECTORY}")
        print(f"\n✅ Open in browser: http://localhost:{PORT}/ui/ui.html")
        print(f"⚠️  Make sure API is running at http://localhost:8000")
        print(f"\nPress Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")
