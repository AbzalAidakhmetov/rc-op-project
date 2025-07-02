import http.server
import socketserver
import os

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve files from the current directory
        super().__init__(*args, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    print("\n" + "="*50)
    print("You can now listen to your audio files in your browser.")
    print("VS Code may show a popup to open the URL. If not, copy the link above.")
    print("\nTo listen to your converted file, navigate to:")
    print(f"http://localhost:{PORT}/converted_p225_to_p226.wav")
    print("\nTo listen to the original source file, navigate to:")
    print(f"http://localhost:{PORT}/data/wav48_silence_trimmed/p225/p225_001_mic1.flac")
    print("\nPress Ctrl+C to stop the server.")
    print("="*50)
    httpd.serve_forever() 