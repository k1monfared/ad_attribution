#!/usr/bin/env sh
# Serve the interactive explorer (docs/) on a free local port and open a browser.
#
# Picks a free port by binding to port 0, serves docs/ with the standard library
# http.server bound to 127.0.0.1 in the background, opens the browser through the
# first available of xdg-open, open, or python3 -m webbrowser, prints the URL,
# and waits on the server until Ctrl-C.

set -eu

ROOT=$(cd "$(dirname "$0")" && pwd)
DOCS="$ROOT/docs"

if [ ! -f "$DOCS/index.html" ]; then
  echo "docs/index.html not found. Run: python scripts/generate_explorer.py" >&2
  exit 1
fi

# Pick a free port by letting the OS assign one (bind to port 0), then reuse it.
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
URL="http://127.0.0.1:$PORT/"

# Serve docs/ in the background.
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DOCS" >/dev/null 2>&1 &
SERVER_PID=$!

# Stop the server on exit or interrupt.
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Give the server a moment to come up.
sleep 1

echo "Serving docs/ at $URL"
echo "Press Ctrl-C to stop."

# Open the browser via the first available launcher, guarded so a missing one
# does not abort the script.
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "$URL" >/dev/null 2>&1 || true
else
  python3 -m webbrowser "$URL" >/dev/null 2>&1 || true
fi

# Wait on the server until interrupted.
wait "$SERVER_PID"
