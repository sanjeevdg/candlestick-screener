import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)

socketio = SocketIO(
    app,
    async_mode="eventlet",
    cors_allowed_origins="*",
)

@app.route("/")
def index():
    return "OK"

@socketio.on("connect")
def on_connect():
    print("✅ socket connected")

if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )
