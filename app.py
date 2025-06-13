# main.py
from flask import Flask, render_template
from flask_cors import CORS
import os

# Import your three Blueprints
from fpred           import generate_sms_api   # your SMS‐generation blueprint
from flask_app     import predict_bp         # your prediction blueprint
from schedule_app   import schedule_bp

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# ——————————————————————————————————————
# App configuration
# ——————————————————————————————————————
app.secret_key = "sms_success_prediction_secret_key"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config['MODEL_PATH'] = os.path.join(os.getcwd(), 'sms_models.pkl')

# Optional: tighten CORS if you’re hosting front‐end separately
CORS(app)

# ——————————————————————————————————————
# Routes
# ——————————————————————————————————————
@app.route("/")
def index():
    # your landing page (webPage_simple.html) lives in templates/
    return render_template("webPage_simple.html")

# ——————————————————————————————————————
# Blueprint registration
# ——————————————————————————————————————
# Mount your SMS‐generation endpoints
app.register_blueprint(generate_sms_api)
# Mount your prediction/upload endpoints
app.register_blueprint(predict_bp)
# register new scheduling blueprint
app.register_blueprint(schedule_bp)

# ——————————————————————————————————————
# Entry point
# ——————————————————————————————————————
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
