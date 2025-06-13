# schedule_app.py
import os
from flask import Blueprint, current_app, request, jsonify
from time_predictions import predict_sms_schedule

schedule_bp = Blueprint(
    "schedule_bp",
    __name__,
    url_prefix="",
)

@schedule_bp.route("/schedule", methods=["POST"])
def schedule_sms():
    """
    Expects a JSON body with:
      - operation_type
      - sms_type
      - sector
      - orientation_du_sms  (if applicable)
      - message      (the exact text from the preview)
      - date_debut   (YYYY-MM-DD)
      - date_fin     (YYYY-MM-DD)
    Returns JSON:
    {
      "success": true,
      "Suggestions optimales pour envoi du SMS (jours + heures)": { ... },
      "Top 3 meilleures combinaisons jour + heure": [ ... ]
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify(success=False, error="Invalid or missing JSON payload"), 400

    # Basic validation
    required = ["message", "date_debut", "date_fin"]
    missing = [f for f in required if f not in data or not data[f]]
    if missing:
        return jsonify(success=False,
                       error=f"Missing fields: {', '.join(missing)}"), 400

    try:
        # Call your scheduling logic
        result = predict_sms_schedule(data,
                                      model_path=current_app.config.get("MODEL_PATH", "sms_models.pkl"))
        # Wrap in success envelope
        result["success"] = True
        return jsonify(result)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
