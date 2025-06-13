#flask_app.py
import os
import io

import pandas as pd
import numpy as np

from flask import (
    Blueprint, current_app, request,
    jsonify, render_template, redirect,
    url_for, session, flash, send_file
)
from werkzeug.utils import secure_filename

# Import your new scheduling function
from time_predictions import predict_sms_schedule
from utils import (
    preprocess_data,
    train_model,
    process_results,
    create_visualizations
)

# Blueprint Declaration
predict_bp = Blueprint(
    "predict_bp",
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Global State
model = None
encoders = {}
df = None

# Upload & train endpoint



@predict_bp.route("/schedule", methods=["POST"])
def schedule():
    #if not session.get("data_loaded"):
      #  return jsonify({"success": False,
                      #  "error": "Upload and train data before scheduling."}), 400

    data = request.get_json(force=True)
    required = ["operation_type","sms_type","sector",
                "partner_name","short_link_type",
                "message","date_debut","date_fin"]
    missing = [f for f in required if f not in data or not data[f]]
    if missing:
        return jsonify({"success": False,
                        "error": f"Missing fields: {', '.join(missing)}"}), 400

    # build nouveau_sms dict expected by pred_divided
    nouveau_sms = {
      "Type d'opération":   data["operation_type"],
      "Type de SMS":        data["sms_type"],
      "Secteur":            data["sector"],
      "orientation du sms": data.get("sms_orientation",""),
      "Message":            data["message"],
      "date_debut":         data["date_debut"],
      "date_fin":           data["date_fin"],
    }
    try:
        result = predict_sms_schedule(nouveau_sms,
                                      model_path=current_app.config.get("MODEL_PATH","sms_models.pkl"))
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



@predict_bp.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Please select a file to upload")
        return redirect(url_for("index"))

    folder = current_app.config.setdefault("UPLOAD_FOLDER", "uploads")
    os.makedirs(folder, exist_ok=True)

    fname = secure_filename(file.filename)
    path = os.path.join(folder, fname)
    file.save(path)

    try:
        raw_df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        processed_df, features, target, loaded_encoders = preprocess_data(raw_df)

        X = processed_df[features].astype(np.float32)
        y = processed_df[target].astype(np.float32)

        global model, encoders, df
        model = train_model(X, y)
        encoders = loaded_encoders

        processed_df["Predicted_Success"] = model.predict(X)
        df = process_results(processed_df)

        graphs = create_visualizations(df)

        session.update({
            "data_loaded": True,
            "filename": fname,
            "record_count": len(df),
            "avg_success": f"{df['Predicted_Success'].mean()*100:.2f}%"
        })

        return render_template(
            "results.html",
            file_name    = fname,
            record_count = len(df),
            avg_success  = session["avg_success"],
            graphs       = graphs
        )

    except Exception as e:
        flash(f"Error processing file: {e}")
        return redirect(url_for("index"))

# View data endpoint
@predict_bp.route("/view_data")
def view_data():
    if not session.get("data_loaded"):
        flash("Upload & train first")
        return redirect(url_for("index"))

    cols = [
        "Message",
        "Predicted_Success_Display",
        "Top_1_Success", "Top_1_Heure",
        "Top_2_Success", "Top_2_Heure",
        "Top_3_Success", "Top_3_Heure"
    ]
    data = df[cols].head(100).to_dict("records")

    return render_template(
        "view_data.html",
        data=data,
        columns=cols,
        total=len(df),
        showing=len(data)
    )

# Download endpoint
@predict_bp.route("/download")
def download():
    if not session.get("data_loaded"):
        flash("No data to download")
        return redirect(url_for("index"))

    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)

    return send_file(
        buf,
        as_attachment=True,
        download_name="sms_success_predictions.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Predict → Scheduling endpoint
@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # validate required fields
        required = ["operation_type","sms_type","sector",
                    "sms_orientation","message",
                    "date_debut","date_fin"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {', '.join(missing)}"
            }), 400

        nouveau_sms = {
            "Type d'opération":   data["operation_type"],
            "Type de SMS":        data["sms_type"],
            "Secteur":            data["sector"],
            "orientation du sms": data["sms_orientation"],
            "Message":            data["message"],
            "date_debut":         data["date_debut"],
            "date_fin":           data["date_fin"],
        }

        # This will load sms_models.pkl internally
        result = predict_sms_schedule(nouveau_sms, model_path="sms_models.pkl")

        return jsonify({"success": True, **result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# AI prediction endpoint
@predict_bp.route("/ai-timing", methods=["POST"])
def ai_timing():
    from AI_top3 import get_top_3_sms_combinations
    
    data = request.get_json(force=True)
    
    campaign_data = {
        "sms_type": data.get("sms_type"),
        "sector": data.get("sector"),
        "operation_type": data.get("operation_type"),
        "orientation": data.get("sms_orientation"),
        "partner_name": data.get("partner_name"),
        "link_type": data.get("short_link_type"),  # could be None
        "keywords": data.get("keywords", []),
        "include_holiday": data.get("include_holiday", False),
        "use_variable_link": data.get("use_variable_link", False),
        "start_date": data.get("date_debut"),
        "end_date": data.get("date_fin")
    }
    
    try:
        combinations = get_top_3_sms_combinations(data.get("message", ""), campaign_data)
        return jsonify({"success": True, **combinations})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

