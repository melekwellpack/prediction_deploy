from flask import Blueprint, request, jsonify
import time
import traceback

# Import your existing SMS generation code
from final import (
    generate_sms_variants,
    process_ui_inputs,
    validate_inputs,
    update_config
)

# Create a Blueprint for SMS endpoints
generate_sms_api = Blueprint("generate_sms_api", __name__)

@generate_sms_api.route("/generate-sms", methods=["POST"])
def generate_sms_endpoint():
    try:
        data = request.json
        processed_inputs = process_ui_inputs(data)
        is_valid, error_message = validate_inputs(processed_inputs)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_message
            }), 400

        start_time = time.time()
        sms_list = generate_sms_variants(processed_inputs)
        processing_time = time.time() - start_time

        if isinstance(sms_list, str):
            sms_list = sms_list.split('\n\n')

        return jsonify({
            'success': True,
            'sms_messages': sms_list,
            'count': len(sms_list),
            'processing_time_seconds': round(processing_time, 2)
        })
    except Exception as e:
        print(f"Error in SMS generation: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generate_sms_api.route("/update-config", methods=["POST"])
def update_config_endpoint():
    try:
        config_data = request.json
        updated_config = update_config(config_data)
        return jsonify({
            'success': True,
            'config': updated_config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
