# time_predictions.py
def predict_sms_schedule(nouveau_sms, model_path='sms_models.pkl'):
    import joblib
    from collections import Counter
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    # Load models
    models = joblib.load(model_path)
    success_model = models['success_model']
    weekday_model = models['weekday_model']
    hour_models = models['hour_models']
    preprocessor = models['preprocessor']

    # Parse date range
    date_debut = datetime.strptime(nouveau_sms['date_debut'], "%Y-%m-%d")
    date_fin = datetime.strptime(nouveau_sms['date_fin'], "%Y-%m-%d")
    delta = (date_fin - date_debut).days + 1
    range_days = [date_debut + timedelta(days=i) for i in range(delta)]

    # French weekday and month names
    french_day_names = {
        0: 'lundi', 1: 'mardi', 2: 'mercredi', 
        3: 'jeudi', 4: 'vendredi', 5: 'samedi', 6: 'dimanche'
    }

    french_months = {
        1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai",
        6: "juin", 7: "juillet", 8: "août", 9: "septembre",
        10: "octobre", 11: "novembre", 12: "décembre"
    }

    # Build available day map with actual datetime object for sorting
    available_days_info = {
        french_day_names[d.weekday()]: {
            'date': d,
            'label': f"{french_day_names[d.weekday()].capitalize()} {d.day} {french_months[d.month]}"  # Capitalize for display
        }
        for d in range_days if d.weekday() != 6
    }

    available_days = list(available_days_info.keys())

    # Format and transform new SMS
    sms_df = pd.DataFrame([nouveau_sms])[[ 
        'Type d\'opération', 'Type de SMS', 'Secteur', 'orientation du sms', 'Message'
    ]]
    sms_df['Message'] = sms_df['Message'].astype(str)
    sms_transformed = preprocessor.transform(sms_df)

    # Predict success
    success_pred = success_model.predict(sms_transformed)[0]


    # Predict day probabilities
    day_probas = weekday_model.predict_proba(sms_transformed)[0]
    top_5_indices = np.argsort(day_probas)[::-1][:5]
    top_5_days = weekday_model.classes_[top_5_indices]

    # Match predicted days with available ones (case-insensitive)
    available_days_lower = [d.lower() for d in available_days]
    top_days = [day for day in top_5_days if day.lower() in available_days_lower]

    suggestions = {}
    best_combos = []

    for day in top_days:
        if day not in hour_models:
            continue
        model = hour_models[day]
        base_pred = model.predict(sms_transformed)[0]
        preds = np.clip(np.round(base_pred + np.random.normal(0, 1, 30)), 0, 23).astype(int)
        hour_counts = Counter(preds)

        # Sort hours by success probability (frequency) - keep most_common order
        top_3_hours_by_success = [h for h, _ in hour_counts.most_common(3)]
        top_3_hours_str = [f"{h}h" for h in top_3_hours_by_success]

        day_label = available_days_info[day]['label']
        suggestions[day_label] = top_3_hours_str

        if top_3_hours_str:
            score = day_probas[np.where(weekday_model.classes_ == day)[0][0]]
            best_combos.append((day, top_3_hours_str[0], score))

    # Sort top combos by score
    top_3_combos = sorted(best_combos, key=lambda x: -x[2])[:3]
    top_3_formatted = [
        f"{available_days_info[day]['label']} à {hour}"
        for day, hour, _ in top_3_combos
        if day in available_days_info
    ]

    # Order days by success probability (highest first) instead of chronologically
    ordered_days = sorted(
        [day for day in top_days if day in available_days_info],
        key=lambda d: day_probas[np.where(weekday_model.classes_ == d)[0][0]],
        reverse=True  # Highest probability first
    )
    ordered_day_labels = [available_days_info[d]['label'] for d in ordered_days]

    # Sort suggestions by success probability order
    sorted_suggestions = {
        available_days_info[d]['label']: suggestions[available_days_info[d]['label']]
        for d in ordered_days if available_days_info[d]['label'] in suggestions
    }
    
    return {
        "ordered_days": ordered_day_labels,
        "times_by_day": sorted_suggestions,
        "top_combinations": top_3_formatted
    }