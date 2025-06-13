import datetime
import json
import holidays
import requests
import calendar
import re
import os
import locale  
from dotenv import load_dotenv
from dateutil.parser import parse as parse_date
from datetime import datetime, timedelta

load_dotenv(override=True)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY_2")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

def get_top_3_sms_combinations(sms_content, campaign_data):
    def get_holidays(year):
        return holidays.France(years=year)

    def load_commercial_holidays():
        HOLIDAYS_DATA = [
        {
        "name": "Nouvel An",
        "month": 1,
        "day": 1,
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Saint-Valentin",
        "month": 2,
        "day": 14,
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Journée de la Femme",
        "month": 3,
        "day": 8,
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Fête des Grands-Mères",
        "rule": "first_sunday_of_march",
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Pâques",
        "rule": "easter_sunday",
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Fête des Mères",
        "rule": "last_sunday_of_may",
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Fête des Pères",
        "rule": "third_sunday_of_june",
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Rentrée scolaire",
        "rule": "first_monday_of_september",
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Fête des Grands-Pères",
        "rule": "first_sunday_of_october",
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Halloween",
        "month": 10,
        "day": 31,
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Black Friday",
        "rule": "last_friday_of_november",
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Cyber Monday",
        "rule": "monday_after_black_friday",
        "type": "commercial",
        "recurring": True
        },
        {
        "name": "Noël",
        "month": 12,
        "day": 25,
        "type": "commercial",
        "recurring": True
        }
  ]

    def get_rule_based_holiday_date(rule, year):
        if rule == "first_sunday_of_march":
            first_day = datetime(year, 3, 1)
            days_ahead = 6 - first_day.weekday()
            if days_ahead < 0:
                days_ahead += 7
            return first_day + timedelta(days=days_ahead)
        elif rule == "easter_sunday":
            a = year % 19
            b = year // 100
            c = year % 100
            d = b // 4
            e = b % 4
            f = (b + 8) // 25
            g = (b - f + 1) // 3
            h = (19 * a + b - d - g + 15) % 30
            i = c // 4
            k = c % 4
            l = (32 + 2 * e + 2 * i - h - k) % 7
            m = (a + 11 * h + 22 * l) // 451
            month = (h + l - 7 * m + 114) // 31
            day = ((h + l - 7 * m + 114) % 31) + 1
            return datetime(year, month, day)
        elif rule == "last_sunday_of_may":
            _, last_day = calendar.monthrange(year, 5)
            last_date = datetime(year, 5, last_day)
            days_back = last_date.weekday() - 6
            if days_back > 0:
                days_back -= 7
            return last_date + timedelta(days=days_back)
        elif rule == "third_sunday_of_june":
            first_day = datetime(year, 6, 1)
            days_ahead = 6 - first_day.weekday()
            if days_ahead < 0:
                days_ahead += 7
            return first_day + timedelta(days=days_ahead + 14)
        elif rule == "first_monday_of_september":
            first_day = datetime(year, 9, 1)
            days_ahead = (0 - first_day.weekday()) % 7
            return first_day + timedelta(days=days_ahead)
        elif rule == "first_sunday_of_october":
            first_day = datetime(year, 10, 1)
            days_ahead = 6 - first_day.weekday()
            if days_ahead < 0:
                days_ahead += 7
            return first_day + timedelta(days=days_ahead)
        elif rule == "last_friday_of_november":
            _, last_day = calendar.monthrange(year, 11)
            last_date = datetime(year, 11, last_day)
            days_back = last_date.weekday() - 4
            if days_back > 0:
                days_back -= 7
            return last_date + timedelta(days=days_back)
        elif rule == "monday_after_black_friday":
            black_friday = get_rule_based_holiday_date("last_friday_of_november", year)
            return black_friday + timedelta(days=3)
        return None

    def is_commercial_holiday(date):
        commercial_holidays = load_commercial_holidays()
        for holiday in commercial_holidays:
            if "month" in holiday and "day" in holiday:
                if holiday["month"] == date.month and holiday["day"] == date.day:
                    return True, holiday["name"]
            elif "rule" in holiday:
                rule_date = get_rule_based_holiday_date(holiday["rule"], date.year)
                if rule_date and rule_date.date() == date.date():
                    return True, holiday["name"]
        return False, ""

    def call_mistral_api(prompt, model="mistral-small-latest"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2048
        }
        try:
            response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Erreur de requête API : {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Réponse JSON invalide depuis l’API"}

    def prepare_prompt():
        start_date = datetime.strptime(campaign_data["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(campaign_data["end_date"], "%Y-%m-%d")

        jours_disponibles = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() != 6:
                jours_disponibles.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

        info_jours_feriés = ""
        if campaign_data.get("include_holiday", False):
            jours_feriés = []
            current_date = start_date
            while current_date <= end_date:
                fr_holidays = get_holidays(current_date.year)
                if current_date in fr_holidays:
                    jours_feriés.append(f"{current_date.strftime('%Y-%m-%d')} : {fr_holidays[current_date]}")
                is_commercial, name = is_commercial_holiday(current_date)
                if is_commercial:
                    jours_feriés.append(f"{current_date.strftime('%Y-%m-%d')} : {name} (commercial)")
                current_date += timedelta(days=1)
            info_jours_feriés = "Jours fériés pendant la période :\n" + "\n".join(jours_feriés) if jours_feriés else "Aucun jour férié détecté."

        return f"""
    Vous êtes un expert en optimisation de campagnes SMS marketing. En vous basant uniquement sur les données suivantes, recommandez les meilleurs jours et horaires pour envoyer ce SMS afin d’obtenir un taux d’engagement maximal.

    Contenu du SMS : "{sms_content}"

    Paramètres de la campagne :
    - Type de SMS : {campaign_data.get('sms_type', '')}
    - Secteur : {campaign_data.get('sector', '')}
    - Type d’opération : {campaign_data.get('operation_type', '')}
    - Type de lien : {campaign_data.get('link_type', '')}
    - Orientation : {campaign_data.get('orientation', '')}
    - Partenaire : {campaign_data.get('partner_name', '')}
    - Mots-clés : {', '.join(campaign_data.get('keywords', []))}
    - Prise en compte des jours fériés : {"Oui" if campaign_data.get('include_holiday', False) else "Non"}
    - Période : **du {campaign_data['start_date']} au {campaign_data['end_date']}**
    Jours disponibles (hors dimanches) : {', '.join(jours_disponibles)}

    **Contraintes de génération :**
    - Vous devez uniquement proposer des jours **compris dans cette période**.
    - **ABSOLUMENT PAS DE DIMANCHES** DANS LES SUGGESTIONS.
    - Vous pouvez inclure le samedi.
    - **Aucune date ne doit être antérieure au {campaign_data['start_date']} ni postérieure au {campaign_data['end_date']}**.
    - Ne pas inclure l’année dans les jours proposés.
    -Proposez exactement 5 combinaisons jour + heure, classées de la meilleure à la moins bonne.
    -Les jours peuvent se répéter, mais chaque combinaison doit avoir un horaire différent.
    -Triez les combinaisons par ordre d’engagement attendu (du meilleur au moins bon).

    {info_jours_feriés}

    Veuillez retourner un JSON structuré comme suit :
    {{
    "Suggestions optimales pour l’envoi du SMS (jours + heures)": {{
        "Mardi 6 juin": ["11h", "12h", "13h"],
        "Mercredi 7 juin": ["14h", "15h", "13h"],
        "Jeudi 8 juin": ["10h", "11h", "12h"]
    }},
    "Top 3 meilleures combinaisons jour + heure": [
        "Mercredi 7 juin à 14h",
        "Mardi 6 juin à 11h",
        "Jeudi 8 juin à 10h"
        "Mercredi 7 juin à 15h",
        "Mardi 6 juin à 12h"
    ]
    }}

    """

    prompt = prepare_prompt()
    response = call_mistral_api(prompt)
    if "error" in response:
        return []

    try:
        raw_text = response["choices"][0]["message"]["content"]
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                return []

        start = datetime.strptime(campaign_data["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(campaign_data["end_date"], "%Y-%m-%d").date()

        french_months = {
        "janvier": 1, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
        "juillet": 7, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
        }

        raw_combinations = data.get("Top 3 meilleures combinaisons jour + heure", [])
        valid_combinations = []

        #for item in raw_combinations:
        for i, item in enumerate(raw_combinations):
            
            # Try multiple regex patterns - including one WITHOUT day name
            patterns = [
                r'(\w+)\s(\d+)\s(\w+)\sà\s(\d+h)',  # With day name: "Jeudi 21 août à 14h"
                r'(\d+)\s(\w+)\sà\s(\d+h)',  # Without day name: "22 août à 11h"
                r'(\w+)\s+(\d+)\s+(\w+)\s+à\s+(\d+h)',  # With day name + multiple spaces
                r'(\d+)\s+(\w+)\s+à\s+(\d+h)',  # Without day name + multiple spaces
            ]
            
            match = None
            pattern_used = None
            for pattern in patterns:
                match = re.search(pattern, item, re.IGNORECASE)
                if match:
                    pattern_used = pattern
                    break
            
            if match:
                groups = match.groups()
                
                # Handle different patterns
                if len(groups) == 4:  # Pattern with day name: (day_name, day, month, hour)
                    day_name, day_str, month_fr, hour = groups
                elif len(groups) == 3:  # Pattern without day name: (day, month, hour)
                    day_str, month_fr, hour = groups
                else:
                    print(f"Unexpected number of groups: {len(groups)}")
                    continue
                
                try:
                    day = int(day_str)
                    month = french_months.get(month_fr.lower())
                    year = int(campaign_data["start_date"][:4])
                
                    if not month:
                        continue
                        
                    date_obj = datetime(year, month, day).date()

                    # Skip if the date is a Sunday
                    if date_obj.weekday() == 6:
                        continue

                    if start <= date_obj <= end:
                        # Get correct weekday in French
                        weekday_names = {
                            0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 
                            4: "Vendredi", 5: "Samedi", 6: "Dimanche"
                        }
                        correct_weekday = weekday_names[date_obj.weekday()]
                        formatted = f"{correct_weekday} {day} {month_fr} à {hour}"
                        valid_combinations.append(formatted)
                    else:
                        print(f"Date {date_obj} is outside range {start} to {end}")
                        
                except Exception as e:
                    print(f"Error processing date: {str(e)}")
                    continue

        return {"valid_combinations": valid_combinations[:3]}

    except Exception:
        return []