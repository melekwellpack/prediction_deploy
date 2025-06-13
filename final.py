import os
import re
import requests
import random
import datetime
import time
import json
import difflib
from pathlib import Path
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta, SU, MO, FR
from functools import lru_cache

load_dotenv(override=True)
# Configuration dictionary that can be modified through the UI
CONFIG = {
    "min_sms_length": 70,
    "default_max_chars": 149,
    "batch_size": 5,
    "max_attempts": 8,
    "api_timeout": 10,  # seconds
    "retry_delay": 1    # seconds between retries
}

def update_config(new_config):
    """Updates the configuration with new values"""
    CONFIG.update(new_config)
    return CONFIG


def process_ui_inputs(data):
    # Convert UI input format to the format expected by generate_sms_variants
    processed_inputs = {
        "sms_type": data.get("smsType", ""),
        "sector": data.get("sector", ""),
        "operation_type": data.get("operationType", ""),
        "link_type": data.get("linkType", ""),
        "orientation": data.get("orientation", ""),
        "partner_name": data.get("partnerName", ""),
        "keywords": data.get("keywords", "").split(",") if isinstance(data.get("keywords"), str) else data.get("keywords", []),
        "use_variable_link": data.get("useVariableLink", False),
        "max_number_of_caracters_without_link": data.get("maxCharacters", 149),
        "include_holiday": data.get("includeHoliday", False),
        "holiday_date": data.get("holidayDate", "")
    }
    
    # Handle date range if provided
    if data.get("startDate") and data.get("endDate"):
        processed_inputs["start_date"] = data.get("startDate")
        processed_inputs["end_date"] = data.get("endDate")
        
    return processed_inputs


def validate_inputs(data):
    """Validates all required inputs are present and in correct format"""
    required_fields = ["sms_type", "sector", "partner_name", "keywords"]
    
    for field in required_fields:
        if not data.get(field):
            return False, f"Missing required field: {field}"
            
    # Check keywords format
    if not isinstance(data.get("keywords"), list) and not isinstance(data.get("keywords"), str):
        return False, "Keywords must be a list or comma-separated string"
        
    return True, ""


# ----- Holiday Data -----

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


@lru_cache(maxsize=128)
def get_cached_holiday_info(date_str):
    """Cache holiday information to prevent repeated calculations"""
    return classify_holiday(date_str)

# ----- Date and Holiday Utility Functions -----
def get_easter_sunday(year):
    """Computes Easter Sunday date using Anonymous Gregorian algorithm."""
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
    return datetime.date(year, month, day)

def resolve_rule_date(rule: str, year: int) -> datetime.date:
    if rule == "first_sunday_of_march":
        return datetime.date(year, 3, 1) + relativedelta(weekday=SU(+1))
    if rule == "last_sunday_of_may":
        return datetime.date(year, 5, 31) + relativedelta(weekday=SU(-1))
    if rule == "third_sunday_of_june":
        return datetime.date(year, 6, 1) + relativedelta(weekday=SU(+3))
    if rule == "first_monday_of_september":
        return datetime.date(year, 9, 1) + relativedelta(weekday=MO(+1))
    if rule == "first_sunday_of_october":
        return datetime.date(year, 10, 1) + relativedelta(weekday=SU(+1))
    if rule == "last_friday_of_november":
        return datetime.date(year, 11, 30) + relativedelta(weekday=FR(-1))
    if rule == "monday_after_black_friday":
        black_friday = datetime.date(year, 11, 30) + relativedelta(weekday=FR(-1))
        return black_friday + datetime.timedelta(days=3)
    if rule == "easter_sunday":
        return get_easter_sunday(year)
    raise ValueError(f"Unknown holiday rule: {rule}")

def get_holidays_in_week(target_date: datetime.date) -> list:
    """Returns a list of holidays in the same week of the given date."""

    # Convert string to date object if needed
    if isinstance(target_date, str):
        target_date_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    else:
        target_date_obj = target_date

    year = target_date_obj.year
    holidays = HOLIDAYS_DATA

    start_of_week = target_date_obj - datetime.timedelta(days=target_date_obj.weekday())
    end_of_week = start_of_week + datetime.timedelta(days=6)
    holidays_in_week = []

    for h in holidays:
        if "rule" in h:
            holiday_date = resolve_rule_date(h["rule"], year)
        else:
            holiday_date = datetime.date(year, h["month"], h["day"])

        if start_of_week <= holiday_date <= end_of_week and holiday_date != target_date_obj:
            holidays_in_week.append((h["name"], h["type"], holiday_date))

    return holidays_in_week

def classify_holiday(date_str: str) -> tuple:
    """Returns 'commercial' if the date is a commercial holiday, else None"""
    target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    year = target_date.year
    holidays = HOLIDAYS_DATA

    for h in holidays:
        if "rule" in h:
            holiday_date = resolve_rule_date(h["rule"], year)
        else:
            holiday_date = datetime.date(year, h["month"], h["day"])

        if holiday_date == target_date:
            return h["name"], h["type"]

    return None, None


# ----- Prompt Generation Functions -----

def sms_prompt(inputs: dict, holiday_name: str = None, holiday_type: str=None, include_holiday: bool = True):
    # Determine max character length based on user inputs
    if inputs.get("use_variable_link") and "max_number_of_caracters_without_link" in inputs:
        max_length = int(inputs.get("max_number_of_caracters_without_link"))
    else:
        max_length = 149 if not inputs.get("use_variable_link") else 120

    prompt = f"""Tu es un expert en marketing digital. Génère un SMS promotionnel EN FRANÇAIS de minimum 70 caractères et maximum {max_length} caractères. Respecte strictement la limite de caractères.

    Type de SMS : {inputs['sms_type']}
    Secteur : {inputs['sector']}
    Type d'opération : {inputs['operation_type']}
    Type de lien court : {inputs['link_type']}
    Orientation : {inputs['orientation']}
    Nom du partenaire : {inputs['partner_name']}
    Mots-clés : {', '.join(inputs['keywords'])}
    Inclure un lien personnalisé si demandé.

    """
    
    prompt += """
    IMPORTANT: EXACTEMENT **UN SEUL SMS** par completion. Ne génère pas plusieurs SMS à la fois.
    + Ne donne **pas** d'instructions supplémentaires.
    + Ne réponds **PAS** avec autre chose que le SMS.

    IMPORTANT : Ne **pas** utiliser d'abbreviations. Mettres les mots en entier. 
    INTERDIT : N'utilise pas des phrases trop générales comme "Profitez dès maintenant" ou "Offre exceptionnelle" sans préciser ce qui est offert. Chaque message doit contenir un contenu concret et spécifique lié à l’offre.
    IMPORTANT : Ne pas couper les phrases. UTILISE des phrases complètes et cohérentes.
    IMPORTANT : Tu dois obligatoirement utiliser TOUS les mots-clés listés. Si il y a un pourcentage (ex : 50%), il doit être mentionné de manière visible dans le message.
    Ne **pas** inclure des nombres ou des caractères qui ne sont pas pertinents pour le message. 
    INTERDIT : Ne PAS inclure la longueur du message, entre parenthèse.
    INTERDIT : Ne PAS inventer de faux chiffres ou des informations non fournies dans les inputs.
    INTERDIT : Ne PAS inventer des pourcentages promotionnels.
    IMPORTANT : Assure-toi que le SMS est complet, avec des phrases entières et cohérentes. Ne termine JAMAIS un message par des mots comme “de”, “en”, “avec”, “pour”, “sur”, “chez”, ou un adjectif seul. Le message doit se terminer par une phrase complète avec une ponctuation claire.

    N'inclure **pas** de lien réel, mais utiliser uniquement la variable {RICH} (par exemple : {RICH}). Ne pas générer un lien complet ni une URL.

    Tu ne dois absolument répondre qu'avec un seul SMS propre, sans commentaires, ni notes, ni éléments décoratifs. Le message ne doit contenir que le texte, pas d'explication, pas d'indication de longueur, pas de texte en dehors du SMS.
    Ne génère **pas** plusieurs options dans un même message.

    """

    if inputs.get("use_variable_link"):
        prompt += "Utilise une variable pour le lien (ex: {RICH}). Ne pas générer de lien.\n"

    if include_holiday and holiday_name and holiday_type=='commercial':
        prompt += f"Inclure une référence subtile à une fête commerciale pertinente « {holiday_name} ».\n"
    else:
       prompt += "Ne pas faire référence à des événements festifs.\n" 

    prompt += "Fournis uniquement le texte du SMS, sans retour à la ligne, ni introduction, ni explication, ni hashtags, ni emojis."

    return prompt

def sms_prompt_without_holiday(inputs: dict):
    
    return sms_prompt(inputs, include_holiday=False)


def progress_callback(message, percentage=None, status="processing"):
    """Can be used to update the UI about generation progress"""
    # In a more sophisticated setup, this could use websockets
    # For now, we'll just print to console
    if percentage is not None:
        print(f"Progress: {percentage}% - {message}")
    else:
        print(f"Status: {status} - {message}")
    # This function would be called from within your generation code


# ----- SMS Generation and Processing Functions -----

def calculate_effective_length(text):
    """Calculate effective length of text, excluding {RICH} placeholder."""
    if "{RICH}" in text:
        return len(text) - len("{RICH}")
    else:
        return len(text)
   
def smart_truncate(text, max_length=140, preserve_rich=True, min_length=70):
    """
    Intelligently truncate text to avoid cutting words or sentences mid-way.
    Preserves {RICH} placeholder if needed.
    """
    # Calculate effective length excluding {RICH}
    effective_length = calculate_effective_length(text)

    # If text is already within range, return it as is
    if min_length <= effective_length <= max_length:
        return text
     
    # Handle texts with {RICH} placeholder
    if preserve_rich and "{RICH}" in text:
        parts = text.split("{RICH}")
        main_text = parts[0].strip()
       
        # Calculate available length for main text
        # Allow space for " {RICH}." at the end
        available_length = max_length - 8  
       
        # Find the last sentence break within available length
        sentence_breaks = [m.start() for m in re.finditer(r'[.!?]\s+', main_text[:available_length])]
       
        if sentence_breaks:
            # Truncate at the last complete sentence
            truncated = main_text[:sentence_breaks[-1]+1].strip()
        else:
            # If no sentence break, find the last word break
            last_space = main_text[:available_length].rfind(' ')
            if last_space > 0:
                truncated = main_text[:last_space].strip()
            else:
                # As a last resort, just cut at available_length
                truncated = main_text[:available_length].strip()
       
        # Ensure truncated text meets minimum length
        if calculate_effective_length(truncated + " {RICH} ") < min_length:  
            # Try to find a longer segment that still fits
            longer_breaks = [m.start() for m in re.finditer(r'[.!?]\s+', main_text)]
            for break_pos in longer_breaks:
                test_truncated = main_text[:break_pos+1].strip()
                if min_length <= calculate_effective_length(test_truncated + " {RICH} ") <= max_length:
                    truncated = test_truncated
                    break

            # If we couldn't find a good truncation point, return the original if it fits max_length
            if calculate_effective_length(truncated + " {RICH} ") < min_length and calculate_effective_length(main_text + " {RICH} ")  <= max_length:
                truncated = main_text

        # Ensure punctuation
        if truncated and truncated[-1] not in ['.', '!', '?']:
            truncated += '.'
           
        # Reattach the {RICH} placeholder
        return f"{truncated.strip()} {'{RICH}'}."
   
    else:
        # For text without {RICH}
        # Find the last sentence break within max_length
        sentence_breaks = [m.start() for m in re.finditer(r'[.!?]\s+', text[:max_length])]
       
        if sentence_breaks:
            # Truncate at the last complete sentence
            truncated = text[:sentence_breaks[-1]+1].strip()
        else:
            # If no sentence break, find the last word break
            last_space = text[:max_length].rfind(' ')
            if last_space > 0:
                truncated = text[:last_space].strip()
            else:
                # As a last resort, just cut at max_length
                truncated = text[:max_length].strip()
       
            # If we couldn't find a good truncation point, return the original if it fits max_length
            if len(truncated) < min_length and len(text) <= max_length:
                truncated = text

        # Add date pattern detection to clean up any date references
        date_patterns = [
            r'jusqu\'au \d{1,2} \w+',
            r'avant le \d{1,2} \w+',
            r'du \d{1,2} au \d{1,2} \w+',
            r'jusqu\'à fin \w+',
            r'jusqu\'à la fin \w+'
        ]
       
        for pattern in date_patterns:
            text = re.sub(pattern, '', text)

        # Ensure punctuation
        if truncated and truncated[-1] not in ['.', '!', '?']:
            truncated += '.'
           
        return truncated.strip()


def clean_sms_text(text, min_length=70):
    """Clean up SMS text by removing character count and extra whitespace."""
   
    original_length = len(text)
   
    # Store original before cleaning
    original_text = text

    # Remove character count notations like (123 caractères) or (123 characters)
    text = re.sub(r'\(\d+\s*(characters|caractères|caracteres|chars|car\.)\)[^\w]*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\d+\s*(characters|caractères|caracteres|chars|car\.)\][^\w]*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(characters|caractères|caracteres|chars|car\.)[^\w]*$', '', text, flags=re.IGNORECASE)

    # Remove quotes that might be added by the model
    text = text.strip('"\'»«\""')
    # Remove dangling asterisks
    text = re.sub(r'\*+$', '', text).strip()
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove extra whitespace at beginning and end
    text = text.strip()
    # Remove emojis using a comprehensive regex pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
    )
    text = emoji_pattern.sub('', text)
    # Remove any mention of "code" or "codes"
    text = re.sub(r'\bcode\b|\bcodes\b', '', text, flags=re.IGNORECASE)
   
    # Check for incomplete sentences or words at the end
    # This helps catch cases like "cultivez votre." where the noun is missing
    incomplete_endings = [
        r'votre\s*\.\s*$',  # "votre."
        r'notre\s*\.\s*$',  # "notre."
        r'faites\s*\.\s*$',  # "faites."
        r'avec\s*le\s*\.\s*$',  # "avec le."
        r'\ble\s*\.\s*$',  # " le."
        r'\bla\s*\.\s*$',  # " la."
        r'\bde\s*\.\s*$',  # " de."
        r'\bdu\s*\.\s*$',  # " du."
        r'\bun\s*\.\s*$',  # " un."
        r'\bune\s*\.\s*$',  # " une."
        r'\bdes\s*\.\s*$',  # " des."
        r'\ben\s*\.\s*$',  # " en."
        r'\bà\s*\.\s*$',  # " à."
        r'\bau\s*\.\s*$',  # " au."
        r'\baux\s*\.\s*$',  # " aux."
        r'\bpour\s*\.\s*$',  # " pour."
        r'\bpar\s*\.\s*$',  # " par."
        r'\bsur\s*\.\s*$',  # " sur."
        r'\bdans\s*\.\s*$',  # " dans."
        r'\bavec\s*\.\s*$',  # " avec."
        r'\bet\s*\.\s*$',  # " et."
        r'\bou\s*\.\s*$',  # " ou."
        r'\bchez\s*\.\s*$',  # " chez."
    ]
   
    for pattern in incomplete_endings:
        if re.search(pattern, text):
            # Remove the incomplete phrase and trailing period
            text = re.sub(pattern, '', text)
   
    # Look for sentences ending with prepositions, articles, etc.
    text = text.strip()
    if text:
        words = text.split()
        if len(words) >= 2:
            last_word = words[-1].lower().strip('.!?,:;')
            common_non_ending_words = ['de', 'du', 'des', 'le', 'la', 'les', 'un', 'une',
                                      'en', 'à', 'au', 'aux', 'pour', 'par', 'sur', 'dans',
                                      'avec', 'et', 'ou', 'ce', 'ces', 'cette', 'mon', 'ma',
                                      'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre',
                                      'votre', 'leur', 'vos', 'nos', 'leurs', 'chez', 'à']
           
            if last_word in common_non_ending_words:
                # Remove the last word as it creates an incomplete sentence
                text = ' '.join(words[:-1])
                # Add a period if needed
                if text and text[-1] not in ['.', '!', '?', ':']:
                    text += '.'

    # Fix spacing issues between words
    text = re.sub(r'([a-zéèêëàâäôöûüùïî])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
    text = re.sub(r'([a-zéèêëàâäôöûüùïî])(\d)', r'\1 \2', text)  # Add space between letter and number
    text = re.sub(r'(\d)([a-zéèêëàâäôöûüùïî])', r'\1 \2', text)  # Add space between number and letter
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space

    # Ensure proper punctuation ending if missing
    if text and text[-1] not in ['.', '!', '?', ':']:
        text += '.'
       
    return text


def is_too_similar(new_sms, existing_sms_list):
    """More aggressive similarity check to prevent repetitive patterns."""
   
    # Check for common repetitive patterns
    common_starts = [
        "profitez de notre offre",
        "découvrez notre offre",
        "offre exclusive",
        "offre exceptionnelle"
    ]
   
    # Count messages starting with common phrases
    start_pattern_count = 0
    for existing in existing_sms_list:
        for pattern in common_starts:
            if existing.lower().startswith(pattern):
                start_pattern_count += 1
   
    # If we already have 2 messages with common starts, reject any new one with same pattern
    if start_pattern_count >= 2:
        for pattern in common_starts:
            if new_sms.lower().startswith(pattern):
                return True
   
    # Regular similarity checks
    if not existing_sms_list:
        return False
       
    for existing in existing_sms_list:
        # Check for exact match
        if new_sms.lower() == existing.lower():
            return True
           
        # Simple similarity measure
        similarity = difflib.SequenceMatcher(None, new_sms.lower(), existing.lower()).ratio()
       
        # More aggressive rejection threshold (70% match)
        if similarity > 0.65:
            return True
   
    return False


def is_french_text(text):
    """Simple check if text is likely French and not English."""
    # List of common English words that shouldn't appear in French SMS
    english_words = ['the', 'and', 'with', 'for', 'your', 'you', 'click', 'here', 'check', 'discover']
   
    # Check if any common English words are present
    text_lower = text.lower()
    for word in english_words:
        if f" {word} " in f" {text_lower} ": # Add spaces to match whole words
            return False
   
    # Check for some common French words (simplified approach)
    french_words = ['votre', 'vous', 'notre', 'pour', 'avec', 'chez', 'découvrez', 'offre', 'profitez']
   
    # Count French words present
    french_count = sum(1 for word in french_words if f" {word} " in f" {text_lower} ")
   
    # If we have at least one French indicator, it's probably French
    return french_count > 0


def enhance_prompt_for_french(base_prompt, use_variable_link, max_char):
    """Add French language requirement to prompt with stronger anti-template instructions."""
    french_instruction = "IMPORTANT: Générez le SMS UNIQUEMENT en français. Ne pas utiliser de mots en anglais."
   
    # Add length guidance with consideration for link placeholder
    # If using variable link, we need to reserve space for the {RICH} placeholder
    link_length = 30 if use_variable_link else 0  # Estimate {RICH} placeholder length
    max_char_without_link = max_char - link_length if use_variable_link else max_char

    length_guidance = f"CRITIQUE: Le SMS DOIT ABSOLUMENT avoir AU MINIMUM 70 caractères, et au maximum {max_char_without_link} caractères. Les SMS de moins de 70 caractères seront REJETÉS."
   
    # Add link instruction based on use_variable_link
    link_instruction = ""
    if use_variable_link:
        link_instruction = "Incluez le placeholder '{RICH}' à la fin du message où le lien sera inséré."
    else:
        link_instruction = "Ne pas inclure de liens ou placeholders dans le message. Ne faire AUCUNE reférence à des liens."
   
    # Add strong diversity and anti-templating instructions
    diversity = "IMPORTANT: N'utilisez AUCUN template ou structure répétitive. Chaque SMS doit être ENTIÈREMENT unique et original."
    uniqueness = "Évitez à tout prix les formulations comme 'Profitez de notre offre' ou 'Découvrez' au début des messages."
    anti_template = "NE GÉNÉREZ PAS de messages qui suivent des modèles similaires entre eux. Variez fortement le style et la structure."
    creative = "Soyez créatif dans vos formulations. Utilisez des approches variées, des questions, des exclamations, des métaphores."
    authentic = "Créez des messages qui sonnent authentiques et personnalisés, pas comme des messages marketing génériques."
    specificity = "Utilisez TOUS les mots-clés et le secteur d'activité pour créer des messages spécifiques et contextuels. Ne pas mettre de codes promos dans le message. Ne pas répéter le nom de partenaire dans un même message."
    # ADD INSTRUCTIONS TO AVOID DATE REFERENCES
    no_dates = "CRUCIAL: N'incluez JAMAIS de référence de dates ou périodes temporelles comme 'jusqu'au 27 décembre' ou 'avant le 15 janvier' ou similaire. Ne mentionnez AUCUNE date spécifique, délai etc. AUCUNE référence temporelle n'est autorisée."
   
    # ADD INSTRUCTIONS TO AVOID INCOMPLETE SENTENCES
    complete_sentences = """CRITIQUE: Assurez-vous que chaque phrase est COMPLÈTE. Vérifiez que chaque phrase se termine par un objet si elle commence par un verbe + préposition.
    INCORRECT: 'Profitez de notre offre exclusive en décoration !'
    CORRECT: 'Profitez de notre offre exclusive en décoration intérieure !'
    INCORRECT: 'Économisez 50% sur nos.'
    CORRECT: 'Économisez 50% sur nos produits bio.'
    Ne terminez JAMAIS une phrase avec des prépositions, articles, ou expressions incomplètes.
    Vérifiez DEUX FOIS que chaque phrase est complète avec un sujet, un verbe et des compléments appropriés si nécessaire."""

    return f"{base_prompt}\n\n{french_instruction}\n{length_guidance}\n{link_instruction}\n{diversity}\n{uniqueness}\n{anti_template}\n{creative}\n{authentic}\n{specificity}\n{no_dates}\n{complete_sentences}"



def query_mistral(prompt, num_responses=5, temperature=0.8):
    """Query Mistral API with simplified error handling."""
    API_KEY = os.getenv("MISTRAL_API_KEY_1",)
    # print("🔑 Clé Mistral :", API_KEY)

    API_URL = "https://api.mistral.ai/v1/chat/completions"
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": temperature,
        "n": num_responses,
    }
   
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        completions = response.json()["choices"]
       
        return [c["message"]["content"].strip() for c in completions]
       
    except requests.exceptions.HTTPError as e:
        # Handle rate limit errors (HTTP 429)
        if hasattr(e, 'response') and e.response.status_code == 429:
            print(f"Rate limit hit. Waiting 2 seconds and retrying with smaller batch...")
            time.sleep(2)
           
            # Try again with smaller batch
            payload["n"] = max(1, num_responses - 1)
            try:
                response = requests.post(API_URL, headers=HEADERS, json=payload)
                response.raise_for_status()
                completions = response.json()["choices"]
                return [c["message"]["content"].strip() for c in completions]
            except:
                print("Second attempt failed.")
       
        print(f"API error: {e}")
        return []
   
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def is_incomplete_ending(text):
    incomplete_words = {
        'de', 'du', 'des', 'le', 'la', 'un', 'une', 'en', 'à', 'au', 'aux',
        'pour', 'par', 'sur', 'dans', 'avec', 'et', 'ou', 'chez'
    }
    if not text:
        return False
    last_word = text.strip().split()[-1].lower().strip('.!?,:;')
    return last_word in incomplete_words


def generate_sms_batch(prompt_func, category_name, required_count, use_variable_link, max_char, existing_sms=None, max_attempts=None):
    """Generate SMS messages with improved truncation to avoid cutting words."""
    existing_sms = existing_sms or []
    category_sms = []
    min_length = CONFIG["min_sms_length"]
    max_attempts = max_attempts or CONFIG["max_attempts"]

    progress_callback(f"Starting generation for {category_name}", 0)
    
    attempts = 0
    while len(category_sms) < required_count and attempts < max_attempts:
        try:
            # Choose temperature with some randomness for diversity
            temp = 0.5 + (attempts * 0.1)  # Increase temperature with each attempt
            temp = min(0.95, temp)  
           
            # Calculate how many SMS we still need
            remaining = required_count - len(category_sms)
            batch_size = min(CONFIG["batch_size"], remaining + 2)  # Request a few more than needed
            
            progress_callback(f"Attempt {attempts+1}/{max_attempts} for {category_name}", 
                             int(100 * len(category_sms) / required_count))
           
            # Generate the prompt
            base_prompt = prompt_func()
           
            # Add explicit instructions to avoid cut-off sentences
            completion_instructions = """
            IMPORTANT: Créez des messages complets avec des phrases entières. Ne coupez JAMAIS un message en plein milieu d'une phrase.
            Chaque message doit être autonome et compréhensible. Évitez les formulations comme 'votre.', 'notre.', 'cette.' à la fin.
            Si vous incluez '{RICH}', assurez-vous qu'il est placé à la fin.
            Le message DOIT avoir AU MOINS {min_length} caractères - c'est une exigence STRICTE.
            Ne mentionnez AUCUNE date ou période temporelle (comme 'jusqu'au 27 décembre').
            Ne pas inclure de codes promo ou des pourcentages sauf si inclue dans les mots-clés.
            N'INVENTE PAS D'OFFRES, de chiffres ou d'informations non fournies dans les inputs.
            Toutes les phrases doivent être complètes et cohérentes et finir pas une ponctuation claire.
            Inclure tous les mots-clés et le nom du partenaire pour créer des messages spécifiques et contextuels.
            Ne coupe jamais les mots surtout pas le nom du partenaire.
            """
           
            enhanced_prompt = enhance_prompt_for_french(base_prompt + completion_instructions, use_variable_link, max_char)
           
            completions = query_mistral(enhanced_prompt, batch_size, temperature=temp)
           
            # Process completions
            for completion in completions:
                if len(category_sms) >= required_count:
                    break
               
                cleaned = clean_sms_text(completion)
               
               
                # FIRST LENGTH CHECK - Skip immediately if too short
                effective_length = calculate_effective_length(cleaned)
                if effective_length < min_length:
                    continue

                # Use smart truncation function
                if use_variable_link:
                    smart_max_length = max_char if "{RICH}" not in cleaned else max_char
                    cleaned = smart_truncate(cleaned, smart_max_length, preserve_rich=True, min_length=min_length)
                else:
                    cleaned = smart_truncate(cleaned, max_char, preserve_rich=False, min_length=min_length)
               
                # SECOND LENGTH CHECK - Skip if truncation made it too short
                effective_length = calculate_effective_length(cleaned)
                if effective_length < min_length:
                    # Try to extend the message if it's close to the minimum length
                    if effective_length >= min_length - 15:
                        # Extract keywords and partner name for extending the message
                        partner_name = ""
                        keywords = []
                        sector = ""
                       
                        if "partner_name" in prompt_func.__closure__[0].cell_contents:
                            partner_info = prompt_func.__closure__[0].cell_contents
                            partner_name = partner_info.get("partner_name", "")
                            keywords = partner_info.get("keywords", [])
                            sector = partner_info.get("sector", "")
                       
                        # Generate extension text based on available info
                        extension = ""
                        if partner_name:
                            extension = f" Uniquement chez {partner_name}."
                        elif sector:
                            extension = f" Spécialiste en {sector}."
                        elif keywords and len(keywords) > 0:
                            random_keyword = random.choice(keywords)
                            extension = f" {random_keyword.capitalize()} garantis."
                        else:
                            extension = " Offre à durée limitée."
                       
                        # Add extension if it would make the text long enough
                        if effective_length + len(extension) >= min_length:
                            if cleaned[-1] in ['.', '!', '?']:
                                cleaned = cleaned[:-1] + extension
                            else:
                                cleaned = cleaned + extension
                   
                    # Skip if still too short
                    effective_length = calculate_effective_length(cleaned)
                    if effective_length < min_length:
                        continue
               
                # Validation checks
                if not is_french_text(cleaned):
                    continue
               
                if is_too_similar(cleaned, existing_sms + category_sms):
                    continue
               
                # Handle variable link placeholder SAFELY
                if use_variable_link:
                    # Remove any existing {RICH} tag first
                    cleaned = cleaned.replace("{RICH}", "").strip()

                    # Remove trailing punctuation
                    cleaned = re.sub(r'[.!?…",;:]+$', '', cleaned).strip()

                    # If it ends in an incomplete word like "avec", "de", etc., remove the last word
                    if is_incomplete_ending(cleaned):
                        cleaned = ' '.join(cleaned.split()[:-1]).strip()
                    # Add proper punctuation before appending {RICH}
                    if cleaned and cleaned[-1] not in ['.', '!', '?']:
                        cleaned += '.'

                    # Append {RICH} at the very end
                    cleaned = re.sub(r'[.!?…",;:]+$', '', cleaned).strip() + " {RICH}"
                else:
                    # Make sure {RICH} is not present
                    cleaned = cleaned.replace("{RICH}", "").strip()
                    cleaned = re.sub(r'\s+', ' ', cleaned)
                    if is_incomplete_ending(cleaned):
                        cleaned = ' '.join(cleaned.split()[:-1]).strip()
                    if cleaned and cleaned[-1] not in ['.', '!', '?']:
                        cleaned += '.'

                # FINAL LENGTH CHECK with link placeholder included
                final_effective_length = calculate_effective_length(cleaned)
                if final_effective_length < min_length:
                    # One last attempt to extend it
                    if "partner_name" in prompt_func.__closure__[0].cell_contents:
                        partner_name = prompt_func.__closure__[0].cell_contents.get("partner_name", "")
                        sector = prompt_func.__closure__[0].cell_contents.get("sector", "")
                       
                        extension = ""
                        if partner_name and sector:
                            extension = f" {partner_name}, spécialiste en {sector}."
                        elif partner_name:
                            extension = f" Uniquement chez {partner_name}."
                        elif sector:
                            extension = f" Spécialiste en {sector}."
                       
                        # Add extension if it fits
                        if extension and len(cleaned) + len(extension) <= max_char + (6 if "{RICH}" in cleaned else 0):
                            # Find where to insert the extension (before {RICH} if present)
                            if "{RICH}" in cleaned:
                                parts = cleaned.split("{RICH}")
                                cleaned = parts[0].rstrip() + extension + "{RICH}" + parts[1]
                            else:
                                if cleaned[-1] in ['.', '!', '?']:
                                    cleaned = cleaned[:-1] + extension + cleaned[-1]
                                else:
                                    cleaned = cleaned + extension
                   
                # Final check if effectove length meets criteria
                final_effective_length = calculate_effective_length(cleaned)
                if min_length <= final_effective_length <= max_char:
                    category_sms.append(cleaned)
           
            # Small delay to avoid rate limits
            time.sleep(1)
           
        except Exception as e:
            progress_callback(f"Error in batch: {str(e)}", status="error")
            time.sleep(CONFIG["retry_delay"])
       
        attempts += 1
        
    progress_percentage = min(100, int(100 * len(category_sms) / required_count))
    progress_callback(f"Completed {category_name}: {len(category_sms)}/{required_count} messages", 
                     progress_percentage,
                     "completed" if len(category_sms) >= required_count else "partial")
   
    return category_sms


def generate_simple_backup_sms(user_inputs, count=1):
    """Generate more diverse backup SMS if the main generation fails, with minimum 70 characters."""
    partner_name = user_inputs.get("partner_name", "nous")
    use_variable_link = user_inputs.get("use_variable_link", False)
    keywords = user_inputs.get("keywords", [])
    sector = user_inputs.get("sector", "")
   
    # Create context-specific elements
    keyword_phrase = ""
    if keywords and len(keywords) >= 2:
        selected_keywords = random.sample(keywords, 2)
        keyword_phrase = f" {selected_keywords[0]} et {selected_keywords[1]}"
   
    # Extended backup messages that meet the 70-character minimum requirement
    backup_messages = []
   
    # Create different formats for messages with or without links
    if use_variable_link:
        backup_messages.extend([
            f"{partner_name} a une nouvelle offre exceptionnelle pour vous. Découvrez tous les détails ici: {{RICH}}.",
            f"Nouveauté exclusive chez {partner_name}{keyword_phrase}. Plus d'informations disponibles: {{RICH}}.",
            f"Une surprise spéciale vous attend chez {partner_name} cette semaine. Découvrez-la maintenant: {{RICH}}.",
            f"Message important de {partner_name} concernant nos services {keyword_phrase}. Cliquez ici: {{RICH}}.",
            f"Offre spéciale par {partner_name} {keyword_phrase}. Ne manquez pas cette opportunité: {{RICH}}.",
            f"Profitez des avantages exclusifs proposés par {partner_name} pour nos clients fidèles. Détails: {{RICH}}.",
            f"Bonne nouvelle! {partner_name} lance une offre promotionnelle qui pourrait vous intéresser. Voir: {{RICH}}."
        ])
    else:
        backup_messages.extend([
            f"{partner_name} a une nouvelle offre exceptionnelle pour vous. Contactez-nous rapidement pour en profiter.",
            f"Nouveauté exclusive chez {partner_name}{keyword_phrase}. Venez nous voir pour découvrir tous les détails.",
            f"Une surprise spéciale vous attend chez {partner_name} cette semaine. À découvrir dans nos locaux.",
            f"Message important de {partner_name} concernant nos services {keyword_phrase}. Contactez-nous dès maintenant.",
            f"Offre spéciale dans notre secteur {sector}{keyword_phrase}. Disponible pour un temps limité chez {partner_name}.",
            f"Profitez des avantages exclusifs proposés par {partner_name} pour nos clients fidèles. Contactez-nous rapidement.",
            f"Bonne nouvelle! {partner_name} lance une offre promotionnelle dans le secteur {sector} qui pourrait vous intéresser."
        ])
   
    # Ensure we have enough messages and return randomly selected ones
    random.shuffle(backup_messages)
    return backup_messages[:count]


def parse_date_range(start_date_str, end_date_str):
    """
    Parse date range and validate it's between 1 and 7 days.
    Returns tuple of datetime objects (start_date, end_date)
    """
    try:
        # Parse dates
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
       
        # Calculate date difference
        delta = (end_date - start_date).days
       
        # Validate range
        if delta < 0:
            raise ValueError("End date must be after start date")
        if delta > 7:
            raise ValueError("Date range cannot exceed 7 days")
       
        return start_date, end_date
       
    except Exception as e:
        print(f"Error parsing date range: {e}")
        raise


def generate_sms_variants(user_inputs: dict):
    """
    Generate SMS variants based on user inputs, with no fallback to templates.
    """
    # Initialize variables
    main_holiday_name = None
    main_holiday_type = None
    holidays_in_week = []
    all_generated_sms = []
   
    # Get use_variable_link status and determine max character count
    use_variable_link = user_inputs.get("use_variable_link", False)
   
    # Get max character count based on user inputs
    if use_variable_link and "max_number_of_caracters_without_link" in user_inputs:
        # If user specified max chars without link, we use that value
        max_char = int(user_inputs.get("max_number_of_caracters_without_link"))
    else:
        # Default to CONFIG value
        max_char = CONFIG["default_max_chars"]
   
    # Process date range if provided
    date_range = []
    if user_inputs.get("start_date") and user_inputs.get("end_date"):
        try:
            start_date, end_date = parse_date_range(
                user_inputs["start_date"],
                user_inputs["end_date"]
            )
           
            # Convert dates to strings for display
            date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
            print(f"Using date range: {date_range[0]} to {date_range[1]}")
           
            # Add date range to user inputs for use in prompts
            user_inputs["date_range"] = date_range
           
        except Exception as e:
            print(f"Error with date range: {e}")
            # Continue without date range
   
    # Check for holiday information
    if user_inputs.get("include_holiday") and user_inputs.get("holiday_date"):
        try:
            # Parse the date string correctly
            holiday_date_str = user_inputs["holiday_date"]

            # Get the holiday information using cached function
            main_holiday_name, main_holiday_type = get_cached_holiday_info(holiday_date_str)
           
            # Get the holiday information directly using the string
            main_holiday_name, main_holiday_type = classify_holiday(holiday_date_str)
            if main_holiday_name and main_holiday_type:
                print(f"Main Holiday: {main_holiday_name} ({main_holiday_type})")
            else:
                print("No holiday found for the specified date.")
               
            # Check for additional holidays in the same week
            holidays_in_week = get_holidays_in_week(holiday_date_str)
            if holidays_in_week:
                print(f"Additional holidays in the week: {', '.join([h[0] for h in holidays_in_week if h and len(h) > 0])}")
        except Exception as e:
            print(f"Error processing holiday date: {e}")
   
    # Generate SMS with retries if needed
    try:
        # Track how many SMS we need to generate for each type
        holiday_sms_required = {}
        total_holiday_sms = 0

        # Add main holiday SMS if applicable
        if main_holiday_name and main_holiday_type:
            holiday_sms_required[(main_holiday_name, main_holiday_type)] = 2
            total_holiday_sms += 2

        # Add additional holidays if applicable
        if holidays_in_week:
            for holiday_info in holidays_in_week:
                if len(holiday_info) >= 2:
                    holiday_name, holiday_type = holiday_info[:2]
                    if holiday_name == main_holiday_name:
                        continue

                    holiday_sms_required[(holiday_name, holiday_type)] = 2
                    total_holiday_sms += 2
       
        normal_count = max(0, 6 - total_holiday_sms)
       
        # Generate holiday SMS if applicable
        for (holiday_name, holiday_type), count in holiday_sms_required.items():
            print(f"Generating {count} SMS for holiday: {holiday_name}")
            holiday_prompt = lambda: sms_prompt(user_inputs, holiday_name, holiday_type, include_holiday=True)
            holiday_sms = generate_sms_batch(
                holiday_prompt,
                f"Holiday-{holiday_name}",
                count,
                use_variable_link,
                max_char,
                all_generated_sms
            )
            all_generated_sms.extend(holiday_sms)
       
        # Generate normal SMS if needed
        if normal_count > 0:
            normal_prompt = lambda: sms_prompt_without_holiday(user_inputs)
            # Increase max_attempts to try harder before giving up
            normal_sms = generate_sms_batch(
                normal_prompt,
                "Normal",
                normal_count,
                use_variable_link,
                max_char,
                all_generated_sms,
                max_attempts=15
            )
            all_generated_sms.extend(normal_sms)
       
        # If we still don't have 6 SMS, try one more time with higher temperature
        total_required = max(6, total_holiday_sms)
        if len(all_generated_sms) < total_required:
            needed = total_required - len(all_generated_sms)
            #print(f"Still need {needed} more SMS. Trying again with higher temperature...")
           
            # Try with higher temperature for more creativity
            def high_temp_prompt():
                base = sms_prompt_without_holiday(user_inputs)
                return base + "\n\nIMPORTANT: Soyez extrêmement créatif et unique. Évitez tout format similaire aux messages précédents."
           
            extra_sms = generate_sms_batch(
                high_temp_prompt,
                "Extra",
                needed,
                use_variable_link,
                max_char,
                all_generated_sms,
                max_attempts=10
            )
            all_generated_sms.extend(extra_sms)
   
    except Exception as e:
        print(f"Error in SMS generation: {e}")
   
    # Final check to ensure all SMS meet the minimum length
    final_sms_list = []
    min_length = 70
    for sms in all_generated_sms:
        effective_length = calculate_effective_length(sms)
        if effective_length >= min_length:
            final_sms_list.append(sms)
        else:
            print(f"Removing SMS that's too short ({effective_length} chars): {sms}")
   
    # If we don't have enough SMS after filtering, add some backup SMS
    if len(final_sms_list) < 6:
        needed = 6 - len(final_sms_list)
        backup_sms = generate_simple_backup_sms(user_inputs, count=needed)
        # Ensure these backup SMS meet the minimum length
        for sms in backup_sms:
            effective_length = calculate_effective_length(sms)
            if effective_length >= min_length:
                final_sms_list.append(sms)
   
    progress_callback("SMS generation complete", 100, "completed")
    return "\n\n".join(final_sms_list)


"""if __name__ == "__main__":
    # Example usage
    query_mistral("Test prompt for Mistral API")
    dotenv_path = find_dotenv()  
    print(f"Using .env file at {dotenv_path}")"""