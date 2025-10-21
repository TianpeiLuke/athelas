import re
import numpy as np
import string


# ======================  Decode TOPIC and ENTRY_POINT ================================


TOPIC_ID_mapping = {
    # // -- Buyer to seller topics -- //
    0: "MISSING",
    1: "ORDER_INFO_BUYER_TO_SELLER",
    2: "PRODUCT_DETAILS_BUYER_TO_SELLER",
    3: "SHIPPING_BUYER_TO_SELLER",
    4: "RETURN_POLICY_BUYER_TO_SELLER",
    5: "OTHER_BUYER_TO_SELLER",
    # // New topics
    27: "WHERES_MY_STUFF_BUYER_TO_SELLER",
    28: "RECEIVED_DAMAGED_DEFECTIVE_WRONG_ITEM_BUYER_TO_SELLER",
    29: "CANCEL_ORDER_BUYER_TO_SELLER",
    31: "THANK_YOU_BUYER_TO_SELLER",
    32: "COD_WRONG_CHARGE_BUYER_TO_SELLER",
    33: "RECEIVED_DAMAGED_DEFECTIVE_ITEM_BUYER_TO_SELLER",
    34: "RECEIVED_WRONG_ITEM_BUYER_TO_SELLER",
    35: "OTHER_SHIPPING_BUYER_TO_SELLER_POST_ORDER",
    36: "RETURN_POLICY_BUYER_TO_SELLER_POST_ORDER",
    37: "PRODUCT_DETAILS_BUYER_TO_SELLER_POST_ORDER",
    38: "OTHER_BUYER_TO_SELLER_POST_ORDER",
    39: "INCOMPLETE_SHIPMENT_BUYER_TO_SELLER_POST_ORDER",
    46: "PRODUCT_CUSTOMIZATION_BUYER_TO_SELLER",
    49: "SHIP_FASTER_BUYER_TO_SELLER_POST_ORDER",
    50: "CHANGE_SHIPPING_ADDRESS_BUYER_TO_SELLER_POST_ORDER",
    # Problem with Your Orders page
    51: "PWO_PACKAGE_DIDNT_ARRIVE_REQUEST_REFUND",
    52: "PWO_PACKAGE_DIDNT_ARRIVE_REQUEST_REPLACEMENT",
    53: "PWO_DAMAGED_OR_DEFECTIVE_REQUEST_REFUND",
    54: "PWO_DAMAGED_OR_DEFECTIVE_REQUEST_REPLACEMENT",
    55: "PWO_DIFFERENT_FROM_ORDERED_REQUEST_REFUND",
    56: "PWO_DIFFERENT_FROM_ORDERED_REQUEST_REPLACEMENT",
    # // -- Seller to buyer topics -- //
    6: "ORDER_INFO_SELLER_TO_BUYER",
    7: "SHIPMENT_NOTIFICATION_SELLER_TO_BUYER",
    8: "FEEDBACK_REQUEST_SELLER_TO_BUYER",
    9: "RETURN_POLICY_SELLER_TO_BUYER",
    10: "REFUND_NOTICE_SELLER_TO_BUYER",
    11: "ADDITIONAL_INFO_SELLER_TO_BUYER",
    # // -- Buyer to seller CBA topics -- //
    12: "ORDER_INFO_CBA_BUYER_TO_SELLER",
    13: "PRODUCT_DETAILS_CBA_BUYER_TO_SELLER",
    14: "SHIPPING_CBA_BUYER_TO_SELLER",
    15: "RETURN_POLICY_CBA_BUYER_TO_SELLER",
    16: "RETURNS_AND_REFUNDS_CBA_BUYER_TO_SELLER",
    17: "RECEIVED_DAMAGED_DEFECTIVE_WRONG_ITEM_CBA_BUYER_TO_SELLER",
    18: "WHERES_MY_STUFF_CBA_BUYER_TO_SELLER",
    19: "CANCEL_ORDER_CBA_BUYER_TO_SELLER",
    20: "OTHER_CBA_BUYER_TO_SELLER",
    # //New topics added 4/2012 for new getGroupedWebformTopics API
    40: "OTHER_SHIPPING_CBA_BUYER_TO_SELLER_POST_ORDER",
    41: "RETURN_POLICY_CBA_BUYER_TO_SELLER_POST_ORDER",
    42: "RETURNS_AND_REFUNDS_CBA_BUYER_TO_SELLER_POST_ORDER",
    43: "PRODUCT_DETAILS_CBA_BUYER_TO_SELLER_POST_ORDER",
    44: "OTHER_CBA_BUYER_TO_SELLER_POST_ORDER",
    45: "INCOMPLETE_SHIPMENT_CBA_BUYER_TO_SELLER_POST_ORDER",
    47: "PRODUCT_CUSTOMIZATION_CBA_BUYER_TO_SELLER",
    # // -- Seller to buyer CBA topics -- //
    21: "ORDER_INFO_CBA_SELLER_TO_BUYER",
    22: "SHIPMENT_NOTIFICATION_CBA_SELLER_TO_BUYER",
    23: "FEEDBACK_REQUEST_CBA_SELLER_TO_BUYER",
    24: "RETURN_POLICY_CBA_SELLER_TO_BUYER",
    25: "REFUND_NOTICE_CBA_SELLER_TO_BUYER",
    26: "ADDITIONAL_INFO_CBA_SELLER_TO_BUYER",
}


ENTRY_POINT_mapping = {
    -1: "MISSING",
    0: "UNDEFINED",
    1: "BUYER_MAIL_UI",
    2: "SELLER_MAIL_UI",
    3: "ORDER_HISTORY_PAGE",
    4: "OFFER_LISTING_PAGE",
    5: "MERCHANT_RETURN_SERVICE",
    6: "CN_INVOICE",
    7: "ANYWHERE_HELP",
    8: "DISPUTE_MANAGEMENT",
    9: "MAGAZINES",
    10: "BBC_TEST",
    11: "CS_CENTRAL",
    12: "MAKE_AN_OFFER",
    13: "SELLER_MOBILE_MAIL_UI",
    14: "BUYER_CUSTOMER_SERVICE",
    15: "CONTACT_SELLER_UI",
    16: "ATHENA",
    17: "ACTION_CENTER",
    18: "PROBLEM_WITH_YOUR_ORDER_PAGE",
    19: "BUYER_DESKTOP_UI",
    20: "3P_Developer_API",
    21: "SMART_CS",
    22: "BUYER_THREADED_UI",
}


Claim_action_encode = {
    "": 0,
    "GRNT": 0,
    "NOGR": 1,
    "SRFD": 0,
    "BWDR": 1,
    "INVG": 2,
    "HOLD": 2,
    "CBCK": 1,
    "SCAN": 0,
    "ACAN": 2,
    "BWDQ": 1,
    "RABI": 2,
    "RASI": 2,
    "BCNF": 2,
    "APAY": 2,
    "SPAY": 0,
}


Customer_action_encode = {
    "": 0,
    "FLUSH": 0,
    "NOT_QUALIFY": 0,
    "PASS": 0,
    "SOLICIT": 1,
    "WARN": 1,
    "CLOSE": 1,
}


Label_combination_mapping = np.array(
    [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [2.0, 1.0, 2.0]]
)


aws_translate_supported_lang = [
    "af",
    "sq",
    "am",
    "ar",
    "hy",
    "az",
    "bn",
    "bs",
    "bg",
    "ca",
    "zh",
    "zh-TW",
    "hr",
    "cs",
    "da",
    "fa-AF",
    "nl",
    "en",
    "et",
    "fa",
    "tl",
    "fi",
    "fr",
    "fr-CA",
    "ka",
    "de",
    "el",
    "gu",
    "ht",
    "ha",
    "he",
    "hi",
    "hu",
    "is",
    "id",
    "ga",
    "it",
    "ja",
    "kn",
    "kk",
    "ko",
    "lv",
    "lt",
    "mk",
    "ms",
    "ml",
    "mt",
    "mr",
    "mn",
    "no",
    "ps",
    "pl",
    "pt",
    "pt-PT",
    "pa",
    "ro",
    "ru",
    "sr",
    "si",
    "sk",
    "sl",
    "so",
    "es",
    "es-MX",
    "sw",
    "sv",
    "ta",
    "te",
    "th",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "cy",
]

spacy_supported_lang = [
    "en",
    "fr",
    "de",
    "nl",
    "ja",
    "zh",
    "ca",
    "hr",
    "da",
    "fi",
    "el",
    "it",
    "ko",
    "lt",
    "mk",
    "xx",
    "nb",
    "pl",
    "pt",
    "ro",
    "ru",
    "es",
    "sv",
    "uk",
]


# ============================== Text Normalization ===================================

re_remove_duplicate_spaces = r"(\s\s+)"
repl_remove_duplicate_spaces = " "

re_remove_leading_space = r"(^\s+)"
re_remove_trailing_space = r"(\s+$)"
re_remove_trailing_symbols = r"([\s>_]+$)"
repl_remove_space = ""

re_remove_trailing_underscores = r"([_]+$)"
repl_remove_trailing_underscores = r""

re_remove_duplicate_punct = r"([.,!?:;])\1+"
repl_remove_duplicate_punct = r"\1"

re_add_space_after_punct = r"(?<=[.,!?:;])(?=[^\s])"
repl_add_space_after_punct = r" "

re_remove_space_before_punct = r"(?<=[^\s])[\s]+(?=[.,!?:;])"
repl_remove_space_before_punct = r""


re_add_dot_end_of_message = r"(?<=[^.!?])$"
repl_add_dot_end_of_message = "."


re_remove_punct_before_dot = r"[.!?,:;_\s]+(?=[.!?])"
repl_remove_punct_before_dot = ""

re_sentence_boundary = r"((?<=[a-z0-9][.?!])|(?<=[a-z0-9][.?!]\"))(\s|\r\n)(?=\"?[A-Z])"

re_special_punct = """[\<\>\(\)\'\"\/@#\$%\^&\*~:\+]"""
re_normal_punct = """[?!\.;]"""
re_normal_punct2 = """(?<=[w])[\s]*:"""

re_emoji_pattern = re.compile(
    pattern="["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)

re_remove_space_btw_punct = r"(?<=[\,\.])\s+(?=[\,\.])"

re_remove_within_parentheses_brackets = r"""(<.*>|\[.*\]|\(.*\))"""

re_remove_html_linebreak = r"""(&nbsp;|&lt;|&gt;|&amp;)"""

re_remove_hyphen_sep_messages = r"""[\-=]{2,}[\d\w\-\*\$\s:]+[\-=]{2,}"""


# ============================== Name Entity Recognition =========================================

re_http_url = re.compile(
    r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))""",
    re.M | re.I | re.S,
)
# r'''\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\"\\\/.,<>?\xab\xbb\u201c\u201d\u2018\u2019]))''' #r'(https?:[\s]*\/\/(www\.)?[\s]*[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))'
# re_emailaddr = r'([0-9a-zA-Z]([-.\w]*[0-9a-zA-Z_+])*@([0-9a-zA-Z][-\w]*[0-9a-zA-Z]\.)+[a-zA-Z]{2,9})'
re_emailaddr = r"[\w-]+@([\w-]+\.)+[\w-]+"
re_amazon_emailladdr = r"(globalstoremerchant[a-z0-9\+\-]*|gs-prod[a-z0-9\+\-]*|no-reply[a-z0-9\+\-]+|ooc-customer-service[a-z0-9\+\-]*|retail-applecare[a-z0-9\+\-]*|gc-orders[a-z0-9\+\-]*|ooc-customer-service[a-z0-9\+\-]*|installation-services[a-z0-9\+\-]*|seller.service[a-z0-9\+\-]*|merch.service[a-z0-9\+\-]*)@amazon.[\w.]+"
re_amazon_ship_confirm = r"fba-customer-ship-confirm@amazon.[\w.]+"
re_asin = r"([0-9]{9}[0-9X]|[A-Z][A-Z0-9]{9})"
re_orderid = r"\b([0-9]{3}\-[0-9]{7}\-[0-9]{7})\b"
re_phoneus = r"""(((\((\+){0,1}(\d){1,2}\))[\s]*|(\d){1,2}-)*([2-9]\d{2}-|\([2-9]\d{2}\) )\d{3}-\d{4})"""
# r'((((\+){,1}(\d){,2})|(\d){,2}-)([2-9]\d{2}-|\([2-9]\d{2}\) )\d{3}-\d{4})\b'
re_dollar_amount = r"(\$[\d,]*(\.\d{2})?(\shundred\w*|\sthousand\w*|\smillion\w*|\sbillion\w*|\strillion\w*)?|([\d]+(\shundred\w*|\sthousand\w*|\smillion\w*|\sbillion\w*|\strillion\w*)?(\sdollar\w*|\scent\w*)(\sand\s[\d]*\scent\w*)?)|(one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|fifty|sixty|seventy|eighty|ninety)+(\shundred\w*|\sthousand\w*|\smillion\w*|\sbillion\w*|\strillion\w*)*(\sdollars))"
re_percentage = r"([0-9]+(\.[0-9]+?)*?\%)"
re_time = r"((([0]?[1-9]|1[0-2])(:|\.)[0-5][0-9]((:|\.)[0-5][0-9])?( )?(AM|am|aM|Am|PM|pm|pM|Pm))|(([0]?[0-9]|1[0-9]|2[0-3])(:|\.)[0-5][0-9]((:|\.)[0-5][0-9])?))"
re_decimal_bps = r"\d*\.?\d+\ bps"
re_decimal = r"\d*\.?\d+"


re_matchUPS1 = (
    "(1Z ?[0-9A-Z]{3} ?[0-9A-Z]{3} ?[0-9A-Z]{2} ?[0-9A-Z]{4} ?[0-9A-Z]{3} ?[0-9A-Z])"
)
re_matchUPS2 = "([kKJj]{1}[0-9]{10})"
re_matchUSPS0 = "(\b\d{30}\b)|(\b\d{20}\b)"
re_matchUSPS1 = "(\bE\D{1}\d{9}\D{2}|9\d{15,21}\b)"
re_matchUSPS2 = "(\b91[0-9]+\b)"
re_matchUSPS3 = "(\b[A-Za-z]{2}[0-9]+US\b)"
re_matchUSPS4 = "(((\d{4})(\s?\d{4}){4}\s?\d{2})|((\d{2})(\s?\d{3}){2}\s?\d{2})|((\D{2})(\s?\d{3}){3}\s?\D{2}))"  #'(\b\d{30}\b)|(\b91\d+\b)|(\b\d{20}\b)|(\b\d{26}\b)|E\D{1}\d{9}\D{2}|9\d{15,21}|91[0-9]+|[A-Za-z]{2}[0-9]+US'
re_matchFedex1 = "(\b96\d{20}\b)|(\b\d{15}\b)|(\b\d{12}\b)"
re_matchFedex2 = "\b((98\d\d\d\d\d?\d\d\d\d|98\d\d) ?\d\d\d\d ?\d\d\d\d( ?\d\d\d)?)\b"
re_matchFedex3 = "[0-9]{15}"
re_match_shiptrackId_all_us = (
    "("
    + "|".join(
        [
            re_matchUSPS0,
            re_matchUPS1,
            re_matchUPS2,
            re_matchUSPS1,
            re_matchUSPS2,
            re_matchUSPS3,
            re_matchUSPS4,
            re_matchFedex1,
            re_matchFedex2,
            re_matchFedex3,
        ]
    )
    + ")"
)

re_uszipcode = r"(\b\d{5}(?:-\d{4})?\b)"


re_state = r"(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New[ ]Hampshire|New[ ]Jersey|New[ ]Mexico|New[ ]York|North[ ]Carolina|North[ ]Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode[ ]Island|South[ ]Carolina|South[ ]Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West[ ]Virginia|Wisconsin|Wyoming)"
re_state_abbreviations = r"(AL|AK|AS|AZ|AR|CA|CO|CT|DE|DC|FM|FL|GA|GU|HI|ID|IL|IN|IA|KS|KY|LA|ME|MH|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|MP|OH|OK|OR|PW|PA|PR|RI|SC|SD|TN|TX|UT|VT|VI|VA|WA|WV|WI|WY)"
re_state_or_abbrev = r"(?-i:A[LKSZRAEP]|C[AOT]|D[EC]|F[LM]|G[AU]|HI|I[ADLN]|K[SY]|LA|M[ADEHINOPST]|N[CDEHJMVY]|O[HKR]|P[ARW]|RI|S[CD]|T[NX]|UT|V[AIT]|W[AIVY])"
re_city = r"((?:[A-Z][a-z.-]+[ ]?)+)"
re_street = r"(\d+[ ](?:[A-Za-z0-9.-]+[ ]?)+(?i)(?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr|Rd|Blvd|Ln|St)\.?)"
re_address_1 = "{city_pattern},[ ](?:{state_pattern}|(?i){abbrev_state_pattern})[ ]{zip_pattern}".format(
    city_pattern=re_city,
    state_pattern=re_state,
    abbrev_state_pattern=re_state_abbreviations,
    zip_pattern=re_uszipcode,
)
re_address_state_zip = "[ ](?i)(?:{state_or_abbrev_pattern})[ ]{zip_pattern}".format(
    state_or_abbrev_pattern=re_state_or_abbrev, zip_pattern=re_uszipcode
)

re_day_backslash_month_backslash_year_date = (
    r"""([0-3]?[0-9]/[0-3]?[0-9]/(?:[0-9]{2})?[0-9]{2})"""
)
re_weekday_comma_month_day_comma_year_date = r"((Mo(n(day)?)?|Tu(e(sday)?)?|We(d(nesday)?)?|Th(u(rsday)?)?|Fr(i(day)?)?|Sa(t(urday)?)?|Su(n(day)?)?),\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4})"
re_weekday = r"(Mo(n(day)?)?|Tu(e(sday)?)?|We(d(nesday)?)?|Th(u(rsday)?)?|Fr(i(day)?)?|Sa(t(urday)?)?|Su(n(day)?)?)"
re_month_day = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(1(?:st)?|2(?:nd)?|3(?:rd)?|\d{1,2}(?:th)?)"


substring_template_url = "URL"
substring_template_email = "EMAIL"
substring_template_asin = "ASIN"
substring_template_orderid = "ORDER_ID"
substring_template_dollar_amount = "DOLLAR_AMOUNT"
substring_template_percentage = "PERCENTAGE"
substring_template_time = "TIME"
substring_template_month_day = "MON_DAY"
substring_template_weekday = "WEEKDAY"
substring_template_date = "DATE"
substring_template_decimal = "DECIMAL"
substring_template_uszipcode = "ZIP"
substring_template_state = "STATE"
substring_template_city = "CITY"
substring_template_street = "STREET"
substring_template_state_zip = "STATE_ZIP"


# ==========================   Stopwords =====================


Noninformative_tokens = {
    "dear",
    "hi",
    "hello",
    "mr",
    "mrs.",
    "customer",
    "pleasure",
    "sorry",
    "however",
    "appreciation",
    "hear",
    "inconvenience",
    "hope",
    "thank",
    "thanks",
    "appreciated",
    "sincerely",
    "best",
    "wishes",
    "yours",
    "apology",
    "consideration",
    "regards",
    "truely",
    "person",
    "product",
    "cardinal",
}

# ==========================  End of Letter Keywords =====================
Farewell_tokens_en = [
    "Best",
    "Cordially yours",
    "Best regards",
    "Best wishes",
    "Fond regards",
    "In appreciation",
    "In sympathy",
    "Kind regards",
    "Kind thanks",
    "Many thanks",
    "Regards",
    "Respectfully",
    "Respectfully yours",
    "Sincerely",
    "Sincerely yours",
    "Thanks",
    "Thank you",
    "Thank you for your",
    "Warm regards",
    "Warm wishes",
    "With appreciation",
    "With deepest sympathy",
    "With gratitude",
    "Your help is greatly appreciated",
    "Yours cordially",
    "Yours sincerely",
    "Yours cordially",
    "Yours truly",
    "Yours faithfully",
    "have a wonderful day",
    "please feel free to",
    "your reply",
]

Farewell_tokens_fr = [
    "Cordialement",
    "Cordialement à vous",
    "Chaleureusement",
    "Bien amicalement",
    "Amitiés",
    "Bien à vous",
    "Meilleures salutations",
    "Je vous envoie mes amicales pensées",
    "Recevez, je vous prie, mes meilleures amitiés",
    "À bientôt",
    "affectueuses pensées",
    "Bien à vous, Bien à toi",
    "Affectueusement",
]

Farewell_tokens_de = [
    "Freundliche Grüße",
    "Mit freundlichen Grüßen",
    "Hochachtungsvoll",
    "Mit besten Grüßen",
    "Mit freundlichen Empfehlungen",
    "Beste Wünsche",
    "Dein",
    "Deine",
    "In Liebe",
    "Viele Grüße",
    "Herzlichst",
    "Ich danke Ihnen für Ihre Aufmerksamkeit und verbleibe mit freundlichen Grüßen",
    "Ich freue mich auf Ihre Antwort",
]

Farewell_tokens_es = [
    "Atenciosamente",
    "Muy atentamente",
    "Muy cordialmente",
    "Un saludo",
    "Un saludo cordial",
    "Saludos cordiales",
    "Saludos",
]

Farewell_tokens_pt = [
    "Atentamente",
    "Saudações cordiais",
    "Com os melhores cumprimentos",
    "Grato",
    "Agradeço desde já",
    "Aguardo sua resposta",
    "Respeitosamente",
]

Farewell_tokens_it = [
    "Cordialmente",
    "Cordiali saluti",
    "Cordiali Saluti",
    "Un Cordiale Saluto",
    "Grazie e Cordiali Saluti",
    "Le porgo i miei distinti saluti",
    "La prego di voler gradire i miei più distinti saluti",
    "Distinti Saluti",
    "Ringraziando anticipatamente per la sempre cortese collaborazione porgo cordiali saluti",
]

Farewell_tokens_ja = [
    "をよろしくお願いいたします",
]

Farewell_tokens_tr = [
    "Saygılarımızla",
]

Farewell_tokens_nl = [
    "Met vriendelijke groet",
]

Farewell_tokens_sv = ["Med vänliga hälsningar"]


Farewell_tokens = (
    Farewell_tokens_en
    + Farewell_tokens_fr
    + Farewell_tokens_de
    + Farewell_tokens_es
    + Farewell_tokens_pt
    + Farewell_tokens_it
    + Farewell_tokens_ja
    + Farewell_tokens_tr
    + Farewell_tokens_nl
    + Farewell_tokens_sv
)

# ==========================  Beginning of Letter Keywords =====================

Greeting_tokens_en = [
    "Dear",
    "Hi",
    "Hello",
    "Mr",
    "Mrs",
    "Greetings",
    "To Whom It May Concern",
    "Hey",
]

Greeting_tokens_fr = [
    "Bonjou",
    "Chers",
    "Chère",
    "Cher",
    "Chères",
    "Mon cher",
    "Ma très chère",
    "Monsieur",
    "Madame",
    "Mademoiselle",
    "Messieurs",
]


Greeting_tokens_de = ["Guten Tag", "Hallo", "Sehr geehrter", "Liebe", "Lieber"]

Greeting_tokens_nl = [
    "Hallo",
    "Beste",
]

Greeting_tokens_pt = ["Prezados", "Olá"]

Greeting_tokens_it = [
    "Gentili",
    "Gentile Venditore",
]

Greeting_tokens_es = [
    "Estimados",
    "Hola",
]

Greeting_tokens_tr = [
    "Sayın",
    "Merhaba",
]

Greeting_tokens_sv = [
    "Kära",
]


Greeting_tokens = (
    Greeting_tokens_en
    + Greeting_tokens_fr
    + Greeting_tokens_de
    + Greeting_tokens_nl
    + Greeting_tokens_pt
    + Greeting_tokens_it
    + Greeting_tokens_es
    + Greeting_tokens_tr
    + Greeting_tokens_sv
)

# ======================= Template for CS Central Emails ====================================

RE_AMAZON_CS_HEADER_EN = re.compile(
    r"""Order\s*(#|Number|number):.*(?=Details:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_FR = re.compile(
    r"""Numéro de commande:.*(?=Détails:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_DE = re.compile(
    r"""Bestellnummer\s*:.*(?=Weitere Angaben:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_PT = re.compile(
    r"""Número do pedido\s*:.*(?=Detalhes:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_ES = re.compile(
    r"""Número de pedido\s*:.*(?=Detalles:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_MX = re.compile(
    r"""Pedido\s*#:.*(?=Detalles:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_IT = re.compile(
    r"""Ordine\s*(#|Numero):.*(?=Dettagli:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_JA = re.compile(r"""注文番号\s*:.*(?=詳細:)""", re.I | re.S | re.M)
RE_AMAZON_CS_HEADER_TR = re.compile(
    r"""Sipariş numarası\s*:.*(?=Ayrıntılar:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_NL = re.compile(
    r"""Bestelnummer\s*:.*(?=Details:)""", re.I | re.S | re.M
)
RE_AMAZON_CS_HEADER_PL = re.compile(
    r"""Numer zamówienia\s*:.*(?=Szczegóły:)""", re.I | re.S | re.M
)


RE_AMAZON_CS_SIGNATURE_EN = re.compile(
    r"""To respond to this customer, please reply to this e-mail or visit your seller account at the following link:.* Sincerely.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_FR = re.compile(
    r"""Nous vous remercions de répondre directement à ce client en répondant à cet e-mail ou en vous rendant dans votre compte vendeur:.* Cordialement.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_DE = re.compile(
    r"""Um Ihren Käufer zu kontaktieren, antworten Sie bitte auf diese E-Mail oder besuchen Sie Ihr Verkäuferkonto über folgenden Link:.* Freundliche Grüße.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_PT = re.compile(
    r"""Para entrar em contato com este cliente, acesse sua conta de vendedor pelo link:.* Atenciosamente.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_ES = re.compile(
    r"""Para responder al cliente,  por favor responde a este email o bien visita tu cuenta de vendedor en el siguiente enlace:.* Atentamente.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_MX = re.compile(
    r"""Para enviarle una respuesta al cliente, por favor responde a este correo o visita tu cuenta de vendedor en el siguiente enlace:.* Cordialmente.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_IT = re.compile(
    r"""Per rispondere a questo cliente, risponda a quest'e-mail oppure utilizzi il suo account venditore cliccando sul seguente link:.* Cordiali saluti.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_JA = re.compile(
    r"""購入者様へのご連絡は、直接こちらのメールにご返信いただくか、以下セラーアカウントからご返信ください。.* 今後ともAmazon.co.jp をよろしくお願いいたします.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_TR = re.compile(
    r"""Bu müşteriye cevap vermek için lütfen bu e-postayı yanıtlayın veya aşağıdaki bağlantıdan satıcı hesabınızı ziyaret edin:.* Saygılarımızla.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_NL = re.compile(
    r"""Je kunt de klant antwoorden door op deze e-mail te reageren. Je kunt ook reageren via je verkopersaccount:.* Met vriendelijke groet.*""",
    re.I | re.S | re.M,
)
RE_AMAZON_CS_SIGNATURE_PL = re.compile(
    r"""Aby skontaktować się z tym klientem, proszę odpowiedzieć na ten e-mail lub odwiedzić swoje konto sprzedawcy pod następującym adresem:.* Z poważaniem.*""",
    re.I | re.S | re.M,
)


# =====================   Template for Merchant Return Service Email ===================================
# Return Autorization
RE_AMAZON_MRS_HEADER_EN = re.compile(
    r"""(^Dear |^).* and .*(:|,)[\s]+This (email|note) is being sent to you by .* to notify and confirm that a return authorization has been requested for the item(\(s\))* listed below.*(?=Return reason:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_FR = re.compile(
    r"""Chers .* et .*(:|,)[\s]+Cet e-mail vous est envoyé par .* afin de vous informer qu’une demande d’autorisation de retour a été soumise pour le ou les articles listés ci-dessous.*(?=Raison du retour:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_FR_3 = re.compile(
    r"""Chers .* et .*(:|,)[\s]+Cet e-mail vous est envoyé par .* afin de vous informer et de vous confirmer qu’une demande d’autorisation de retour a été soumise pour le ou les articles listés ci-dessous.*(?=Raison du retour.*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_PT = re.compile(
    r"""Prezados .* e .*(:|,)[\s]+Este e-mail lhe foi enviado pela .* para confirmar que um pedido de autorização para devolução foi emitido para o\(s\) item(\(ns\))* abaixo.*(?=Razão para a devolução:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_DE = re.compile(
    r"""Guten Tag,[\s]+Amazon.* sendet Ihnen diese E-Mail zur Bestätigung, dass ein Rücksendeantrag für folgende\(n\) Artikel gestellt wurde.*(?=Rücksendegrund:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_MX = re.compile(
    r"""Estimados .* y .*(:|,)[\s]+Amazon os envía este correo electrónico para dejar constancia de la solicitud de autorización presentada para efectuar la devolución de los artículos indicados más adelante.*(?=Motivo de la devolución:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_IT = re.compile(
    r"""Gentili .* e .*(:|,)[\s]+questa e-mail vi viene inviata da Amazon quale notifica e conferma della richiesta di autorizzazione ad effettuare il reso degli articoli elencati di seguito.*(?=Motivo del reso:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_TR = re.compile(
    r"""Sayın .* ve .*(:|,)[\s]+Bu not, aşağıdaki ürünler için bir iade onayının istendiğine dair sizi bilgilendirmek ve bunu teyit etmek üzere Amazon tarafından gönderilmiştir.*(?=İade nedeni:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_NL = re.compile(
    r"""Beste .* en .*(:|,)[\s]+Amazon stuurt je deze e-mail om te bevestigen dat er autorisatie voor een retourzending is aangevraagd voor de onderstaande items.*(?=Reden voor retourneren:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_SW = re.compile(
    r"""Kära .* och .*(:|,)[\s]+Detta e-postmeddelande skickas till dig av Amazon för att meddela och bekräfta att ett returtillstånd har begärts för följande artiklar.*(?=Anledning till retur:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_JA = re.compile(
    r"""このメールは、出品者様および購入者様の両方にお送りしております。.*Amazon.co.jpよりお知らせいたします。以下の商品について返品が依頼されました。.*(?=返品理由:\s*)""",
    re.M | re.S,
)

# Return Cancellation
RE_AMAZON_MRS_HEADER_EN_2 = re.compile(
    r"""Dear .* and .*, This note has been sent by Amazon to inform you and confirm that .* has cancelled the refund request for the following order.*(?=Return reason:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_FR_2 = re.compile(
    r"""Chers .* et .*, Cet e-mail vous est envoyé par Amazon pour vous faire part du fait que .* a annulé sa demande de retour pour la commande suivante.*(?=Raison du retour:\s*)""",
    re.M | re.S,
)
RE_AMAZON_MRS_HEADER_TR_2 = re.compile(
    r"""Sayın .* ve .*,[\s]+Bu not, .* adlı alıcının aşağıdaki siparişle ilgili iade talebini iptal ettiğine dair sizi bilgilendirmek ve bunu teyit etmek üzere Amazon tarafından gönderilmiştir.*(?=İade nedeni:\s*)""",
    re.M | re.S,
)


RE_AMAZON_MRS_SIGNATURE_EN = re.compile(
    r"""Request received:.* Sincerely,\s+Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_EN_2 = re.compile(r"""Request cancelled:.*""", re.M | re.S)
RE_AMAZON_MRS_SIGNATURE_FR = re.compile(
    r"""Demande reçue:.* Cordialement,\s+Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_FR_2 = re.compile(
    r"""Requête reçue le\s*:.* Cordialement,\s+Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_FR_3 = re.compile(r"""Demande annulée:.*""", re.M | re.S)
RE_AMAZON_MRS_SIGNATURE_PT = re.compile(
    r"""Pedido recebido:.* Atenciosamente,\s+Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_DE = re.compile(
    r"""Eingang des Rücksendeantrags:.* Freundliche Grüße.*Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_ES = re.compile(
    r"""Solicitud recibida el.* Atentamente,\s+Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_IT = re.compile(
    r"""Richiesta ricevuta il:.* Distinti Saluti,\s+Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_TR = re.compile(
    r"""Alınan talep.* Saygılarımızla.*Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_TR_2 = re.compile(r"""İptal edilen talep:.*""", re.M | re.S)
RE_AMAZON_MRS_SIGNATURE_NL = re.compile(
    """Verzoek ontvangen:.* Met vriendelijke groet,\s+Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_SW = re.compile(
    r"""Begäran mottagen:.* Med vänliga hälsningar,\s+Amazon.*""", re.M | re.S
)
RE_AMAZON_MRS_SIGNATURE_JA = re.compile(
    r"""返品リクエストの送信日:.*今後ともAmazon.co.jpをよろしくお願いいたします。.*""",
    re.M | re.S,
)
RE_AMAZON_MRS_SIGNATURE_PL = re.compile(
    r"""Żądanie otrzymano:.* Pozdrawiamy,\s+Amazon.*""", re.M | re.S
)

# ======== Template for High Quality Review Solicitation: 3rd Party API Emails =============================

RE_AMAZON_3P_EN = re.compile(
    r"""Message from seller [\w\d\-\s]+:[\s]+(?=[\w])""", re.S | re.M
)
RE_AMAZON_3P_FR = re.compile(
    r"""Message du vendeur [\w\d\-\s]+:[\s]+(?=[\w])""", re.S | re.M
)
RE_AMAZON_3P_DE = re.compile(
    r"""Nachricht vom Verkäufer [\w\d\-\s]+:[\s]+(?=[\w])""", re.S | re.M
)
RE_AMAZON_3P_PT = re.compile(
    r"""Mensagem do vendedor [\w\d\-\s]+:[\s]+(?=[\w])""", re.S | re.M
)
RE_AMAZON_3P_ES = re.compile(
    r"""Mensaje del vendedor [\w\d\-\s]+:[\s]+(?=[\w])""", re.S | re.M
)
RE_AMAZON_3P_IT = re.compile(
    r"""Messaggio dal venditore [\w\d\-\s]+:[\s]+(?=[\w])""", re.S | re.M
)
RE_AMAZON_3P_NL = re.compile(
    r"""Bericht van verkoper [\w\d\-\s]+:[\s]+(?=[\w])""", re.S | re.M
)
RE_AMAZON_3P_PL = re.compile(
    r"""Wiadomość od sprzedawcy [\w\d\-\s]+:[\s]+(?=[\w])""", re.S | re.M
)
RE_AMAZON_3P_JA = re.compile(r"""出品者.*からのメッセージ：\s+""", re.S | re.M)


# ===================== Beginning of Reply Quotatioins ======================

Quotation_pattern_str_en = [
    "(Did this solve your problem)",
    "(Sent from my [\w]+)",
    "(Get Outlook for )",
    "([-]{2,}[ ]*Original Message[ ]*[-]{2,})",
    "([-]{2,}[ ]*Original Email[ ]*[-]{2,})",
    "([-]{2,}[ ]*Message[ ]*[-]{2,})",
    "(On .*(?=wrote:))",
    "([_]+\s+From: )",
    "(From: )",
    "(Sent: )",
    "([-]+[ ]*Forwarded message[ ]*[-]+)",
]

Quotation_pattern_str_de = [
    "(Von meinem [\w]+ gesendet)" "([-]{2,}[ ]*Original-Nachricht[ ]*[-]{2,})",
    "([-]{2,}[ ]*Ursprüngliche Daten[ ]*[-]{2,})",
    "(Am .*(?=schrieb\s*.*:))",
    "([-]{2,}\s*Originalnachricht)",
]

Quotation_pattern_str_fr = [
    "(Inviato da .*(?=\s))",
    "([-]{2,}[ ]*Message original en langue anglaise)",
    "([-]{2,}[ ]*Finalizar mensaje[ ]*[-]{2,})",
    "(Le .*(?=wrote\s*:))",
    "(Le .*(?=a écrit\s*:))",
]
Quotation_pattern_str_es = [
    "(Enviado desde mi [\w]+)",
    "(El .*(?=escribió\s*:))",
    "(<[dirección de correo electrónico eliminada]>.*(?=escribió\s*:))",
]

Quotation_pattern_str_it = [
    "([-]{2,}[ ]*Messaggio originale[ ]*[-]{2,})",
    "(Il .*(?=ha scritto:))",
]

Quotation_pattern_str_pt = [
    "(Obter o Outlook para .*)",
    "([-]{2,}[ ]*Iniciar mensagem[ ]*[-]{2,})",
    "(Em .*(?=escreveu))",
    "([-]{2,}\s*Atualizado por:\s+)",
]

Quotation_pattern_str_tr = [
    "(Gönderen:.* Gönderildi)",
    "([^.!?]+\s+cihazımdan gönderildi)",
    "([-]{2,}[ ]*Orijinal mesaj[ ]*[-]{2,})",
    "((?<=[\s.?!]) .* tarihinde .*(?=şunu yazdı:))",
    #'(.* tarihinde .*(?=şunu yazdı:))',
    "([-]{2,}\s*Gönderen:\s+)",
]

Quotation_pattern_str_nl = [
    "([-]{2,}[ ]*Oorspronkelijk Bericht[ ]*[-]{2,})",
    "(Op .*(?=schreef .*:))",
]

Quotation_pattern_str_pl = [
    "(Wysłane z .*'a)",
    "([-]{2,}[ ]*Ursprüngliche Nachricht[ ]*[-]{2,})",
    "(Wiadomość napisana przez .* (?=<\[adres e-mail usunięty\]> w dniu .*:))",
    "(<.*\[adres e-mail usunięty\]>\s+(?=napisał\(a\):))",
]

Quotation_pattern_str_sv = ["(skrev .*<\[E-postadress borttagen\]>:)", "([_]+\s+Från:)"]


Quotation_pattern_str = (
    Quotation_pattern_str_en
    + Quotation_pattern_str_de
    + Quotation_pattern_str_fr
    + Quotation_pattern_str_es
    + Quotation_pattern_str_it
    + Quotation_pattern_str_pt
    + Quotation_pattern_str_tr
    + Quotation_pattern_str_nl
    + Quotation_pattern_str_pl
    + Quotation_pattern_str_sv
    + ["(\s+[>]{1,})", "(\s+[_]{2,})"]
)

RE_BEGIN_QUOTATION = re.compile(
    r"""(%s)""" % ("|".join(Quotation_pattern_str)), re.S | re.M
)


re_remove_removed_part = r"\[.*(removed|entfernt|supprimée|eliminado|eliminato|eliminada|removido|usunięty|verwijderd|borttagen)\]"
RE_REMOVED_PART = re.compile(re_remove_removed_part, re.S | re.M)
