"""
Drug names used on the synthetic labels, in English and Arabic.
Each entry is one drug: (English name, Arabic name).
Add or remove drugs here and both scripts pick them up.
"""

DRUGS = [
    ("Normal Saline 0.9%",   "محلول ملحي 0.9%"),
    ("Dextrose 5%",          "جلوكوز 5%"),
    ("Ringer Lactate",       "رينجر لاكتات"),
    ("Paracetamol",          "باراسيتامول"),
    ("Ceftriaxone",          "سيفترياكسون"),
    ("Metronidazole",        "ميترونيدازول"),
    ("Potassium Chloride",   "كلوريد البوتاسيوم"),
    ("Amoxicillin",          "أموكسيسيلين"),
]

VOLUMES = ["100 mL", "250 mL", "500 mL", "1000 mL"]


def other_names(name):
    """Given a drug name in either language, return both names for that drug.
    If the name is not in the list, return it unchanged."""
    for en, ar in DRUGS:
        if name.strip().lower() in (en.lower(), ar):
            return [en, ar]
    return [name]
