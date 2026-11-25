"""
Drug names, in English and Arabic.

DRUGS     : (English name, Arabic name) - printed on the synthetic test labels.
SYNONYMS  : other names for the same drug, as written on real labels
            (e.g. a "Normal Saline" bag is often labelled "Sodium Chloride").
A prescription in ANY of these names matches a label showing any other name of the same drug.
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

SYNONYMS = {
    "Normal Saline 0.9%": ["Sodium Chloride 0.9%", "كلوريد الصوديوم 0.9%", "NaCl 0.9%"],
    "Dextrose 5%":        ["Glucose 5%", "دكستروز 5%", "D5W"],
    "Ringer Lactate":     ["Lactated Ringer", "Hartmann", "رينجر"],
}

VOLUMES = ["100 mL", "250 mL", "500 mL", "1000 mL"]


def other_names(name):
    """All names of the drug called `name` (English, Arabic and synonyms).
    If the name is unknown, return it unchanged."""
    key = name.strip().lower()
    for en, ar in DRUGS:
        names = [en, ar] + SYNONYMS.get(en, [])
        if key in (n.lower() for n in names):
            return names
    return [name]
