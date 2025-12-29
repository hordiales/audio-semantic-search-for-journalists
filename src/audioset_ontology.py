import logging

# AudioSet ontology - principales clases de sonidos
AUDIOSET_CLASSES = {
    # Voz humana
    "speech": {"id": "/m/09x0r", "name": "Speech", "description": "Voz humana hablando"},
    "male_speech": {"id": "/m/05zppz", "name": "Male speech, man speaking", "description": "Voz masculina"},
    "female_speech": {"id": "/m/02zsn", "name": "Female speech, woman speaking", "description": "Voz femenina"},
    "child_speech": {"id": "/m/0ytgt", "name": "Child speech, kid speaking", "description": "Voz infantil"},
    "conversation": {"id": "/m/01h8n0", "name": "Conversation", "description": "Conversación"},
    "narration": {"id": "/m/02rhddq", "name": "Narration, monologue", "description": "Narración"},
    "babbling": {"id": "/m/03qtq", "name": "Babbling", "description": "Balbuceo"},

    # Sonidos humanos
    "laughter": {"id": "/m/01j3sz", "name": "Laughter", "description": "Risas"},
    "applause": {"id": "/m/0b_fwt", "name": "Applause", "description": "Aplausos"},
    "clapping": {"id": "/m/028v0c", "name": "Clapping", "description": "Palmadas"},
    "cheering": {"id": "/m/03qjg", "name": "Cheering", "description": "Vítores"},
    "booing": {"id": "/m/02p01q", "name": "Booing", "description": "Abucheos"},
    "crying": {"id": "/m/07r4k8j", "name": "Crying, sobbing", "description": "Llanto"},
    "whistling": {"id": "/m/02bk07", "name": "Whistling", "description": "Silbidos"},
    "shouting": {"id": "/m/07pggtn", "name": "Shouting, yelling", "description": "Gritos"},
    "whispering": {"id": "/m/0dl9sf8", "name": "Whispering", "description": "Susurros"},

    # Multitudes
    "crowd": {"id": "/m/01bjv", "name": "Crowd", "description": "Multitud"},
    "hubbub": {"id": "/m/03qtwd", "name": "Hubbub, speech noise", "description": "Murmullo de voces"},
    "chatter": {"id": "/m/07pws3f", "name": "Chatter", "description": "Charla"},
    "chanting": {"id": "/m/02rtxlg", "name": "Chanting", "description": "Cánticos"},

    # Música
    "music": {"id": "/m/04rlf", "name": "Music", "description": "Música"},
    "singing": {"id": "/m/015lz1", "name": "Singing", "description": "Canto"},
    "instrumental_music": {"id": "/m/02mscn", "name": "Instrumental music", "description": "Música instrumental"},
    "piano": {"id": "/m/05r5c", "name": "Piano", "description": "Piano"},
    "guitar": {"id": "/m/0342h", "name": "Guitar", "description": "Guitarra"},
    "drum": {"id": "/m/026t6", "name": "Drum", "description": "Tambor"},
    "orchestra": {"id": "/m/05k0_", "name": "Orchestra", "description": "Orquesta"},

    # Vehículos
    "vehicle": {"id": "/m/07yv9", "name": "Vehicle", "description": "Vehículo"},
    "car": {"id": "/m/0k4j", "name": "Car", "description": "Automóvil"},
    "truck": {"id": "/m/07r04", "name": "Truck", "description": "Camión"},
    "motorcycle": {"id": "/m/04_sv", "name": "Motorcycle", "description": "Motocicleta"},
    "bus": {"id": "/m/01bjv", "name": "Bus", "description": "Autobús"},
    "train": {"id": "/m/07jdr", "name": "Train", "description": "Tren"},
    "aircraft": {"id": "/m/0cmf2", "name": "Aircraft", "description": "Aeronave"},
    "helicopter": {"id": "/m/03j1ly", "name": "Helicopter", "description": "Helicóptero"},
    "engine": {"id": "/m/02mk9", "name": "Engine", "description": "Motor"},

    # Sirenas y alarmas
    "siren": {"id": "/m/01yrx", "name": "Siren", "description": "Sirena"},
    "alarm": {"id": "/m/0b_fwt", "name": "Alarm", "description": "Alarma"},
    "fire_alarm": {"id": "/m/016622", "name": "Fire alarm", "description": "Alarma de incendio"},
    "smoke_detector": {"id": "/m/03kmc9", "name": "Smoke detector", "description": "Detector de humo"},
    "buzzer": {"id": "/m/07pkxks", "name": "Buzzer", "description": "Zumbador"},

    # Naturaleza
    "wind": {"id": "/m/09t49", "name": "Wind", "description": "Viento"},
    "rain": {"id": "/m/05r5wn", "name": "Rain", "description": "Lluvia"},
    "thunder": {"id": "/m/0ngt1", "name": "Thunder", "description": "Trueno"},
    "water": {"id": "/m/0838f", "name": "Water", "description": "Agua"},
    "ocean": {"id": "/m/05kq4", "name": "Ocean", "description": "Océano"},
    "bird": {"id": "/m/015p6", "name": "Bird", "description": "Pájaro"},
    "dog": {"id": "/m/0bt9lr", "name": "Dog", "description": "Perro"},
    "cat": {"id": "/m/01yrx", "name": "Cat", "description": "Gato"},

    # Tecnología
    "telephone": {"id": "/m/07gql", "name": "Telephone", "description": "Teléfono"},
    "phone_ringing": {"id": "/m/0c2wf", "name": "Phone ringing", "description": "Teléfono sonando"},
    "computer": {"id": "/m/01c648", "name": "Computer", "description": "Computadora"},
    "keyboard": {"id": "/m/01m2v", "name": "Keyboard", "description": "Teclado"},
    "mouse": {"id": "/m/04brg2", "name": "Mouse", "description": "Ratón"},
    "printer": {"id": "/m/01whjb", "name": "Printer", "description": "Impresora"},
    "fax": {"id": "/m/02x8cch", "name": "Fax machine", "description": "Fax"},
    "microphone": {"id": "/m/04szw", "name": "Microphone", "description": "Micrófono"},
    "loudspeaker": {"id": "/m/0d31p", "name": "Loudspeaker", "description": "Altavoz"},
    "television": {"id": "/m/07c52", "name": "Television", "description": "Televisión"},
    "radio": {"id": "/m/06bz3", "name": "Radio", "description": "Radio"},

    # Construcción y trabajo
    "construction": {"id": "/m/03j1ly", "name": "Construction", "description": "Construcción"},
    "drilling": {"id": "/m/02p01q", "name": "Drilling", "description": "Taladrado"},
    "hammer": {"id": "/m/07gql", "name": "Hammer", "description": "Martillo"},
    "saw": {"id": "/m/05r5wn", "name": "Saw", "description": "Sierra"},
    "machinery": {"id": "/m/0838f", "name": "Machinery", "description": "Maquinaria"},

    # Sonidos de ambiente
    "silence": {"id": "/m/028v0c", "name": "Silence", "description": "Silencio"},
    "ambient": {"id": "/m/015lz1", "name": "Ambient", "description": "Ambiente"},
    "noise": {"id": "/m/02mscn", "name": "Noise", "description": "Ruido"},
    "white_noise": {"id": "/m/05r5c", "name": "White noise", "description": "Ruido blanco"},
    "background_noise": {"id": "/m/0342h", "name": "Background noise", "description": "Ruido de fondo"},

    # Medios y comunicación
    "broadcast": {"id": "/m/026t6", "name": "Broadcast", "description": "Transmisión"},
    "news": {"id": "/m/05k0_", "name": "News", "description": "Noticias"},
    "interview": {"id": "/m/07yv9", "name": "Interview", "description": "Entrevista"},
    "commentary": {"id": "/m/0k4j", "name": "Commentary", "description": "Comentario"},
    "announcement": {"id": "/m/07r04", "name": "Announcement", "description": "Anuncio"},
    "presentation": {"id": "/m/04_sv", "name": "Presentation", "description": "Presentación"},
    "lecture": {"id": "/m/01bjv", "name": "Lecture", "description": "Conferencia"},
    "seminar": {"id": "/m/07jdr", "name": "Seminar", "description": "Seminario"},
    "meeting": {"id": "/m/0cmf2", "name": "Meeting", "description": "Reunión"},
    "conference": {"id": "/m/03j1ly", "name": "Conference", "description": "Conferencia"},

    # Eventos políticos y sociales
    "protest": {"id": "/m/01yrx", "name": "Protest", "description": "Protesta"},
    "demonstration": {"id": "/m/0b_fwt", "name": "Demonstration", "description": "Manifestación"},
    "rally": {"id": "/m/016622", "name": "Rally", "description": "Mitin"},
    "debate": {"id": "/m/03kmc9", "name": "Debate", "description": "Debate"},
    "argument": {"id": "/m/07pkxks", "name": "Argument", "description": "Discusión"},
    "discussion": {"id": "/m/09t49", "name": "Discussion", "description": "Discusión"},
}

# Categorías principales
AUDIOSET_CATEGORIES = {
    "human_voice": ["speech", "male_speech", "female_speech", "child_speech", "conversation", "narration"],
    "human_sounds": ["laughter", "applause", "clapping", "cheering", "booing", "crying", "whistling", "shouting", "whispering"],
    "crowd_sounds": ["crowd", "hubbub", "chatter", "chanting"],
    "music": ["music", "singing", "instrumental_music", "piano", "guitar", "drum", "orchestra"],
    "vehicles": ["vehicle", "car", "truck", "motorcycle", "bus", "train", "aircraft", "helicopter", "engine"],
    "alarms": ["siren", "alarm", "fire_alarm", "smoke_detector", "buzzer"],
    "nature": ["wind", "rain", "thunder", "water", "ocean", "bird", "dog", "cat"],
    "technology": ["telephone", "phone_ringing", "computer", "keyboard", "mouse", "printer", "microphone", "loudspeaker", "television", "radio"],
    "construction": ["construction", "drilling", "hammer", "saw", "machinery"],
    "ambient": ["silence", "ambient", "noise", "white_noise", "background_noise"],
    "media": ["broadcast", "news", "interview", "commentary", "announcement", "presentation", "lecture", "seminar", "meeting", "conference"],
    "political": ["protest", "demonstration", "rally", "debate", "argument", "discussion"]
}

def get_audio_classes_by_category(category: str):
    """Obtiene las clases de audio por categoría"""
    return AUDIOSET_CATEGORIES.get(category, [])

def get_audio_class_info(class_name: str):
    """Obtiene información detallada de una clase de audio"""
    return AUDIOSET_CLASSES.get(class_name, {})

def search_audio_classes(query: str):
    """Busca clases de audio que contengan el término"""
    results = []
    query_lower = query.lower()

    for class_name, info in AUDIOSET_CLASSES.items():
        if (query_lower in class_name.lower() or
            query_lower in info.get("name", "").lower() or
            query_lower in info.get("description", "").lower()):
            results.append({
                "class": class_name,
                "info": info
            })

    return results

def get_all_categories():
    """Obtiene todas las categorías disponibles"""
    return list(AUDIOSET_CATEGORIES.keys())

def get_all_classes():
    """Obtiene todas las clases disponibles"""
    return list(AUDIOSET_CLASSES.keys())

# Función para mostrar la ontología
def print_ontology():
    """Imprime la ontología completa"""
    logging.info("🎵 AudioSet Ontology")
    logging.info("=" * 50)

    for category, classes in AUDIOSET_CATEGORIES.items():
        logging.info(f"\n📂 {category.upper().replace('_', ' ')}")
        logging.info("-" * 30)

        for class_name in classes:
            info = AUDIOSET_CLASSES.get(class_name, {})
            name = info.get("name", class_name)
            description = info.get("description", "")
            logging.info(f"  🔊 {class_name:<20} → {name}")
            if description:
                logging.info(f"      {description}")

if __name__ == "__main__":
    print_ontology()
