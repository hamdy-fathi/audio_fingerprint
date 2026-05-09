"""
Dialect conversion module.
Transcribes audio, converts words to target dialect, synthesizes new audio.
"""
import asyncio, os, io, base64, tempfile
import edge_tts

# Voice mapping for each dialect (male and female)
DIALECT_VOICES = {
    'egyptian': {
        'male': 'ar-EG-ShakirNeural',
        'female': 'ar-EG-SalmaNeural'
    },
    'gulf': {
        'male': 'ar-SA-HamedNeural',
        'female': 'ar-SA-ZariyahNeural'
    },
    'levantine': {
        'male': 'ar-SY-LaithNeural',
        'female': 'ar-SY-AmanyNeural'
    },
    'maghrebi': {
        'male': 'ar-MA-JamalNeural',
        'female': 'ar-MA-MounaNeural'
    }
}

# Dialect-specific word dictionaries for common phrases
# Maps MSA/common words to dialect-specific equivalents
DIALECT_DICTIONARIES = {
    'egyptian': {
        'أريد': 'عايز', 'كيف حالك': 'إزيك', 'جيد': 'كويس', 'ماذا': 'إيه',
        'الآن': 'دلوقتي', 'كثير': 'كتير', 'جميل': 'حلو', 'لماذا': 'ليه',
        'أذهب': 'أروح', 'هذا': 'ده', 'هذه': 'دي', 'ليس': 'مش',
        'نعم': 'أيوه', 'أنظر': 'بص', 'يتكلم': 'بيتكلم', 'أين': 'فين',
        'صغير': 'صغيّر', 'رجل': 'راجل', 'امرأة': 'ست', 'طفل': 'عيّل',
        'منزل': 'بيت', 'سيارة': 'عربية', 'طعام': 'أكل', 'ماء': 'مية',
        'أفهم': 'فاهم', 'أعرف': 'عارف', 'مشكلة': 'مشكلة', 'شكرا': 'شكرا',
        'صباح الخير': 'صباح الخير', 'مساء الخير': 'مساء الخير',
        'كيف': 'إزاي', 'متى': 'إمتى', 'أحب': 'بحب', 'لا أريد': 'مش عايز'
    },
    'gulf': {
        'أريد': 'أبي', 'كيف حالك': 'شلونك', 'جيد': 'زين', 'ماذا': 'شنو',
        'الآن': 'الحين', 'كثير': 'وايد', 'جميل': 'حلو', 'لماذا': 'ليش',
        'أذهب': 'أروح', 'هذا': 'هذا', 'هذه': 'هذي', 'ليس': 'مو',
        'نعم': 'إي', 'أنظر': 'شوف', 'يتكلم': 'يتكلم', 'أين': 'وين',
        'صغير': 'صغير', 'رجل': 'ريّال', 'امرأة': 'حرمة', 'طفل': 'يهال',
        'منزل': 'بيت', 'سيارة': 'سيارة', 'طعام': 'أكل', 'ماء': 'ماي',
        'أفهم': 'فاهم', 'أعرف': 'أدري', 'مشكلة': 'مشكلة', 'شكرا': 'مشكور',
        'صباح الخير': 'صباح الخير', 'مساء الخير': 'مساء الخير',
        'كيف': 'شلون', 'متى': 'متى', 'أحب': 'أحب', 'لا أريد': 'ما أبي'
    },
    'levantine': {
        'أريد': 'بدي', 'كيف حالك': 'كيفك', 'جيد': 'منيح', 'ماذا': 'شو',
        'الآن': 'هلق', 'كثير': 'كتير', 'جميل': 'حلو', 'لماذا': 'ليش',
        'أذهب': 'روح', 'هذا': 'هاد', 'هذه': 'هاي', 'ليس': 'مش',
        'نعم': 'إي', 'أنظر': 'تطلع', 'يتكلم': 'بيحكي', 'أين': 'وين',
        'صغير': 'زغير', 'رجل': 'زلمة', 'امرأة': 'مرا', 'طفل': 'ولد',
        'منزل': 'بيت', 'سيارة': 'سيارة', 'طعام': 'أكل', 'ماء': 'مي',
        'أفهم': 'فاهم', 'أعرف': 'بعرف', 'مشكلة': 'مشكلة', 'شكرا': 'يسلمو',
        'صباح الخير': 'صباح الخير', 'مساء الخير': 'مساء الخير',
        'كيف': 'كيف', 'متى': 'إيمتى', 'أحب': 'بحب', 'لا أريد': 'ما بدي'
    },
    'maghrebi': {
        'أريد': 'بغيت', 'كيف حالك': 'لاباس', 'جيد': 'مزيان', 'ماذا': 'شنو',
        'الآن': 'دابا', 'كثير': 'بزاف', 'جميل': 'زوين', 'لماذا': 'علاش',
        'أذهب': 'نمشي', 'هذا': 'هاد', 'هذه': 'هادي', 'ليس': 'ماشي',
        'نعم': 'إيه', 'أنظر': 'شوف', 'يتكلم': 'كيهضر', 'أين': 'فين',
        'صغير': 'صغيور', 'رجل': 'راجل', 'امرأة': 'مرا', 'طفل': 'درّي',
        'منزل': 'دار', 'سيارة': 'طونوبيل', 'طعام': 'ماكلة', 'ماء': 'لما',
        'أفهم': 'فاهم', 'أعرف': 'كنعرف', 'مشكلة': 'مشكيل', 'شكرا': 'شكرا',
        'صباح الخير': 'صباح الخير', 'مساء الخير': 'مساء الخير',
        'كيف': 'كيفاش', 'متى': 'فوقاش', 'أحب': 'كنبغي', 'لا أريد': 'ما بغيتش'
    }
}

# Sample sentences for each dialect (for demonstration / audio generation)
DIALECT_SENTENCES = {
    'egyptian': [
        'أنا عايز أروح السوق دلوقتي عشان أشتري حاجات كتير للبيت',
        'إزيك يا صاحبي إنت فين من زمان ما شفتك',
        'الجو النهاردة حلو أوي يلا نطلع نتمشى شوية',
        'أنا بحب الأكل المصري كتير خصوصا الكشري والفول'
    ],
    'gulf': [
        'أنا أبي أروح السوق الحين عشان أشتري أشياء وايد للبيت',
        'شلونك يا خوي وينك من زمان ما شفتك',
        'الجو اليوم حلو وايد يلا نطلع نتمشى شوي',
        'أنا أحب الأكل الخليجي وايد خصوصا الكبسة والمندي'
    ],
    'levantine': [
        'أنا بدي روح عالسوق هلق عشان اشتري أغراض كتير للبيت',
        'كيفك يا صاحبي وينك من زمان ما شفتك',
        'الجو اليوم كتير حلو يلا نطلع نتمشى شوي',
        'أنا بحب الأكل الشامي كتير خصوصا الشاورما والفلافل'
    ],
    'maghrebi': [
        'أنا بغيت نمشي للسوق دابا باش نشري حوايج بزاف للدار',
        'لاباس عليك يا صاحبي فينك من بكري ما شفتك',
        'الجو اليوم زوين بزاف يلا نخرجو نتمشاو شوية',
        'أنا كنبغي الماكلة المغربية بزاف خصوصا الطاجين والكسكس'
    ]
}


def convert_text_to_dialect(text, source_dialect, target_dialect):
    """Convert text from one dialect to another using word dictionaries."""
    if source_dialect == target_dialect:
        return text

    # Build reverse mapping from source dialect words to MSA keys
    source_dict = DIALECT_DICTIONARIES.get(source_dialect, {})
    reverse_source = {v: k for k, v in source_dict.items()}

    # Get target dictionary
    target_dict = DIALECT_DICTIONARIES.get(target_dialect, {})

    converted = text
    # First: source dialect words → MSA
    for dialect_word, msa_word in reverse_source.items():
        if dialect_word in converted:
            converted = converted.replace(dialect_word, msa_word)

    # Then: MSA → target dialect words
    for msa_word, dialect_word in target_dict.items():
        if msa_word in converted:
            converted = converted.replace(msa_word, dialect_word)

    return converted


async def synthesize_speech(text, dialect, gender='male'):
    """Synthesize speech using edge-tts with dialect-appropriate voice."""
    voice = DIALECT_VOICES.get(dialect, DIALECT_VOICES['egyptian']).get(gender, 'ar-EG-ShakirNeural')

    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    return audio_bytes


def synthesize_speech_sync(text, dialect, gender='male'):
    """Synchronous wrapper for synthesize_speech."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(synthesize_speech(text, dialect, gender))
    finally:
        loop.close()
