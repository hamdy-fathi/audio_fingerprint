"""
Generate sample audio files using edge-tts for each dialect.
4 dialects x 4 speakers = 16 audio files.
"""
import asyncio
import os
import edge_tts

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'audio_samples')

# 4 different voices per dialect (mix of male/female where available)
DIALECT_CONFIG = {
    'egyptian': {
        'voices': [
            ('ar-EG-ShakirNeural', 'speaker1'),
            ('ar-EG-SalmaNeural', 'speaker2'),
            ('ar-EG-ShakirNeural', 'speaker3'),
            ('ar-EG-SalmaNeural', 'speaker4'),
        ],
        'texts': [
            'أنا عايز أروح السوق دلوقتي عشان أشتري حاجات كتير للبيت. النهاردة الجو حلو أوي والشمس طالعة. يلا بينا نتمشى في الشارع ونشوف الناس. مصر حلوة أوي وأنا بحبها كتير. الأكل المصري من أحسن الأكل في الدنيا كلها.',
            'إزيك يا صاحبي إنت فين من زمان ما شفتك. كنت عايز أكلمك في موضوع مهم. تعالى نقعد في القهوة ونتكلم شوية. عندي كلام كتير عايزة أقولهولك. الدنيا مشغولة بس لازم نلاقي وقت لبعض.',
            'الجو النهاردة حلو أوي يلا نطلع نتمشى شوية في الشارع. نروح الحديقة ونقعد هناك. الولاد عايزين يلعبوا برة. هنشتري أكل من المحل اللي جنب البيت. أنا جعان أوي ومش قادر أستنى كتير.',
            'أنا بحب الأكل المصري كتير خصوصا الكشري والفول والطعمية. أمي بتعمل أحلى أكل في الدنيا. كل يوم الصبح بنفطر فول وطعمية. والعيش المصري ما فيش زيه. الحمد لله على نعمة الأكل.',
        ]
    },
    'gulf': {
        'voices': [
            ('ar-SA-HamedNeural', 'speaker1'),
            ('ar-SA-ZariyahNeural', 'speaker2'),
            ('ar-SA-HamedNeural', 'speaker3'),
            ('ar-SA-ZariyahNeural', 'speaker4'),
        ],
        'texts': [
            'أنا أبي أروح السوق الحين عشان أشتري أشياء وايد للبيت. اليوم الجو حلو وايد والشمس طالعة. يلا بينا نتمشى في الشارع ونشوف الناس. السعودية حلوة وايد وأنا أحبها. الأكل الخليجي من أحسن الأكل في الدنيا كلها.',
            'شلونك يا خوي وينك من زمان ما شفتك. كنت أبي أكلمك في موضوع مهم. تعال نقعد في المقهى ونتكلم شوي. عندي كلام وايد أبي أقولك إياه. الدنيا مشغولة بس لازم نلقى وقت لبعض.',
            'الجو اليوم حلو وايد يلا نطلع نتمشى شوي في الشارع. نروح الحديقة ونقعد هناك. العيال يبون يلعبون برا. بنشتري أكل من المحل اللي جنب البيت. أنا يوعان وايد وما أقدر أستنى.',
            'أنا أحب الأكل الخليجي وايد خصوصا الكبسة والمندي. أمي تسوي أحلى أكل في الدنيا. كل يوم الصبح نفطر فول وتميس. والعيش السعودي ما في زيه أبد. الحمد لله على نعمة الأكل.',
        ]
    },
    'levantine': {
        'voices': [
            ('ar-SY-LaithNeural', 'speaker1'),
            ('ar-SY-AmanyNeural', 'speaker2'),
            ('ar-SY-LaithNeural', 'speaker3'),
            ('ar-SY-AmanyNeural', 'speaker4'),
        ],
        'texts': [
            'أنا بدي روح عالسوق هلق عشان اشتري أغراض كتير للبيت. اليوم الجو كتير حلو والشمس طالعة. يلا نتمشى بالشارع ونشوف الناس. الشام حلوة كتير وأنا بحبها. الأكل الشامي من أطيب الأكل بالدنيا كلها.',
            'كيفك يا صاحبي وينك من زمان ما شفتك. كنت بدي احكي معك بموضوع مهم. تعال نقعد بالمقهى ونحكي شوي. عندي حكي كتير بدي قولك إياه. الدنيا مشغولة بس لازم نلاقي وقت لبعض.',
            'الجو اليوم كتير حلو يلا نطلع نتمشى شوي بالشارع. نروح عالحديقة ونقعد هنيك. الولاد بدهم يلعبو برا. بنشتري أكل من المحل يلي جنب البيت. أنا جوعان كتير وما بقدر استنى.',
            'أنا بحب الأكل الشامي كتير خصوصا الشاورما والفلافل. أمي بتعمل أطيب أكل بالدنيا. كل يوم الصبح منفطر فول وحمص. والخبز الشامي ما في متلو. الحمد لله على نعمة الأكل.',
        ]
    },
    'maghrebi': {
        'voices': [
            ('ar-MA-JamalNeural', 'speaker1'),
            ('ar-MA-MounaNeural', 'speaker2'),
            ('ar-MA-JamalNeural', 'speaker3'),
            ('ar-MA-MounaNeural', 'speaker4'),
        ],
        'texts': [
            'أنا بغيت نمشي للسوق دابا باش نشري حوايج بزاف للدار. اليوم الجو زوين بزاف والشمس طالعة. يلا نتمشاو في الزنقة ونشوفو الناس. المغرب زوين بزاف وأنا كنبغيه. الماكلة المغربية من أحسن الماكلة في الدنيا كلها.',
            'لاباس عليك يا صاحبي فينك من بكري ما شفتك. كنت بغيت نهضر معاك في موضوع مهم. أجي نقعدو في القهوة ونهضرو شوية. عندي هضرة بزاف بغيت نقولها ليك. الدنيا مشغولة بصح لازم نلقاو وقت لبعضياتنا.',
            'الجو اليوم زوين بزاف يلا نخرجو نتمشاو شوية في الزنقة. نمشيو للجردة ونقعدو تما. الدراري بغاو يلعبو برا. غنشريو ماكلة من الحانوت اللي حدا الدار. أنا جيعان بزاف وما نقدرش نتسنى.',
            'أنا كنبغي الماكلة المغربية بزاف خصوصا الطاجين والكسكس. أمي كتصوب أحسن ماكلة في الدنيا. كل نهار الصباح كنفطرو بالحرشة والبغرير. والخبز المغربي ما كاين والو بحالو. الحمد لله على نعمة الماكلة.',
        ]
    }
}


async def generate_all_samples():
    """Generate all sample audio files."""
    print("=" * 60)
    print("Generating Arabic Dialect Audio Samples")
    print("=" * 60)

    for dialect, config in DIALECT_CONFIG.items():
        dialect_dir = os.path.join(SAMPLES_DIR, dialect)
        os.makedirs(dialect_dir, exist_ok=True)
        print(f"\n[{dialect.upper()}]")

        for i, (voice, speaker_name) in enumerate(config['voices']):
            text = config['texts'][i]
            output_path = os.path.join(dialect_dir, f"{speaker_name}.wav")

            if os.path.exists(output_path):
                print(f"  {speaker_name}.wav already exists, skipping")
                continue

            print(f"  Generating {speaker_name}.wav with voice {voice}...")
            try:
                # Generate MP3 first, then we'll use it as-is (edge-tts outputs mp3)
                mp3_path = output_path.replace('.wav', '.mp3')
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(mp3_path)

                # Convert mp3 to wav using pydub
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(output_path, format="wav")
                os.remove(mp3_path)

                print(f"  [OK] {speaker_name}.wav generated successfully")
            except Exception as e:
                print(f"  [ERR] Error generating {speaker_name}.wav: {e}")

    print("\n" + "=" * 60)
    print("Done! All samples generated.")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(generate_all_samples())
