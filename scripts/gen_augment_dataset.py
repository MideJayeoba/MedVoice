#!/usr/bin/env python
"""Generate augmentation data: day-to-day conversational English complaints.

The error analysis (scripts/retrain_eval.py) showed the model was trained
mostly on US medical-forum text — long, rambling, typo-heavy. It fails on
short everyday phrasing like "my ear hurts" or "I twisted my ankle at
football". This generates plain conversational complaints, weighted toward
the weakest classes, plus a small share (~10%) of Nigerian Pidgin.

Output: Downloads/triage_everyday_train.csv, triage_everyday_test.csv
"""

import random
from pathlib import Path

import pandas as pd

random.seed(42)

OUT_DIR = Path.home() / "Downloads"

# ---------------------------------------------------------------------------
# Symptom banks: category -> department, [(complaint core, priority), ...]
# Complaint cores are plain everyday English, the way a person actually talks.
# WEIGHT: how many template expansions per core (weak classes get more).
# ---------------------------------------------------------------------------

BANKS = {
    "Sports & Exercise Injuries": ("Orthopaedics", 4, [
        ("I twisted my ankle playing football and it is swollen", "Moderate"),
        ("I pulled a muscle in my thigh while running", "Low"),
        ("my knee started hurting after the gym", "Low"),
        ("I felt something snap in my calf during a match", "High"),
        ("my shoulder hurts whenever I lift weights", "Low"),
        ("I fell during a game and my wrist is swollen and painful", "Moderate"),
        ("I have pain in my heel every time I jog", "Low"),
        ("I strained my back lifting weights and now I can barely bend", "Moderate"),
        ("my elbow aches after playing tennis", "Low"),
        ("I got hit on the head during football and feel dizzy", "Emergency"),
    ]),
    "General Health & Wellness": ("General Medicine", 4, [
        ("I just feel tired all the time even after sleeping well", "Low"),
        ("I want to do a general checkup, nothing is wrong exactly", "Low"),
        ("I feel weak and have no energy these days", "Low"),
        ("I want advice on how to boost my immune system", "Low"),
        ("I keep feeling somehow, not sick but not well either", "Low"),
        ("my body just feels off lately and I can't explain it", "Low"),
        ("I want to know which vitamins I should be taking", "Low"),
        ("I get tired too quickly even from small tasks", "Low"),
    ]),
    "Throat & Voice Problems": ("ENT (Otorhinolaryngology)", 4, [
        ("my throat has been sore for three days and it hurts to swallow", "Moderate"),
        ("I lost my voice and it has not come back", "Moderate"),
        ("my voice has been hoarse for two weeks now", "Moderate"),
        ("it feels like something is stuck in my throat", "Moderate"),
        ("my tonsils are swollen and painful", "Moderate"),
        ("I have a burning feeling in my throat every morning", "Low"),
        ("there is pain on one side of my throat when I swallow", "Moderate"),
    ]),
    "Men's Health": ("Nephrology / Urology", 4, [
        ("I have pain in my testicles that comes and goes", "Moderate"),
        ("I noticed a swelling in my scrotum", "Moderate"),
        ("I am having trouble keeping an erection", "Low"),
        ("I wake up too many times at night to urinate", "Moderate"),
        ("there is a lump on my testicle I am worried about", "High"),
        ("my prostate test came back high and I want to understand it", "Moderate"),
        ("sudden severe pain in my testicle since this morning", "Emergency"),
    ]),
    "Diabetes & Hormonal (Endocrine) Disorders": ("General Medicine", 4, [
        ("I am always thirsty and urinating too often", "Moderate"),
        ("my blood sugar reading was very high this morning", "High"),
        ("I am diabetic and my sugar has not been stable", "Moderate"),
        ("I have been losing weight without trying and always hungry", "Moderate"),
        ("my feet tingle and feel numb, I am diabetic", "Moderate"),
        ("I feel shaky and sweaty when I have not eaten", "Moderate"),
        ("my thyroid results were abnormal and I feel tired always", "Moderate"),
        ("I am gaining weight and feeling cold all the time", "Low"),
        ("I am diabetic and I have a sore on my foot that is not healing", "High"),
    ]),
    "Injuries, Trauma & First Aid": ("Emergency Medicine", 4, [
        ("I cut my hand with a knife and it is bleeding a lot", "Emergency"),
        ("I fell off a bike and hit my head", "Emergency"),
        ("hot water poured on my arm and the skin is peeling", "High"),
        ("I stepped on a nail and my foot is swollen", "High"),
        ("my child fell from the bed and has a bump on the head", "High"),
        ("I was in a car accident and my neck hurts", "Emergency"),
        ("a dog bit me on the leg yesterday", "High"),
        ("I slammed my finger in the door and the nail is turning black", "Moderate"),
        ("something hit my eye at work and it is very painful", "Emergency"),
    ]),
    "Child & Infant Health": ("Paediatrics", 4, [
        ("my baby has been crying nonstop and refusing to feed", "High"),
        ("my two year old has a fever and is not eating", "High"),
        ("my baby has diarrhoea since yesterday", "High"),
        ("my child keeps scratching his ear and crying", "Moderate"),
        ("my baby's temperature is very high and she is having convulsions", "Emergency"),
        ("my son has a rash all over his body", "Moderate"),
        ("my baby is vomiting everything she eats", "High"),
        ("my child has a cough that won't go away", "Moderate"),
        ("my newborn's eyes look yellow", "High"),
        ("my child swallowed a coin", "Emergency"),
    ]),
    "Medication, Poisoning & Substance Use": ("Psychiatry", 4, [
        ("I took too many painkillers by mistake", "Emergency"),
        ("my child drank kerosene", "Emergency"),
        ("I missed my medication for three days, what should I do", "Moderate"),
        ("I think I am addicted to sleeping pills", "Moderate"),
        ("I have been drinking too much lately and can't stop", "Moderate"),
        ("the new drug I was given is making me dizzy and nauseous", "Moderate"),
        ("I want to stop smoking but the cravings are too strong", "Low"),
        ("someone gave me a substance at a party and I feel strange", "Emergency"),
    ]),
    "Ear Problems": ("ENT (Otorhinolaryngology)", 4, [
        ("my ear has been paining me since yesterday", "Moderate"),
        ("there is a ringing sound in my ears that won't stop", "Moderate"),
        ("water entered my ear while swimming and now it hurts", "Moderate"),
        ("I can't hear well from my left ear", "Moderate"),
        ("something is coming out of my ear and it smells", "Moderate"),
        ("my ear is blocked and feels full", "Low"),
        ("an insect entered my ear", "High"),
    ]),
    "Hair & Scalp Problems": ("Dermatology", 4, [
        ("my hair is falling out in patches", "Low"),
        ("my scalp itches badly and has dandruff", "Low"),
        ("I noticed a bald spot on my head", "Low"),
        ("my hairline is receding and I am only 25", "Low"),
        ("there are painful bumps on my scalp", "Moderate"),
        ("my scalp is flaky and my hair breaks easily", "Low"),
    ]),
    "Allergies & Immune Disorders": ("General Medicine", 4, [
        ("I break out in hives whenever I eat certain foods", "Moderate"),
        ("my eyes itch and I sneeze every morning", "Low"),
        ("my face swelled up after taking a new medicine", "Emergency"),
        ("dust makes me sneeze nonstop and my eyes water", "Low"),
        ("I get rashes whenever the weather changes", "Low"),
        ("my throat feels tight after eating groundnuts", "Emergency"),
        ("I keep falling sick, my immunity seems very low", "Moderate"),
    ]),
    "Nose Problems": ("ENT (Otorhinolaryngology)", 4, [
        ("my nose has been blocked for a week", "Low"),
        ("my nose bleeds sometimes for no reason", "Moderate"),
        ("I can't smell anything anymore", "Moderate"),
        ("I have catarrh that won't clear", "Low"),
        ("my nose is always running and I sneeze a lot", "Low"),
        ("there is pain and pressure around my nose and forehead", "Moderate"),
    ]),
    "Sleep Disorders": ("Neurology", 4, [
        ("I can't fall asleep at night no matter how tired I am", "Low"),
        ("I wake up several times every night", "Low"),
        ("my husband says I snore very loudly and stop breathing", "Moderate"),
        ("I feel sleepy all day even after a full night's sleep", "Moderate"),
        ("I keep having nightmares that wake me up", "Low"),
        ("I sometimes can't move when I wake up", "Low"),
    ]),
    "Women's Health": ("Obstetrics & Gynaecology", 3, [
        ("my period has been irregular for months", "Moderate"),
        ("I have very painful cramps during my period", "Moderate"),
        ("my period is heavier than usual this month", "Moderate"),
        ("I have an unusual discharge with a bad smell", "Moderate"),
        ("I missed my period but the pregnancy test is negative", "Moderate"),
        ("I feel a lump in my breast", "High"),
        ("I have pain during intercourse", "Moderate"),
        ("I am having hot flashes and think it may be menopause", "Low"),
        ("I bleed a little after intercourse", "Moderate"),
    ]),
    "Bone, Joint & Muscle Problems": ("Orthopaedics", 3, [
        ("my knees ache when I climb stairs", "Low"),
        ("my lower back has been hurting for two weeks", "Moderate"),
        ("my joints are stiff every morning", "Moderate"),
        ("my neck is stiff and I can't turn it well", "Moderate"),
        ("my fingers hurt and look swollen at the joints", "Moderate"),
        ("I have waist pain that goes down my leg", "Moderate"),
        ("my shoulder is frozen, I can't raise my arm", "Moderate"),
    ]),
    "Sexual Health": ("Nephrology / Urology", 3, [
        ("I noticed a discharge and burning when I urinate", "Moderate"),
        ("I had unprotected sex and I am worried about infection", "Moderate"),
        ("there are painful blisters on my private part", "Moderate"),
        ("I want to get tested for STIs", "Low"),
        ("my partner was diagnosed with an infection, should I be treated too", "Moderate"),
        ("I have itching around my private area", "Moderate"),
    ]),
    "Brain & Nervous System": ("Neurology", 2, [
        ("I get severe headaches almost every day", "Moderate"),
        ("half of my face suddenly feels weak", "Emergency"),
        ("my hands tremble when I try to hold things", "Moderate"),
        ("I fainted twice this week", "High"),
        ("I have pins and needles in my hands and feet", "Moderate"),
        ("I suddenly can't speak properly and my arm is weak", "Emergency"),
        ("I have seizures and ran out of my medication", "High"),
        ("I keep forgetting things, even names of people I know", "Moderate"),
    ]),
    "Infectious Diseases": ("General Medicine", 2, [
        ("I have fever, headache and body pains, I think it is malaria", "Moderate"),
        ("I have been having fever on and off for a week", "Moderate"),
        ("I was tested positive for typhoid last week and still feel weak", "Moderate"),
        ("my whole family has been having fever and rashes", "High"),
        ("I have a boil that is hot and spreading", "Moderate"),
        ("I have night sweats and have been coughing for a month", "High"),
    ]),
    "Digestive & Stomach Problems": ("Gastroenterology", 2, [
        ("my stomach has been running since yesterday", "Moderate"),
        ("I feel a burning pain in my chest after eating", "Low"),
        ("I have not been able to pass stool for four days", "Moderate"),
        ("I keep purging and vomiting since last night", "High"),
        ("my stomach bloats after every meal", "Low"),
        ("there was blood in my stool this morning", "High"),
        ("I have ulcer and the pain is worse this week", "Moderate"),
    ]),
    "Kidney & Urinary Problems": ("Nephrology / Urology", 2, [
        ("it burns when I urinate", "Moderate"),
        ("I saw blood in my urine", "High"),
        ("I have sharp pain in my side that moves to my lower belly", "High"),
        ("my legs and face are swollen in the morning", "High"),
        ("I urinate very little even when I drink plenty water", "High"),
        ("I keep feeling like urinating even after I just went", "Moderate"),
    ]),
    "Liver & Gallbladder Problems": ("Gastroenterology", 2, [
        ("my eyes and skin are turning yellow", "High"),
        ("I have pain under my right ribs after fatty food", "Moderate"),
        ("I was told I have hepatitis B and I don't know what to do", "Moderate"),
        ("my belly is swelling and my eyes look yellowish", "High"),
    ]),
    "Heart & Circulatory Problems": ("Cardiology", 2, [
        ("my chest feels tight and my left arm is numb", "Emergency"),
        ("my heart beats very fast even when I am resting", "High"),
        ("my blood pressure was very high at the pharmacy", "High"),
        ("I get short of breath climbing just a few stairs", "Moderate"),
        ("my legs swell by evening every day", "Moderate"),
        ("I feel chest pain when I walk fast and it stops when I rest", "High"),
    ]),
    "Cancer & Tumor Concerns": ("General Medicine", 2, [
        ("I found a lump under my armpit", "High"),
        ("I have been losing weight fast with no appetite", "High"),
        ("there is a mole on my back that is growing and changing colour", "High"),
        ("my father had cancer and I want to get screened", "Low"),
        ("I have a swelling on my neck that has been growing for months", "High"),
    ]),
    "Mental & Emotional Health": ("Psychiatry", 2, [
        ("I have been feeling sad and empty for weeks", "Moderate"),
        ("I worry about everything and my chest feels tight", "Moderate"),
        ("I don't enjoy anything anymore and I cry easily", "Moderate"),
        ("I have panic attacks where my heart races and I can't breathe", "High"),
        ("sometimes I think about hurting myself", "Emergency"),
        ("I can't concentrate at work and I feel worthless", "Moderate"),
        ("I hear voices when no one is around", "High"),
    ]),
    "Respiratory & Breathing Problems": ("Pulmonology (Respiratory Medicine)", 2, [
        ("I have been coughing for two weeks", "Moderate"),
        ("my chest makes a whistling sound when I breathe", "High"),
        ("I am asthmatic and my inhaler is finished", "High"),
        ("I can't breathe well when I lie down", "High"),
        ("I cough up thick yellow phlegm every morning", "Moderate"),
        ("my child is breathing very fast and her chest is pulling in", "Emergency"),
    ]),
    "Nutrition, Diet & Weight Management": ("General Medicine", 3, [
        ("I want to lose weight but nothing I try works", "Low"),
        ("I am underweight and want to gain in a healthy way", "Low"),
        ("what foods should I avoid with high blood pressure", "Low"),
        ("my child is not growing well and looks too thin", "Moderate"),
        ("I want a meal plan for my diabetes", "Low"),
    ]),
    "Eye & Vision Problems": ("Ophthalmology", 2, [
        ("my eyes are red and itchy", "Low"),
        ("my vision has become blurry recently", "Moderate"),
        ("I see floating spots in my vision", "Moderate"),
        ("my eye is painful and light worsens it", "High"),
        ("something entered my eye and it won't come out", "High"),
        ("I suddenly can't see from one eye", "Emergency"),
    ]),
    "Skin Disorders": ("Dermatology", 2, [
        ("I have an itchy rash on my arms and legs", "Low"),
        ("there are white patches appearing on my skin", "Low"),
        ("my pimples are getting worse and leaving dark spots", "Low"),
        ("I have ringworm on my body that keeps spreading", "Moderate"),
        ("my skin peels and cracks between my toes", "Low"),
        ("a painful boil came up on my thigh", "Moderate"),
    ]),
    "Fever & Flu-like Illnesses": ("General Medicine", 2, [
        ("my body is hot and I have been shivering", "Moderate"),
        ("I have fever, catarrh and my body aches", "Moderate"),
        ("my temperature keeps going up at night", "Moderate"),
        ("I have had a fever for three days that paracetamol won't reduce", "High"),
    ]),
    "Pregnancy & Maternal Health": ("Obstetrics & Gynaecology", 2, [
        ("I am pregnant and have been spotting", "High"),
        ("I am six months pregnant and my feet are very swollen", "Moderate"),
        ("I am pregnant and vomiting everything I eat", "Moderate"),
        ("I am pregnant and the baby has not moved since yesterday", "Emergency"),
        ("I just found out I am pregnant, when should I start antenatal", "Low"),
        ("I am pregnant with severe headache and blurry vision", "Emergency"),
    ]),
    "Dental & Oral Health": ("Dentistry / Oral Health", 2, [
        ("my tooth has been aching seriously for two days", "Moderate"),
        ("my gums bleed whenever I brush", "Low"),
        ("I have a hole in my tooth and cold water pains it", "Moderate"),
        ("my mouth has sores and it hurts to eat", "Moderate"),
        ("my wisdom tooth is coming out and the pain is much", "Moderate"),
        ("my breath smells bad no matter how I brush", "Low"),
    ]),
}

# ---------------------------------------------------------------------------
# Everyday-English templates: how people actually phrase a complaint.
# {c} = complaint core
# ---------------------------------------------------------------------------

TEMPLATES = [
    "{c}",
    "{c}, what can I do?",
    "please {c} and I don't know what to use",
    "good morning, {c}",
    "hello doctor, {c}",
    "{c}. it started {when}",
    "{c} and it is getting worse",
    "{c}, should I be worried?",
    "for {dur} now {c}",
    "{c}. I have not taken any medicine yet",
    "{c}. I took paracetamol but no change",
    "my problem is that {c}",
    "I need help, {c}",
    "{c} and it is disturbing my work",
    "{c}, is there any drug I can buy for it?",
    "{c}. do I need to come to the hospital?",
]

WHEN = ["yesterday", "this morning", "two days ago", "last week", "a few days ago", "last night"]
DUR = ["two days", "three days", "about a week", "almost two weeks", "some time"]

# Small pidgin share so PHC users are still covered (~10%)
PIDGIN_TEMPLATES = [
    "abeg {c}",
    "{c} and e no dey better at all",
    "na since {when} {c}",
    "{c}, wetin I go do?",
    "{c} abeg which medicine I fit use?",
]


def render(core: str, tmpl: str) -> str:
    return tmpl.format(c=core, when=random.choice(WHEN), dur=random.choice(DUR))


def main():
    rows = []
    for category, (department, weight, entries) in BANKS.items():
        for core, priority in entries:
            n_eng = 4 * weight          # everyday-English expansions
            n_pidgin = max(1, weight // 2)  # small pidgin share
            picked = random.sample(TEMPLATES, min(n_eng, len(TEMPLATES)))
            picked += random.sample(PIDGIN_TEMPLATES, min(n_pidgin, len(PIDGIN_TEMPLATES)))
            for tmpl in picked:
                rows.append({
                    "patient_complaint": render(core, tmpl),
                    "disease_category": category,
                    "medical_department": department,
                    "priority_level": priority,
                })

    random.shuffle(rows)
    df = pd.DataFrame(rows).drop_duplicates(subset="patient_complaint")
    n_test = max(1, len(df) // 10)
    test, train = df.iloc[:n_test], df.iloc[n_test:]

    train_path = OUT_DIR / "triage_everyday_train.csv"
    test_path = OUT_DIR / "triage_everyday_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"generated {len(train)} train + {len(test)} test rows")
    print(f"  -> {train_path}\n  -> {test_path}")
    print("\nper-category counts (train):")
    print(train["disease_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
