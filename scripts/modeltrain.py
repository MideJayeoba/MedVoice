#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for VoiceMedAI Triage Classification Models.
Prompts user for training and testing files, combines them, trains
three separate SVM models (Category, Department, Priority),
and saves them individually and bundled.
"""

import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def clean_medical_text(text):
    """
    Cleans medical text by lowercasing, removing special characters/punctuation,
    and stripping whitespace.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def get_valid_file_path(prompt_label):
    """Interactively prompts the user for a valid, existing file path."""
    while True:
        try:
            path = input(f"Enter path for {prompt_label}: ").strip().strip('\'"')
            if not path:
                print("❌ Input cannot be empty. Please enter a valid path.")
                continue

            # Standard existence check
            if os.path.exists(path):
                return os.path.abspath(path)

            # Fallback check relative to parent directory if script is run from scripts/
            fallback_path = os.path.join("..", path)
            if os.path.exists(fallback_path):
                return os.path.abspath(fallback_path)

            print(f"❌ File not found at '{path}'. Please check the path and try again.")
        except KeyboardInterrupt:
            print("\nTraining cancelled by user.")
            exit(0)


def main():
    print("=======================================================")
    print("🏥 VOICE MED AI - TRIAGE MODEL TRAINING SETUP")
    print("=======================================================")
    print("You will be prompted to provide paths for:")
    print("  - Two (2) training CSV files (which will be combined)")
    print("  - Two (2) testing CSV files (which will be combined)")
    print("=======================================================\n")

    # Interactively prompt for files
    train_file_1 = get_valid_file_path("Training File 1 (CSV)")
    train_file_2 = get_valid_file_path("Training File 2 (CSV)")
    test_file_1 = get_valid_file_path("Testing File 1 (CSV)")
    test_file_2 = get_valid_file_path("Testing File 2 (CSV)")

    print("\n=======================================================")
    print("1. LOADING & COMBINING DATASETS")
    print("=======================================================")
    
    print(f"Loading training data from:\n  1. {train_file_1}\n  2. {train_file_2}")
    df_train_1 = pd.read_csv(train_file_1)
    df_train_2 = pd.read_csv(train_file_2)
    train_df = pd.concat([df_train_1, df_train_2], ignore_index=True)
    
    print(f"Loading testing data from:\n  1. {test_file_1}\n  2. {test_file_2}")
    df_test_1 = pd.read_csv(test_file_1)
    df_test_2 = pd.read_csv(test_file_2)
    test_df = pd.concat([df_test_1, df_test_2], ignore_index=True)

    # Features (Patient complaint text)
    X_train = train_df['patient_complaint'].fillna("")
    X_test = test_df['patient_complaint'].fillna("")

    # Target columns
    y_train_cat = train_df['disease_category']
    y_test_cat = test_df['disease_category']

    y_train_dept = train_df['medical_department']
    y_test_dept = test_df['medical_department']

    y_train_prio = train_df['priority_level']
    y_test_prio = test_df['priority_level']

    print(f"\n✅ Successfully combined and loaded {len(X_train)} training rows and {len(X_test)} testing rows.")

    print("\n=======================================================")
    print("2. CONFIGURING AND TRAINING SVM PIPELINES")
    print("=======================================================")

    # Model 1: Disease Category Classifier
    print("Training Model 1: Disease Category...")
    pipeline_cat = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=clean_medical_text,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True
        )),
        ('clf', LinearSVC(C=0.5, class_weight='balanced', random_state=42, dual='auto'))
    ])
    pipeline_cat.fit(X_train, y_train_cat)
    print("✅ Disease Category Model Trained.")

    # Model 2: Medical Department Classifier
    print("Training Model 2: Medical Department...")
    pipeline_dept = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=clean_medical_text,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True
        )),
        ('clf', LinearSVC(C=0.5, class_weight='balanced', random_state=42, dual='auto'))
    ])
    pipeline_dept.fit(X_train, y_train_dept)
    print("✅ Medical Department Model Trained.")

    # Model 3: Priority Level Classifier
    print("Training Model 3: Medical Urgency (Priority)...")
    pipeline_prio = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=clean_medical_text,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=3,
            sublinear_tf=True
        )),
        ('clf', LinearSVC(C=0.5, class_weight='balanced', random_state=42, dual='auto'))
    ])
    pipeline_prio.fit(X_train, y_train_prio)
    print("✅ Priority Level Model Trained.")

    print("\n=======================================================")
    print("3. EVALUATION REPORTS")
    print("=======================================================")

    # Evaluate Category
    y_pred_cat = pipeline_cat.predict(X_test)
    cat_acc = accuracy_score(y_test_cat, y_pred_cat)
    print(f"🎯 Disease Category Accuracy: {cat_acc * 100:.2f}%")
    print(classification_report(y_test_cat, y_pred_cat))

    # Evaluate Department
    y_pred_dept = pipeline_dept.predict(X_test)
    dept_acc = accuracy_score(y_test_dept, y_pred_dept)
    print(f"\n🎯 Medical Department Accuracy: {dept_acc * 100:.2f}%")
    print(classification_report(y_test_dept, y_pred_dept))

    # Evaluate Priority
    y_pred_prio = pipeline_prio.predict(X_test)
    prio_acc = accuracy_score(y_test_prio, y_pred_prio)
    print(f"\n🎯 Medical Urgency (Priority) Accuracy: {prio_acc * 100:.2f}%")
    print(classification_report(y_test_prio, y_pred_prio))

    print("\n=======================================================")
    print("4. EXPORTING MODELS")
    print("=======================================================")
    
    # Smart output directory resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "scripts" else script_dir
    model_dir = os.path.join(workspace_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save individual models
    # cat_path = os.path.join(model_dir, "category_model.joblib")
    # dept_path = os.path.join(model_dir, "department_model.joblib")
    # prio_path = os.path.join(model_dir, "priority_model.joblib")
    
    # print(f"Saving category model to:\n  -> {cat_path}")
    # joblib.dump(pipeline_cat, cat_path)
    
    # print(f"Saving department model to:\n  -> {dept_path}")
    # joblib.dump(pipeline_dept, dept_path)
    
    # print(f"Saving priority model to:\n  -> {prio_path}")
    # joblib.dump(pipeline_prio, prio_path)

    # Save bundled models dictionary for increased loading efficiency
    bundled_path = os.path.join(model_dir, "triage_models.joblib")
    print(f"Saving bundled models (efficiency mode) to:\n  -> {bundled_path}")
    bundled_dict = {
        "category": pipeline_cat,
        "department": pipeline_dept,
        "priority": pipeline_prio
    }
    joblib.dump(bundled_dict, bundled_path)
    print("\n✅ All models successfully saved to disk!")

    # Live simulation
    run_live_simulation(bundled_path)


def run_live_simulation(bundled_path):
    """Demonstrates loading the bundled models and running inference."""
    if not os.path.exists(bundled_path):
        return

    print("\n=======================================================")
    print("🎤 LIVE REAL-WORLD TRIAGE SIMULATION (USING BUNDLED MODEL)")
    print("=======================================================")
    
    # Load the single bundled file containing all three pipelines
    models = joblib.load(bundled_path)
    pipeline_cat = models["category"]
    pipeline_dept = models["department"]
    pipeline_prio = models["priority"]

    def triage_complaint(text):
        print(f"🗣️ Patient Speaks: '{text}'")
        cleaned_text = clean_medical_text(text)
        
        # Calculate a decision/confidence score from the category classifier
        decision_scores = pipeline_cat.decision_function([cleaned_text])[0]
        # In multi-class, decision_function returns a score per class. 
        # We take the maximum score to assess confidence.
        max_score = decision_scores.max() if hasattr(decision_scores, "max") else decision_scores

        # Rejection threshold check (safety fallback)
        if max_score < -0.5:
            emergency_keywords = ['temperature', 'bleeding', 'numb', 'tight', 'convulsion', 'pain']
            if any(word in cleaned_text for word in emergency_keywords):
                 print("  ⚠️ Low Confidence, but High-Risk Keywords Detected. Routing to Emergency Triage.")
                 print("-" * 50)
                 return
            print("  ⚠️ Unrecognized Complaint. Manual Triage Required.")
            print("-" * 50)
            return

        predicted_category = pipeline_cat.predict([cleaned_text])[0]
        predicted_dept = pipeline_dept.predict([cleaned_text])[0]
        predicted_prio = pipeline_prio.predict([cleaned_text])[0]

        print(f"  📌 Category:     {predicted_category}")
        print(f"  🏥 Department:   {predicted_dept}")
        print(f"  🚨 Urgency:      {predicted_prio}")
        print(f"  📊 Confidence:   {max_score:.2f}")
        print("-" * 50)

    # Test cases
    triage_complaint("My chest has been tight all morning and my left arm is going numb.")
    triage_complaint("I can't stop throwing up and my stomach hurts really bad.")
    triage_complaint("I need to see a skin doctor for this weird rash on my leg.")
    triage_complaint("My son's temperature is very high and he is having convulsions.")
    triage_complaint("I'm looking to get some advice on diet and losing weight.")


if __name__ == "__main__":
    main()