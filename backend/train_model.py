"""
train_model.py

Trains the phishing-email classifier described in the spec:
  TF-IDF (max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95)
  -> Logistic Regression (C=1.0, max_iter=1000, random_state=42)

Usage:
    python train_model.py --data ../dataset/emails.csv
    python train_model.py --generate 4000   # build a synthetic dataset first

IMPORTANT
---------
The research paper this spec is based on trained on 53,973 real emails and
reported 95.41% accuracy for Logistic Regression. No real dataset was
provided here, so this script can generate a synthetic-but-structured
dataset (clearly-templated phishing emails vs. clearly-templated legitimate
emails) so the full pipeline runs end-to-end. Swap in a real CSV
(columns: text,label with label in {phishing,legitimate}) to reproduce the
paper's actual numbers.
"""

import argparse
import csv
import os
import random
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocessing import clean_text

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Synthetic dataset generator (only used when --generate is passed or no
# dataset file exists yet)
# --------------------------------------------------------------------------
PHISHING_SUBJECTS = [
    "Account Suspended", "Urgent Security Alert", "Verify Your Identity",
    "Payment Failed", "Unusual Sign-in Activity", "Your Package Could Not Be Delivered",
    "Final Notice: Invoice Overdue", "Claim Your Prize Now", "Password Expires Today",
    "Confirm Your Bank Details", "Action Required: Tax Refund",
]
PHISHING_BODIES = [
    "Dear Customer, your account has been suspended due to unusual activity. "
    "Click here to verify your identity within 24 hours or your account will be "
    "permanently closed: http://secure-login-verify.example.com/reset",
    "URGENT! We detected a problem with your billing information. Please "
    "confirm your password and card number immediately to avoid suspension. "
    "Click the link below to update your account now.",
    "Congratulations! You have been selected as the winner of a $1000 gift "
    "card. Claim your prize now before this limited offer expires. Act now "
    "and click here to verify your details.",
    "Your bank account requires immediate verification. Failure to confirm "
    "your identity within 24 hours will result in a permanent suspension of "
    "your account. Login here to secure your account.",
    "We could not process your recent payment. Update your billing details "
    "immediately by clicking the secure link below, or your subscription "
    "will be cancelled today.",
    "Security Alert: someone tried to access your account from an unknown "
    "device. If this was not you, verify your identity now and reset your "
    "password using the link provided.",
    "Your invoice is overdue. To avoid late fees, please wire transfer the "
    "outstanding amount immediately and confirm your account details via "
    "the link below.",
]

LEGIT_SUBJECTS = [
    "Team Meeting Notes", "Project Update", "Lunch Tomorrow?",
    "Q3 Report Attached", "Weekly Newsletter", "Reminder: Doctor Appointment",
    "Flight Confirmation", "Welcome to the Team", "Your Order Has Shipped",
    "Happy Birthday!", "Conference Schedule",
]
LEGIT_BODIES = [
    "Hi team, thanks for joining today's meeting. I've attached the notes "
    "and action items below. Let me know if I missed anything before Friday.",
    "Hey, are you free for lunch tomorrow around noon? There's a new place "
    "near the office I've been wanting to try. Let me know what works.",
    "Please find attached the Q3 report for review. Happy to walk through "
    "the numbers on our call next week if that's helpful.",
    "This week's newsletter covers our latest product updates, a customer "
    "spotlight, and upcoming events. Thanks for reading, see you next week.",
    "Just a reminder that your appointment is scheduled for next Tuesday at "
    "10am. Please arrive fifteen minutes early to complete your paperwork.",
    "Your flight has been confirmed for the 14th, departing at 9:45am. "
    "Check-in opens 24 hours before departure. Have a great trip!",
    "Welcome to the team! We're excited to have you on board. Your manager "
    "will reach out this week to set up your onboarding schedule.",
    "Good news, your order has shipped and should arrive within 3-5 "
    "business days. You can track the package using the link in your "
    "account order history.",
]


def _synth_row(templates_subject, templates_body, label, rng):
    subj = rng.choice(templates_subject)
    body = rng.choice(templates_body)
    # light randomization so rows aren't all identical duplicates
    filler = rng.choice(["", " Thanks.", " Regards.", " Best,\nTeam", " Let me know."])
    return f"Subject: {subj}\n\n{body}{filler}", label


def generate_synthetic_dataset(path: str, n_rows: int = 4000, seed: int = RANDOM_STATE):
    rng = random.Random(seed)
    n_phish = int(n_rows * 0.55)
    n_legit = n_rows - n_phish

    rows = []
    for _ in range(n_phish):
        rows.append(_synth_row(PHISHING_SUBJECTS, PHISHING_BODIES, "phishing", rng))
    for _ in range(n_legit):
        rows.append(_synth_row(LEGIT_SUBJECTS, LEGIT_BODIES, "legitimate", rng))
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"[generate] wrote {len(rows)} synthetic rows -> {path}")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def load_dataset(path: str):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"].strip().lower())
    return texts, labels


def train(data_path: str, out_dir: str):
    texts, labels = load_dataset(data_path)
    print(f"[train] loaded {len(texts)} emails "
          f"({labels.count('phishing')} phishing / {labels.count('legitimate')} legitimate)")

    print("[train] cleaning text...")
    cleaned = [clean_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.20, random_state=RANDOM_STATE, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression (primary model, per spec)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_vec, y_train)
    lr_preds = lr.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(
        y_test, lr_preds, average="weighted", zero_division=0
    )

    # Multinomial Naive Bayes (comparison model, per spec section 4)
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_preds = nb.predict(X_test_vec)
    nb_acc = accuracy_score(y_test, nb_preds)
    nb_p, nb_r, nb_f1, _ = precision_recall_fscore_support(
        y_test, nb_preds, average="weighted", zero_division=0
    )

    print("\n=== Results ===")
    print(f"{'Model':<22}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1'}")
    print(f"{'Naive Bayes':<22}{nb_acc*100:<12.2f}{nb_p*100:<12.2f}{nb_r*100:<12.2f}{nb_f1*100:.2f}")
    print(f"{'Logistic Regression':<22}{lr_acc*100:<12.2f}{lr_p*100:<12.2f}{lr_r*100:<12.2f}{lr_f1*100:.2f}")
    print("\n" + classification_report(y_test, lr_preds, zero_division=0))

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(lr, os.path.join(out_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.pkl"))
    print(f"[train] saved model.pkl and vectorizer.pkl -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../dataset/emails.csv",
                         help="Path to CSV with columns: text,label")
    parser.add_argument("--generate", type=int, default=0,
                         help="If > 0, generate a synthetic dataset with this many rows first")
    parser.add_argument("--out", default=".", help="Output directory for model.pkl / vectorizer.pkl")
    args = parser.parse_args()

    if args.generate > 0 or not os.path.exists(args.data):
        generate_synthetic_dataset(args.data, n_rows=args.generate or 4000)

    train(args.data, args.out)
