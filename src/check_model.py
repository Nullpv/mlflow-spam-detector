from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "logistic_regression.pkl")

messages = [
    "Congratulations! You have won a FREE iPhone.",
    "URGENT! You have won a 1 week FREE membership in our £100000 Prize Jackpot! Text WIN to 80086.",
    "Hi Ali, are you coming to class tomorrow?"
]

for msg in messages:
    pred = model.predict([msg])[0]
    print(msg)
    print("Prediction:", "SPAM" if pred == 1 else "HAM")
    print("-" * 50)
