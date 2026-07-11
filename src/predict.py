from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "spam_model.pkl")

print("=" * 50)
print("SMS Spam Detection")
print("=" * 50)

while True:

    print("\nEnter your message.")
    print("Finish by typing END on a new line.")
    print("Type EXIT to quit.\n")

    lines = []

    while True:
        line = input()

        if line.upper() == "EXIT":
            exit()

        if line.upper() == "END":
            break

        lines.append(line)

    text = " ".join(lines)

    prediction = model.predict([text])[0]

    print("\n------------------------")

    if prediction == 1:
        print("Prediction : SPAM")
    else:
        print("Prediction : HAM")

    print("------------------------")