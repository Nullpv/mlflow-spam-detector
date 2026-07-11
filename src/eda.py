import pandas as pd

# خواندن دیتاست
df = pd.read_csv(
    "../data/sms.tsv",
    sep="\t",
    names=["label", "message"]
)

print("=" * 50)
print("پنج سطر اول دیتاست")
print(df.head())

print("=" * 50)
print("اطلاعات دیتاست")
df.info()

print("=" * 50)
print("تعداد پیام‌های اسپم و عادی")
print(df["label"].value_counts())