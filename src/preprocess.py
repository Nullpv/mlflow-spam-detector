import pandas as pd

# خواندن دیتاست
df = pd.read_csv(
    "../data/sms.tsv",
    sep="\t",
    names=["label", "message"]
)

print("Before preprocessing")
print(df.head())

# تبدیل برچسب‌ها به عدد
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nAfter label encoding")
print(df.head())


import pandas as pd

# خواندن دیتاست
df = pd.read_csv(
    "../data/sms.tsv",
    sep="\t",
    names=["label", "message"]
)

# تبدیل برچسب‌ها به عدد
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# حذف پیام‌های تکراری
df = df.drop_duplicates()

# بررسی داده‌های خالی
df = df.dropna()

print(df.head())

print("\nShape:")
print(df.shape)

# ذخیره دیتاست تمیز
df.to_csv("../data/processed.csv", index=False)

print("\nprocessed.csv saved successfully.")