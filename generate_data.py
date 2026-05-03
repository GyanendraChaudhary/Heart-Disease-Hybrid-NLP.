import pandas as pd
import random

data = []

for i in range(500):
    age = random.randint(30, 80)
    sex = random.randint(0, 1)
    cp = random.randint(0, 3)
    bp = random.randint(100, 180)
    chol = random.randint(150, 350)
    hr = random.randint(100, 200)

    if chol > 240 or bp > 140:
        target = 1
    else:
        target = 0

    data.append([age, sex, cp, bp, chol, hr, target])

df = pd.DataFrame(data, columns=[
    "age", "sex", "cp", "trestbps", "chol", "thalach", "target"
])

df.to_csv("data/heart.csv", index=False)

print("✅ Dataset Generated Successfully!")