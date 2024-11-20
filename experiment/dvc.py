import pandas as pd

data = [
    {"name": "Ahmed",  "age": 30, "city": "Casa"},
    {"name": "Saber",  "age": 28, "city": "Rabat"},
    {"name": "Marwane",  "age": 35, "city": "Sale"},
    {"name": "Yacine",  "age": 20, "city": "Sale"},
]


df = pd.DataFrame(data)
df.to_csv("experiment/data/data.csv", index=False)