import pandas as pd 

data = pd.read_csv("data_box_office.csv")

print(data["year"].max())