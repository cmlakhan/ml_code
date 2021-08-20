import json
import pandas as pd

f = open('News_Category_Dataset_v2.json',)

data = []
for line in f:
    data.append(json.loads(line))

news_df = pd.DataFrame(data)


