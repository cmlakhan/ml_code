import urllib.request, json

with urllib.request.urlopen("https://www.dropbox.com/s/nkja4978p96f2th/News_Category_Dataset_v2.json?dl=0") as url:
    data = json.loads(url.read().decode())
    print(data)


