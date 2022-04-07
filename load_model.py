import joblib
import requests
import os

def get_model(model_path):
    
    try:
        with open(model_path, "rb") as mh:
            rf = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")
        if not 'model.pkl' in os.listdir('.'):
            # example url: "https://drive.google.com/u/1/uc?id=18IxYOI-whucBTZmt5qTvvYgjlxleaSqO&export=download&confirm=t"
            url = "https://drive.google.com/file/d/17BSrJSDmL7m5ipyK0fEezK42MPouw0bR/view?usp=sharing"
            r = requests.get(url, allow_redirects=True)
            open(r"model.pkl", 'wb').write(r.content)
            del r
        with open(r"model.pkl", "rb") as m:
            rf = joblib.load(m)
    return rf
