from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import codecs
import csv
from typing import List
import pickle
import numpy as np
import pandas as pd
import re
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import requests
import shutil

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    with open('test.pickle', 'rb') as f:
        model, scaler = pickle.load(f)
    return _predict_item(_transform_class(item), model, scaler)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    with open('test.pickle', 'rb') as f:
        model, scaler = pickle.load(f)
    return [_predict_item(_transform_class(item), model, scaler) for item in items]


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    csv_values = []
    csvReader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    data = {}
    for rows in csvReader:
        key = rows['name']
        data[key] = rows
    for val in data.values():
        csv_values.append(val)

    results = predict_items(csv_values)

    for i, r in enumerate(csv_values):
        csv_values[i]['predicted_price'] = round(results[i], 2)

    df = pd.DataFrame(csv_values)

    file.file.close()
    return get_csv(df)


def _predict_item(d: dict, model, scaler) -> float:
    cats = {'fuel': ['Diesel', 'LPG', 'Petrol', 'CNG'],
            'seller_type': ['Individual', 'Dealer', 'Trustmark Dealer'],
            'transmission': ['Manual', 'Automatic'],
            'owner': ['First Owner', 'Fourth & Above Owner' 'Second Owner', 'Third Owner', ],
            'seats': [2, 4, 5, 6, 7, 8, 9, 10, 14]}

    X = np.array([d['km_driven'], d['mileage'],
                 d['engine'], d['max_power']])

    X_end = np.array([d['year']*d['year'], d['max_power']/d['mileage']])

    for field in ['fuel', 'seller_type', 'transmission', 'owner', 'seats']:
        X_new = pd.get_dummies(
            pd.Series(cats[field]+[d[field]]), drop_first=True).values[-1].astype(float)
        X = np.concatenate((X, X_new))
    
    X = np.concatenate((X, X_end))
    
    X = scaler.transform(X.reshape(1, -1))
    return model.predict(X)[0]

def _transform_class(item: Item) -> dict:
    if type(item) is not dict:
        d = item.dict()
    else:
        d = item
    for key in ['mileage', 'engine', 'max_power']:
        d[key] = float(re.sub(' \w+', '', d[key]))
    for key in ['year', 'km_driven', 'seats']:
        d[key] = int(d[key])
    return d

def get_csv(df):
    url = "http://127.0.0.1:8000/upload"
    download_file(url)

    df_csv = df.to_csv(index=False)
    response = StreamingResponse(iter([df_csv]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=output.csv"
    return response

def download_file(url):
    local_filename = "download.csv"
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename