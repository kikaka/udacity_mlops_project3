import requests

url = 'https://udacity-project-3-kaue.herokuapp.com/'

r = requests.get(url)
print(r.json())
print(r.status_code)

data = [
    {
        "age": 75,
        "workclass": "Federal-gov",
        "fnlgt": 100005,
        "education": "Doctorate",
        "education_num": 15,
        "marital_status": "Widowed",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 13450,
        "capital_loss": 1110,
        "hours_per_week": 40,
        "native_country": "England"
    }
]

r2 = requests.post(url + 'predict', json=data)
print(r2.status_code)
print(r2.json())
