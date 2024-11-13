from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import os
from joblib import load
from sklearn.ensemble import RandomForestClassifier 

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
disease_model_path = r'D:\agro\Harvestify\models\plant_disease_model.pth'
crop_recommendation_model_path = r'D:\agro\Harvestify\notebooks\Crop_Recommendation_Model.ipynb'

disease_model = None
try:
    if not os.path.exists(disease_model_path):
        raise FileNotFoundError(f"Disease model file not found at {disease_model_path}")
    
    disease_model = ResNet9(3, len(disease_classes))
    disease_model.load_state_dict(torch.load(
        disease_model_path,
        map_location=torch.device('cpu'),
        weights_only=True
    ))
    disease_model.eval()
    print("Disease model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error loading disease model: {e}")

# Load crop recommendation model
crop_recommendation_model = None
try:
    if not os.path.exists(crop_recommendation_model_path):
        raise FileNotFoundError(f"Crop recommendation model file not found at {crop_recommendation_model_path}")
    
    crop_recommendation_model = load(crop_recommendation_model_path)
    print("Crop recommendation model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    crop_recommendation_model = RandomForestClassifier()
    print("Using a default RandomForestClassifier instead.")
except Exception as e:
    print(f"Error loading crop recommendation model: {e}")
    crop_recommendation_model = RandomForestClassifier()
    print("Using a default RandomForestClassifier instead.")

if disease_model is None:
    print("Warning: Disease model is not loaded. Disease prediction functionality may not work.")

if crop_recommendation_model is None:
    print("Warning: Crop recommendation model is not loaded. Crop recommendation functionality may not work.")


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "https://api.openweathermap.org/data/2.5/weather?lat=44.34&lon=10.99&appid=5b1344bf7d7739587dc1367408eb3b3d"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction


app = Flask(__name__)

# render home page
@app.route('/')
def home():
    title = 'Agronaut- Home'
    return render_template('index.html', title=title)

# render crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'Agronaut - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Agronaut - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Agronaut - Crop Recommendation'

    if request.method == 'POST':
        try:
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            city = request.form.get("city")

            if weather_fetch(city) != None:
                temperature, humidity = weather_fetch(city)
                data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                my_prediction = crop_recommendation_model.predict(data)
                final_prediction = my_prediction[0]

                return render_template('crop-result.html', prediction=final_prediction, title=title)
            else:
                return render_template('try_again.html', title=title)
        except Exception as e:
            print(f"Error in crop prediction: {e}")
            return render_template('try_again.html', title=title, error="An error occurred during prediction. Please try again.")

# render fertilizer recommendation result page
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Agronaut - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        key = 'NHigh' if n < 0 else "Nlow"
    elif max_value == "P":
        key = 'PHigh' if p < 0 else "Plow"
    else:
        key = 'KHigh' if k < 0 else "Klow"

    response = Markup(str(fertilizer_dic[key]))
    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Agronaut - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
