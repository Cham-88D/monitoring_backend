from flask import Flask, jsonify, request
import threading
from datetime import datetime,timedelta
import time
import cv2
import os
import base64
import tensorflow as tf
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from pymongo import MongoClient
from flask_pymongo import PyMongo
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import custom_object_scope
import tensorflow_hub as hub
from win10toast import ToastNotifier
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
client = MongoClient("mongodb+srv://monitor:1234@cluster0.ysfwo.mongodb.net/?retryWrites=true&w=majority")
db = client.get_database("monitoring")
test = db.test
position = db.position
stress = db.stress
eye = db.eye
sleep = db.sleep

# Flag to indicate if the background task should continue running
running_flag = False
cal_flag = False

position_model_active = True
sleepiness_model_active = True
eye_model_active = True
stress_model_active = True

username = ""
EPOCHS = 10
INIT_LR = 1e-3
BS = 32
DEFAULT_IMAGE_SIZE = tuple((256, 256))
STRESS_IMAGE_SIZE = tuple((224, 224))
IMAGE_SIZE = 0
DIRECTORY_ROOT = 'Data'
WIDTH = 256
HEIGHT = 256
DEPTH = 3

# class CustomKerasLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(CustomKerasLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         pass
#
#     def call(self, inputs):
#         pass
#
#     def get_config(self):
#         config = super(CustomKerasLayer, self).get_config()
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

model_position = tf.keras.models.load_model('./models/posture_classification_model.h5')
model_stress = tf.keras.models.load_model('./models/stress_model.h5')
model_eye = tf.keras.models.load_model('./models/eyes_model.h5')
model_sleep= tf.keras.models.load_model(
       ('./models/model_sleep.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)
# model_sleep = tf.keras.models.load_model('./models/model_sleep.h5')

# feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"  # Replace with the URL or path of the model you are using
# feature_extractor = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
#
# with custom_object_scope({'CustomKerasLayer': CustomKerasLayer}):
#     model_sleep = tf.keras.models.load_model('./models/model_sleep.h5')

def convert_image_to_array(image_dir , size):

    """
    This function converts an image to a numpy array.
    :param image_dir: The directory of the image.
    :return: The numpy array of the image.
    """
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, size)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

#position prediction
def predict_position(image_path):

    global username , position

    image_array = convert_image_to_array(image_path , DEFAULT_IMAGE_SIZE)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image, 0)
    result = model_position.predict(np_image)
    predicted_class = np.argmax(result[0], axis=-1)
    if predicted_class == 0:
        posture = 'Incorrect'
    elif predicted_class == 1:
        posture = 'Correct'
    else:
        posture = 'Not sure'
    print(result)

    # Calculate the confidence level of the prediction
    confidence = round(result[0][predicted_class] * 100, 2)

    if not cal_flag:

        print("postion")
        # Get the current date and time
        current_datetime = datetime.now()
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

        image = cv2.imread(image_path)
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

        # Convert image bytes to base64 for storage
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        posture_data = {
            "username":username,
            "posture":posture,
            "confidence":confidence,
            "date":formatted_datetime.split()[0],
            "time":formatted_datetime.split()[0],
            "image":image_base64
        }

        saved = position.insert_one(posture_data)
        print("saved" , saved)

    print(f'Predicted: {posture} | Confidence: {confidence}% | Confidence: {username}')
    return posture


def predict_stress(image_path):

    global username , stress

    image_array = convert_image_to_array(image_path, STRESS_IMAGE_SIZE)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image, 0)

    prediction = model_stress.predict(np_image)
    prob = prediction
    print(prob[0][0])

    value = prob[0][0]
    predict = ""

    if value >= 0.5:
        predict = "Stress"
    else:
        predict = "Not Stress"

    if not cal_flag:
        # Get the current date and time
        current_datetime = datetime.now()
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

        image = cv2.imread(image_path)
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

        # Convert image bytes to base64 for storage
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        posture_data = {
            "username": username,
            "stress": predict,
            "confidence": value*100,
            "date": formatted_datetime.split()[0],
            "time": formatted_datetime.split()[0],
            "image": image_base64
        }

        saved = stress.insert_one(posture_data)
        print("saved", saved)

    print(f'Predicted: {predict} |  value: {value} |  username: {username}')
    return predict



def calibrate(file_path):
    posture = False
    eye_level = False
    posture = predict_position(file_path)
    eye_result = predict_eye(file_path)
    eye_prediction = eye_result.get("prediction")
    print(eye)
    if posture == "Correct":
        prediction = True
    data = {
        "posture":posture,
        "eye":eye_prediction
    }
    return data

def eye_level(image_path):
    img = cv2.imread(image_path)

    detector = FaceMeshDetector(maxFaces=1)

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        face_landmarks=""

        if len(face) >= 8:
            face_landmarks = face[8]  # Extract face contour landmarks

            # Print the X and Y coordinates of the face contour landmarks
            print("X:", face_landmarks[0], ", Y:", face_landmarks[1])

        pointLeft = face[145]
        pointRight = face[374]

        # pixel value of distance between eyes
        w, _ = detector.findDistance(pointLeft, pointRight)

        # average distance between eyes
        W = 6.3

        # getting the distance(need to find out the f value)
        # f = 600 #for video
        f = 885  # for image
        d = (W * f) / w
        print(d," distance")

        return face_landmarks[0],face_landmarks[1], d
def get_classes(data):
  prob = model_eye.predict(data)[0][0]
  if prob<=0.5:
    return 'Eyes Close', (1 - prob)*100
  else:
    return 'Eyes Open', prob*100

def predict_eye(image_path):

    global username

    posture_data ="test"
    try:
        img = load_img(image_path, target_size=(80, 80, 3))
        img = img_to_array(img)
        img = img / 255.0
        img = img.reshape(1, 80, 80, 3)
        pred , prob = get_classes(img)
        print(f'Predicted EYE : {pred} |  value: {prob} |  username: {username}')

        image_path = os.path.join("./images", "image.jpg")
        x,y,distance = eye_level(image_path)


        # Get the current date and time
        current_datetime = datetime.now()
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

        image = cv2.imread(image_path)
        image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

        # Convert image bytes to base64 for storage
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        posture_data = {
            "username": username,
            "prediction": pred,
            "confidence": prob,
            "x":x,
            "y":y,
            "distance":distance,
            "date": formatted_datetime.split()[0],
            "time": formatted_datetime.split()[0],
            "image": image_base64
        }

        saved = eye.insert_one(posture_data)
        print("saved", saved)
    except Exception as e:
        print("An error occurred:", e)
    return posture_data


def predict_sleepness(filename, model):

    classes = ['not sleepy','sleepy']
    img_ = load_img(filename, target_size=(224, 224))
    img_array = img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    prob = prediction
    print(prob[0])

    # Get the current date and time
    current_datetime = datetime.now()
    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    image = cv2.imread(filename)
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()

    # Convert image bytes to base64 for storage
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    index = np.argmax(prediction)
    print("classes :", classes[index])

    posture_data = {
        "username": username,
        "stress": classes[index],
        "date": formatted_datetime.split()[0],
        "time": formatted_datetime.split()[0],
        "image": image_base64
    }

    saved = sleep.insert_one(posture_data)
    print("saved sleep :", classes[index])

# Your method that runs every second
def background_task():
    global running_flag
    print("running")
    if running_flag:
        global cal_flag

        # Open the camera (use the appropriate index for your camera)
        camera = cv2.VideoCapture(0)

        while running_flag:
            # Capture a frame from the camera
            ret, frame = camera.read()
            image_path = ""

            if ret:
                image_path = os.path.join("./images", "image.jpg")
                # Save the captured frame as an image
                cv2.imwrite(image_path, frame)
                print(f"Image saved")

            if not cal_flag:
                if position_model_active:
                    predict_position(image_path)
                else:
                    print("position mode paused")
                if stress_model_active:
                    predict_stress(image_path)
                else:
                    print("stress mode paused")
                if eye_model_active:
                    predict_eye(image_path)
                else:
                    print("stress mode paused")
                if sleepiness_model_active:
                    predict_sleepness(image_path ,model_sleep)
                else:
                    print("stress mode paused")
                print("send to monitoring")
            else:
                print("send to calibrate")

            time.sleep(1)

        # Release the camera when the task is stopped
        camera.release()
        print("Camera released.")

def background_getting_task():
    global running_flag , username , sleepiness_model_active , position_model_active , stress_model_active , eye_model_active
    toaster = ToastNotifier()

    while running_flag:
        try:
            pos_count = 0
            sleep_count = 0
            stress_count = 0
            # Retrieve latest 10 records from MongoDB collection
            current_datetime = datetime.now()
            three_minutes_ago = current_datetime - timedelta(minutes=3)

            print(three_minutes_ago)

            # Convert the datetime objects to formatted date and time strings
            formatted_current_date = current_datetime.strftime('%Y-%m-%d')
            formatted_current_time = current_datetime.strftime('%H:%M:%S')
            formatted_three_minutes_ago_date = three_minutes_ago.strftime('%Y-%m-%d')
            formatted_three_minutes_ago_time = three_minutes_ago.strftime('%H:%M:%S')

            # Query the records using the date, time, and username
            latest_pos_records = position.find({
                "username": username
                # "date": {"$gte": formatted_three_minutes_ago_date, "$lte": formatted_current_date},
                # "time": {"$gte": formatted_three_minutes_ago_time, "$lte": formatted_current_time}
            }).sort("_id",-1).limit(50)
            latest_sleep_records = sleep.find({
                "username": username
                # "date": {"$gte": formatted_three_minutes_ago_date, "$lte": formatted_current_date},
                # "time": {"$gte": formatted_three_minutes_ago_time, "$lte": formatted_current_time}
            }).sort("_id",-1).limit(50)

            latest_stress_records = stress.find({
                "username": username
                # "date": {"$gte": formatted_three_minutes_ago_date, "$lte": formatted_current_date},
                # "time": {"$gte": formatted_three_minutes_ago_time, "$lte": formatted_current_time}
            }).sort("_id",-1).limit(50)

            latest_eye_records = eye.find({
                "username": username,
                # "date": {"$gte": formatted_three_minutes_ago_date, "$lte": formatted_current_date},
                # "time": {"$gte": formatted_three_minutes_ago_time, "$lte": formatted_current_time}
            }).sort("_id",-1).limit(50)

            print("fetched")

            # Do something with the records (print or process them)
            print("postion")
            for record in latest_pos_records:
                print(record["posture"])
                if record["posture"] == "Incorrect":
                    pos_count += 1  # Increment pos_count using +=

            print("sleep")
            for record in latest_sleep_records:
                if record["stress"] == "sleepy":
                    sleep_count += 1  # Increment pos_count using +=

            print("stress")
            for record in latest_stress_records:
                if record["stress"] == "Stress":
                    stress_count += 1  # Increment pos_count using +=

            print("eye")
            distance = 0
            closed = 0
            level = 0
            for record in latest_eye_records:
                if record["prediction"] == "Eyes Close":
                    print("closed")
                    closed += 1
                if record["distance"] < 50 or record["distance"] > 75:
                    print("distance exceed")
                    distance += 1
                if record["x"] > 100 or record["x"] < 20 or record["y"] > 100 or record["y"] < 20:
                    level += 1


            blink_rate = 0
            if closed > 1:
                blink_rate = closed/3
            print(closed)
            print(blink_rate)

            print("count done")
            # Initialize an alert message
            alert_message = ""

            # Check for different alerts
            if pos_count > 25 and position_model_active:
                alert_message += "Bad posture in last 3 mins\n"

            if sleep_count > 25 and sleepiness_model_active:
                alert_message += "You are Sleepy in last 3 mins, Get Rest\n"

            if stress_count > 25 and stress_model_active:
                alert_message += "You are stressed in last 3 mins, touch the grass\n"

            if distance > 25 and eye_model_active:
                alert_message += "You are stressed due to improper distance from screen, touch the grass\n"

            if (blink_rate > 16 or blink_rate < 8) and eye_model_active:
                alert_message += "Your blink rate is high, consider taking a break\n"

            if level > 25 and eye_model_active:
                alert_message += "Improper Eye level, try to maintain correct eye contact\n"

            if alert_message:
                # Send a consolidated notification
                toaster.show_toast("Alerts", alert_message, duration=10)

            # Sleep for 3 minutes
            time.sleep(180)  # Sleep for 3 minutes (180 seconds)
        except Exception as e:
            print("An error occurred:", e)

# Endpoint to start the background task
@app.route('/start_task', methods=['POST'])
def start_background_task():
    global running_flag , username

    # Get the username from the request data
    data = request.get_json()
    if 'username' in data:
        username = data['username']
    else:
        return jsonify({'message': 'Please provide the "username" in the request data!'}), 400

    if not running_flag:

        running_flag = True
        # Start the background task in a new thread
        task_thread = threading.Thread(target=background_task)
        task_thread.daemon = True  # This will allow the thread to be killed when the main thread exits
        task_thread.start()
        #
        task_thread_2 = threading.Thread(target=background_getting_task())
        task_thread_2.daemon = True  # This will allow the thread to be killed when the main thread exits
        task_thread_2.start()

        # Return a response immediately
        return jsonify({'message': 'Background task started successfully!'})
    else:
        return jsonify({'message': 'Background task is already running!'})

# Endpoint to stop the background task
@app.route('/stop_task', methods=['GET'])
def stop_background_task():
    global running_flag

    if running_flag:
        # Stop the background task by setting the flag to False
        running_flag = False
        return jsonify({'message': 'Background task stopped successfully!'})
    else:
        return jsonify({'message': 'Background task is not running!'})


# Endpoint to start calibrate task
@app.route('/switch', methods=['GET'])
def switch_task():
    global cal_flag , running_flag

    if not cal_flag:
        running_flag = False
        cal_flag = True
        # calibrate()
        # running_flag = True
        # cal_flag = False
        # task_thread = threading.Thread(target=background_task)
        # task_thread.daemon = True  # This will allow the thread to be killed when the main thread exits
        # task_thread.start()
        return jsonify({'message': 'Switch to Calibrate!'})
    else:
        running_flag = True
        cal_flag = False
        task_thread = threading.Thread(target=background_task)
        task_thread.daemon = True  # This will allow the thread to be killed when the main thread exits
        task_thread.start()
        task_thread_2 = threading.Thread(target=background_getting_task())
        task_thread_2.daemon = True  # This will allow the thread to be killed when the main thread exits
        task_thread_2.start()
        return jsonify({'message': 'Switch to Monitoring!'})


@app.route('/test', methods=['POST'])
def upload_image():
    try:
        data = request.json
        if 'image' in data:
            image_data = data['image']
            image_data = image_data.split(',')[1]  # Remove the 'data:image/jpeg;base64,' prefix
            image_bytes = base64.b64decode(image_data)

            filename = os.path.join("./images", 'image.jpg')
            with open(filename, 'wb') as f:
                f.write(image_bytes)
            
            data = calibrate(filename)
            return jsonify({'message': data}), 200
        else:
            return jsonify({'error': 'No image provided.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/control_panel', methods=['POST'])
def control_panel():
    global running_flag, position_model_active, sleepiness_model_active, stress_model_active, eye_model_active

    # Check if the background task is running
    if not running_flag:
        return jsonify({'message': 'Background task is not running!'}), 400

    # Get the request data (position, eye, sleep, and stress values)
    data = request.get_json()
    position = data.get('position', True)
    eye = data.get('eye', True)
    sleep = data.get('sleep', True)
    stress = data.get('stress', True)

    position_model_active = position
    stress_model_active = stress
    sleepiness_model_active = sleep
    eye_model_active = eye

    # Perform actions based on the received values (example: print them for demonstration)
    print(f'Position: {position_model_active}, Eye: {eye_model_active}, Sleep: {sleepiness_model_active}, Stress: {stress_model_active}')

    # You can further process the received values as per your application's requirements.

    return jsonify({'message': 'Control panel data received successfully!'})


if __name__ == '__main__':
    app.run(debug=True)
