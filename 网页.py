import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
# Load a model
import json
from ultralytics import YOLO
start = time.time()
# Create a new YOLO model from scratch
model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
#model = YOLO(r'F:\yolov8\ultralytics-main\ultralytics-main\weights\yolov8n.pt')
model = YOLO(r'F:\yolov8\ultralytics-main\ultralytics-main\runs\detect\train76\weights\best.pt')
# Load a pretrained YOLO model (recommended for training)
print(time.time()-start)

from flask import Flask, jsonify

app = Flask(__name__)
@app.route('/',methods=["GET"])
def myqr():

    results = model.predict(
        source=r"F:\yolov8\ultralytics-main\ultralytics-main\breathe_data\divide\总数据\images\IMG_20240429_100131_TIMEBURST2.jpg",
        save=True)
    return app.response_class(response=json.dumps(results[-1], ensure_ascii=False))

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080,debug=True)



