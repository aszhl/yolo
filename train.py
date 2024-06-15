import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
# Load a model

from ultralytics import YOLO
"""start = time.time()
print(start)"""
# Create a new YOLO model from scratch
model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
#model = YOLO(r'F:\yolov8\ultralytics-main\ultralytics-main\weights\yolov8n.pt')
#model = YOLO(r'F:\yolov8\ultralytics-main\ultralytics-main\runs\detect\train63\weights\best.pt')
# Load a pretrained YOLO model (recommended for training)

model = YOLO(r"F:\yolov8\ultralytics-main\ultralytics-main\runs\detect\train76\weights\best.pt")  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data=r"breathe_data/data.yaml",epochs=100)
#print(time.time(),'模型加载完毕')
results = model.predict(source = r"F:\yolov8\ultralytics-main\ultralytics-main\breathe_data\divide\第三次数据\images",save=True)
#print(results)
#pred =time.time()
#print(pred-start,'预测1')
#results1 = model.predict(source = r"F:\yolov8\ultralytics-main\ultralytics-main\breathe_data\buchongdata\images\IMG20240229112635_BURST010.jpg",save = True)

print(results[-1])
#print(results1[-1])
#print(time.time()-pred,'预测2')


