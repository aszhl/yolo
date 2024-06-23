import os
import time
import json
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask import Flask, jsonify
from ultralytics import YOLO

# YOLO模型路径和参数
YOLO_MODEL_PATH = r'F:\yolov8\ultralytics-main\ultralytics-main\runs\detect\train76\weights\best.pt'
MONITOR_FOLDER = r'F:\yolov8\ultralytics-main\ultralytics-main\data\images'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 加载YOLO模型
start = time.time()
model = YOLO(YOLO_MODEL_PATH)
print(f"Model loaded in {time.time() - start} seconds")

# Flask应用
app = Flask(__name__)

# 保存最新检测的照片路径
latest_photo_path = None


class NewPhotoHandler(FileSystemEventHandler):
    def on_created(self, event):
        global latest_photo_path
        if not event.is_directory and event.src_path.endswith(('.png', '.jpg', '.jpeg')):
            print(f"New photo detected: {event.src_path}")
            latest_photo_path = event.src_path
            # 调用YOLO模型进行预测
            results = model.predict(source=latest_photo_path, save=True)
            # 保存预测结果为JSON格式
            with open('result.json', 'w', encoding='utf-8') as f:
                json.dump(results[-1], f, ensure_ascii=False)


@app.route('/', methods=["GET"])
def myqr():
    # 返回最新检测照片的预测结果
    if latest_photo_path:
        with open('result.json', 'r', encoding='utf-8') as f:
            result = json.load(f)
        return app.response_class(response=json.dumps(result, ensure_ascii=False), mimetype='application/json')
    else:
        return jsonify({"message": "No photo detected yet"}), 400


def watch_folder(folder_path):
    event_handler = NewPhotoHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    # 启动文件夹监控
    observer_thread = threading.Thread(target=watch_folder, args=(MONITOR_FOLDER,))
    observer_thread.start()

    # 启动Flask应用
    app.run(host='127.0.0.1', port=8080, debug=True)
