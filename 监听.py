import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewPhotoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"New photo detected: {event.src_path}")

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

if __name__ == "__main__":
    folder_to_watch = "/path/to/your/folder"
    watch_folder(folder_to_watch)
