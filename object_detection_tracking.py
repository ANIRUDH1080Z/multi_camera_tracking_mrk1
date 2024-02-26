import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Define camera URLs
cam_urls = [
    "videos\Double1.mp4",
    "videos\Single1.mp4"
    # Add more camera URLs if needed
]

# Create video capture, model, and tracker objects for each camera
video_caps = [cv2.VideoCapture(url) for url in cam_urls]
models = [YOLO("yolov8n.pt") for _ in cam_urls]
trackers = [DeepSort(max_age=50, embedder='torchreid', embedder_model_name='osnet_ain_x1_0', embedder_wts='osnet_x1_0_imagenet.pth') for _ in cam_urls]

def process_video(cam_index, video_cap, model, tracker):
    while True:
        start = datetime.datetime.now()

        ret, frame = video_cap.read()

        if not ret:
            break

        # run the YOLO model on the frame
        detections = model(frame, classes=[0])[0]

        # initialize the list of bounding boxes and confidences
        results = []

        # loop over the detections
        for data in detections.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        # update the tracker with the new detections
        tracks = tracker.update_tracks(results, frame=frame)

        # loop over the tracks
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, f"Cam {cam_index} - ID: {track_id}", (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

        end = datetime.datetime.now()
        print(f"Cam {cam_index} - Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        cv2.imshow(f"Cam {cam_index} - Frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

# Create threads for each camera
threads = [threading.Thread(target=process_video, args=(i, video_cap, model, tracker)) for i, (video_cap, model, tracker) in enumerate(zip(video_caps, models, trackers))]

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Release resources
for video_cap in video_caps:
    video_cap.release()

cv2.destroyAllWindows()






































































# import datetime
# from ultralytics import YOLO
# import cv2
# from helper import create_video_writer
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import threading

# CONFIDENCE_THRESHOLD = 0.8
# GREEN = (0, 255, 0)
# WHITE = (255, 255, 255)

# # Define camera URLs
# cam_urls = [
#     # "rtsp://admin:admin12345@192.168.1.223:554/Streaming/channels/401",
#     # "rtsp://admin:admin12345@192.168.1.223:554/Streaming/channels/301",
#     "videos\Double1.mp4",
#     "videos\Single1.mp4"
#     # Add more camera URLs if needed
# ]

# # Create video capture, video writer, model, and tracker objects for each camera
# video_caps = [cv2.VideoCapture(url) for url in cam_urls]
# writers = [create_video_writer(cap, f"output_camera_{i}.mp4") for i, cap in enumerate(video_caps)]
# models = [YOLO("yolov8n.pt") for _ in cam_urls]
# trackers = [DeepSort(max_age=50, embedder='torchreid', embedder_model_name='osnet_ain_x1_0', embedder_wts='osnet_x1_0_imagenet.pth') for _ in cam_urls]

# def process_video(cam_index, video_cap, writer, model, tracker):
#     while True:
#         start = datetime.datetime.now()

#         ret, frame = video_cap.read()

#         if not ret:
#             break

#         # run the YOLO model on the frame
#         detections = model(frame, classes=[0])[0]

#         # initialize the list of bounding boxes and confidences
#         results = []

#         # loop over the detections
#         for data in detections.boxes.data.tolist():
#             confidence = data[4]

#             if float(confidence) < CONFIDENCE_THRESHOLD:
#                 continue

#             xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
#             class_id = int(data[5])
#             results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

#         # update the tracker with the new detections
#         tracks = tracker.update_tracks(results, frame=frame)

#         # loop over the tracks
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue

#             track_id = track.track_id
#             ltrb = track.to_ltrb()

#             xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
#             cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
#             cv2.putText(frame, f"Cam {cam_index} - ID: {track_id}", (xmin + 5, ymin - 8),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

#         end = datetime.datetime.now()
#         print(f"Cam {cam_index} - Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
#         fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
#         cv2.putText(frame, fps, (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

#         cv2.imshow(f"Cam {cam_index} - Frame", frame)
#         writer.write(frame)

#         if cv2.waitKey(1) == ord("q"):
#             break

# # Create threads for each camera
# threads = [threading.Thread(target=process_video, args=(i, video_cap, writer, model, tracker)) for i, (video_cap, writer, model, tracker) in enumerate(zip(video_caps, writers, models, trackers))]

# # Start the threads
# for thread in threads:
#     thread.start()

# # Wait for all threads to finish
# for thread in threads:
#     thread.join()

# # Release resources
# for video_cap in video_caps:
#     video_cap.release()

# for writer in writers:
#     writer.release()

# cv2.destroyAllWindows()
