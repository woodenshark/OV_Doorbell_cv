#!/usr/bin/env python3
from imutils.video import VideoStream, FileVideoStream
from imutils import resize
import face_recognition
import argparse
import time
import cv2
import numpy as np
from logging import Logger, basicConfig, DEBUG, getLogger

try:
    from .utils import Utils
except:
    from utils import Utils


class FaceRecognizer():
    def __init__(self, log: Logger, model: str, tolerance: float):
        self.log = log
        self.data = Utils.pickle_model_loader(log, model)
        self.tolerance = tolerance

    def find_faces(self, frame):
        boxes = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []
        percs = []

        distance = lambda face_encodings, face_to_compare: \
            np.linalg.norm(face_encodings - face_to_compare, axis=1) \
            if len(face_encodings) != 0 else np.empty(0)

        for encoding in encodings:
            percent = distance(self.data['encodings'], encoding)
            matches = list(percent <= self.tolerance)

            (name, perc) = ('Stranger', 0)
            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                percents = {}

                for i in matched_idxs:
                    name = self.data['names'][i]
                    counts[name] = counts.get(name, 0) + 1
                    percents[name] = min([percents.get(name, 100), percent[i]])

                name = max(counts, key=counts.get)
                perc = int((1 - percents[name]) * 100)

            names.append(name)
            percs.append(perc)

        return (boxes, names, percs)

def recognition_loop(log: Logger, video_source: str, model: str, tolerance: float):
    face_req = FaceRecognizer(log, model, tolerance)
    log.info('Starting video stream')
    try:
        video_source = int(video_source)
        vstream = VideoStream(src=video_source, framerate=10).start()
    except:
        if video_source == 'pi':
            vstream = VideoStream(usePiCamera=True, framerate=10).start()
        else:
            vstream = FileVideoStream(video_source).start()
    time.sleep(2)

    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vstream.read()
        frame = resize(frame, width=500)

        face_result = face_req.find_faces(frame)
        (boxes, names, tolerances) = face_result
        log.debug(f'Found {names} with {tolerances} confidence in {boxes}.')

        Utils.draw_face_boxes(face_result, frame)

        # display the image to our screen
        cv2.imshow('Facial Recognition', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == 113:
            # esc or q button
            log.info(f'Quitting...')
            break

    cv2.destroyAllWindows()
    vstream.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take dataset photos of new known persons",
        add_help=False)
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Press space to take a photo, escape or q to exit.')
    parser.add_argument('-v', '--video', help='Video source (camera index, filepath or "pi" for picamera)', required=True)
    parser.add_argument('-m', '--model', help='Model encodings source file', default='encodings.pickle')
    parser.add_argument('-t', '--tolerance', help='Detection tolerance', default=0.6)
    args = parser.parse_args()
    video = args.video
    model = args.model
    tolerance = float(args.tolerance)

    basicConfig(
        level=DEBUG,
        format='%(asctime)s %(name)s %(levelname)s -- %(message)s'
        )
    log = getLogger()

    recognition_loop(log, video, model, tolerance)
