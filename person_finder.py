#!/usr/bin/env python3
from re import A
from typing import OrderedDict
import cv2
import time
import imutils
import argparse
import numpy as np

from multiprocessing import Process
from multiprocessing import Queue
from imutils.video import VideoStream, FileVideoStream
from tracker import ObjectTracker
from face_finder import FaceFinder
from utils import Utils

class CaffeModelLoader():
    @staticmethod
    def load(proto, model):
        net = cv2.dnn.readNetFromCaffe(proto, model)
        return net

class FrameProcessor():
    def __init__(self, size, scale, mean):
        self.size = size
        self.scale = scale
        self.mean = mean

    def get_blob(self, frame):
        img = frame
        (h, w, __) = frame.shape
        if w > h:
            dx = int((w - h) / 2)
            img = frame[0:h, dx:dx + h]
        resized = imutils.resize(img, self.size, self.size, cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(resized, self.scale, (self.size, self.size), self.mean)
        return blob

class ShotDetector():
    def __init__(self, frame_proc, ssd_net):
        self.proc = frame_proc
        self.net = ssd_net

    def detect(self, frame):
        blob = self.proc.get_blob(frame)
        self.net.setInput(blob)
        detections = self.net.forward()
        # detected object count
        k = detections.shape[2]
        obj_data = []
        for i in np.arange(0, k):
            obj = detections[0, 0, i, :]
            obj_data.append(obj)

        return obj_data

    def get_object(self, frame, data):
        tolerance = int(data[2] * 100.0)
        (h, w, _) = frame.shape
        r_x = int(data[3] * h)
        r_y = int(data[4] * h)
        r_w = int((data[5] - data[3]) * h)
        r_h = int((data[6] - data[4]) * h)

        if w > h:
            dx = int((w - h) / 2)
            r_x = r_x + dx

        obj_rect = (r_x, r_y, r_w, r_h)

        return (tolerance, obj_rect)

    def find_objects(self, frame, obj_data, class_num, tolerance):
        objects = []
        for data in obj_data:
            obj_class = int(data[1])
            obj_confidence = data[2]
            if obj_class == class_num and obj_confidence >= tolerance:
                obj = self.get_object(frame, data)
                objects.append(obj)

        return objects

    def _check_inclusion(box_a, box_b):
        padding = 50 # for decrease target box dimensions
        pt_inside = lambda lim1, lim2, coord: lim1 < coord < lim2
        x_a, y_a, w_a, h_a = box_a
        x_b, y_b, w_b, h_b = box_b
        return (pt_inside(x_a, x_a + w_a, x_b + padding) and pt_inside(x_a, x_a + w_a, x_b + w_b - padding)
                and pt_inside(y_a, y_a + h_a, y_b + padding) and pt_inside(y_a, y_a + h_a, y_b + h_b - padding))

    def destroy_subboxes(self, objects):
        fake_objects = []
        for i in range(len(objects)):
            (_, source_rect) = objects[i]
            for j in range(len(objects)):
                if j != i:
                    (_, target_rect) = objects[j]
                    if self._check_inclusion(source_rect, target_rect):
                        fake_objects.append(objects[j])
        for fobj in fake_objects:
            objects.remove(fobj)
        return objects

    def get_objects(self, frame, obj_data, class_num, tolerance):
        objects = self.find_objects(frame, obj_data, class_num, tolerance)
        if (objects is not None) and (len(objects) > 0):
            objects = self.destroy_subboxes(objects)
        return objects

def handle_objects(tracker, finder, objects, frame) -> None:
    if (objects is None) or (len(objects) == 0):
        tracker.update([])
        return None

    padding = 20 # for face recognition frame enlarge
    rects = []
    faces = []
    for obj in objects:
        (_, (x_start, y_start, width, height)) =  obj
        x_end = x_start + width
        y_end = y_start + height

        rect = (x_start, y_start, x_end, y_end)
        rects.append(rect)

        sub_y_start = max(y_start - padding, 0)
        sub_y_end = min(y_end + padding, frame.shape[0])
        sub_x_start = max(x_start - padding, 0)
        sub_x_end = min(x_end + padding, frame.shape[1])
        subframe = frame[
            sub_y_start:sub_y_end,
            sub_x_start:sub_x_end
        ]

        (boxes, names, percents) = finder.find_faces(subframe)
        # reframe face boxes to main frame
        for i in range(len(boxes)):
            (sy1, sx2, sy2, sx1) = boxes[i]
            boxes[i] = (
                sub_y_start + sy1, sub_x_start + sx2,
                sub_y_start + sy2, sub_x_start + sx1,
            )
        faces.append((boxes, names, percents))
    ids = tracker.update(rects)
    Utils.draw_ids(ids, (0, 255, 0), frame)
    Utils.draw_objects(objects, 'PERSON', (0, 0, 255), frame)
    for face in faces:
        Utils.draw_face_boxes(face[0], face[1], face[2], frame)

def detect_in_process(proto, model, ssd_proc, frame_queue, person_queue, class_num, tolerance):
    ssd_net = CaffeModelLoader.load(proto, model)
    ssd = ShotDetector(ssd_proc, ssd_net)
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            obj_data = ssd.detect(frame)
            persons = ssd.get_objects(frame, obj_data, class_num, tolerance)
            person_queue.put(persons)

class RealtimeVideoDetector:
    def __init__(self, proto, model, ssd_proc):
        self.ssd_proc = ssd_proc
        self.proto = proto
        self.model = model
        self._delay = 0.040

    def detect(self, video_source, class_num, tolerance):
        try:
            video_source = int(video_source)
            vstream = VideoStream(src=video_source, framerate=10).start()
        except:
            if video_source == 'pi':
                vstream = VideoStream(usePiCamera=True, framerate=10).start()
            else:
                vstream = FileVideoStream(video_source).start()
        time.sleep(2)

        frame_queue = Queue(maxsize=1)
        person_queue = Queue(maxsize=1)
        detect_proc = Process(
            target=detect_in_process,
            args=(self.proto, self.model, self.ssd_proc, frame_queue, person_queue, class_num, tolerance))
        detect_proc.daemon = True
        detect_proc.start()

        persons = None
        ids = None
        tracker = ObjectTracker()
        finder = FaceFinder('encodings.pickle', 0.6)
        # Capture all frames
        while(True):
            t1 = time.time()

            frame = vstream.read()
            if frame is None:
                break

            if frame_queue.empty():
                # for debug
                #print ('Put into frame queue ...' + str(frame_num))
                frame_queue.put(frame)

            dt = time.time() - t1
            if dt < self._delay:
                st = self._delay - dt
                time.sleep(st)

            if not person_queue.empty():
                persons = person_queue.get()
                # for debug
                #print ('Get from person queue ...' + str(len(persons)))

            handle_objects(tracker, finder, persons, frame)

            # Display the resulting frame
            cv2.imshow('Person detection', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == 113:
                # esc or q button
                print('Quitting...')
                break

        cv2.destroyAllWindows()
        vstream.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person detection script.',
        add_help=False)
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Works asynchronously using queue and threaded video.')
    parser.add_argument('-v', '--video', help='Video source (camera index, filepath or "pi" for picamera)', required=True)
    parser.add_argument('-t', '--tolerance', help='Detection tolerance', default=0.5)
    args = parser.parse_args()
    video = args.video
    tolerance = float(args.tolerance)

    proto_file = r'mobilenet.prototxt'
    model_file = r'mobilenet.caffemodel'
    proc_frame_size = 300
    person_class = 15

    # frame processor for MobileNet
    frame_proc = FrameProcessor(proc_frame_size, 1.0/127.5, 127.5)

    video_ssd = RealtimeVideoDetector(proto_file, model_file, frame_proc)
    video_ssd.detect(video, person_class, tolerance)
