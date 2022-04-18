#!/usr/bin/env python3
import cv2
import time
import argparse
import numpy as np

from multiprocessing import Process
from multiprocessing import Queue
from imutils.video import VideoStream, FileVideoStream
from imutils import resize
from tracker import ObjectTracker
from face_finder import FaceFinder
from utils import Utils
from statistics import mode
from datetime import datetime

from output import Output


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
        resized = resize(img, self.size, self.size, cv2.INTER_AREA)
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

    def _check_inclusion(self, box_a, box_b):
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
        self.frame = None
        self.tracker = ObjectTracker()
        self.face_finder = FaceFinder('encodings.pickle', 0.6)
        self.output = Output()
        self.statistics = dict()
        self._delay = 0.040
        self._padding = 20 # for face recognition frame enlarge
        self._person_acc_filter = 50

    def match_faces(self, ids, matching_array):
        for (id_num, id_point) in ids.items():
            name = 'Unknown'
            for match in matching_array:
                match_point, match_name = match
                if match_point == tuple(id_point):
                    name = match_name[0]
            if id_num in self.statistics:
                if not self.statistics[id_num]['detected']:
                    self.statistics[id_num]['person'].append(name)
                    if len(self.statistics[id_num]['person']) > self._person_acc_filter:
                        self.statistics[id_num]['detected'] = True
                        self.statistics[id_num]['person'] = mode(self.statistics[id_num]['person'])
                        self.statistics[id_num]['timestamp'] = str(datetime.now())
                        self.output.proceed_attendance(self.statistics[id_num]['person'], self.frame)
            else:
                self.statistics[id_num] = dict()
                self.statistics[id_num]['detected'] = False
                self.statistics[id_num]['person'] = [name]

    def proceed_objects(self, objects):
        if (objects is None) or (len(objects) == 0):
            self.tracker.update([])
            return None

        rects = []
        faces = []
        matching_array = []
        for obj in objects:
            (_, (x_start, y_start, width, height)) =  obj
            x_end = x_start + width
            y_end = y_start + height

            rect = (x_start, y_start, x_end, y_end)
            rects.append(rect)

            sub_y_start = max(y_start - self._padding, 0)
            sub_y_end = min(y_end + self._padding, self.frame.shape[0])
            sub_x_start = max(x_start - self._padding, 0)
            sub_x_end = min(x_end + self._padding, self.frame.shape[1])
            subframe = self.frame[
                sub_y_start:sub_y_end,
                sub_x_start:sub_x_end
            ]

            (boxes, names, percents) = self.face_finder.find_faces(subframe)
            # reframe face boxes to main frame
            for i in range(len(boxes)):
                (sy1, sx2, sy2, sx1) = boxes[i]
                boxes[i] = (
                    sub_y_start + sy1, sub_x_start + sx2,
                    sub_y_start + sy2, sub_x_start + sx1,
                )
            faces.append((boxes, names, percents))

            start_x, start_y, end_x, end_y = rect
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            centroid = (c_x, c_y)
            if names:
                matching_array.append((centroid, names))

        ids = self.tracker.update(rects)
        self.match_faces(ids, matching_array)
        Utils.draw_ids(ids, (0, 255, 0), self.frame)
        Utils.draw_objects(objects, 'PERSON', (0, 0, 255), self.frame)
        for face in faces:
            Utils.draw_face_boxes(face[0], face[1], face[2], self.frame)

    def detect(self, video_source, class_num, tolerance):
        try:
            video_source = int(video_source)
            vstream = VideoStream(src=video_source, framerate=10).start()
            self._delay = 0
        except:
            if video_source == 'pi':
                vstream = VideoStream(usePiCamera=True, framerate=10).start()
                self._delay = 0
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
        # Capture all frames
        while(True):
            if self._delay:
                t1 = time.time()

            self.frame = vstream.read()
            if self.frame is None:
                break

            if frame_queue.empty():
                frame_queue.put(self.frame)

            if self._delay:
                dt = time.time() - t1
                if dt < self._delay:
                    st = self._delay - dt
                    time.sleep(st)

            if not person_queue.empty():
                persons = person_queue.get()

            self.proceed_objects(persons)

            # Display the resulting frame
            cv2.imshow('Person detection', self.frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == 113:
                # esc or q button
                print('Quitting...')
                break

        cv2.destroyAllWindows()
        vstream.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Person detection and face recognition tool.',
        add_help=False)
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Press escape or q key to exit.')
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
