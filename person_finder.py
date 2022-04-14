import cv2
import time
import imutils
import argparse
import numpy as np

from multiprocessing import Process
from multiprocessing import Queue
from imutils.video import VideoStream, FileVideoStream

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

    def get_objects(self, frame, obj_data, class_num, tolerance):
        objects = []
        for data in obj_data:
            obj_class = int(data[1])
            obj_confidence = data[2]
            if obj_class == class_num and obj_confidence >= tolerance:
                obj = self.get_object(frame, data)
                objects.append(obj)

        return objects

class Utils():
    @staticmethod
    def draw_object(obj, label, color, frame):
        (tolerance, (x1, y1, w, h)) =  obj
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        y3 = y1 - 12
        # for debug
        #text = label + " " + str(tolerance) + "%"
        #cv2.putText(frame, text, (x1, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_objects(objects, label, color, frame):
        for obj in objects:
            Utils.draw_object(obj, label, color, frame)

def detect_in_process(proto, model, ssd_proc, frame_queue, person_queue, class_num, min_confidence):
    ssd_net = CaffeModelLoader.load(proto, model)
    ssd = ShotDetector(ssd_proc, ssd_net)
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            obj_data = ssd.detect(frame)
            persons = ssd.get_objects(frame, obj_data, class_num, min_confidence)
            person_queue.put(persons)

class RealtimeVideoDetector:
    def __init__(self, proto, model, ssd_proc):
        self.ssd_proc = ssd_proc
        self.proto = proto
        self.model = model

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
        # Capture all frames
        while(True):
            frame = vstream.read()
            if frame is None:
                break

            if frame_queue.empty():
                # for debug
                #print ('Put into frame queue ...' + str(frame_num))
                frame_queue.put(frame)

            if not person_queue.empty():
                persons = person_queue.get()
                # for debug
                #print ('Get from person queue ...' + str(len(persons)))

            if (persons is not None) and (len(persons) > 0):
                Utils.draw_objects(persons, 'PERSON', (0, 0, 255), frame)

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
