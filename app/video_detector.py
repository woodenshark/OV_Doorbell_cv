#!/usr/bin/env python3
import cv2
import time
from multiprocessing import Process
from multiprocessing import Queue
from imutils.video import VideoStream, FileVideoStream
from statistics import mode
from logging import Logger

from .single_shot_detector import FrameProcessor, SingleShotDetector
from .tracker import ObjectTracker
from .face_recognizer import FaceRecognizer
from .utils import Utils
from .output import Output


def detect_in_process(proto, model, proc_frame_size, frame_queue, person_queue, class_num, tolerance):
    ssd_net = Utils.caffe_model_loader(proto, model)
    ssd_proc = FrameProcessor(proc_frame_size, 1.0/127.5, 127.5)
    ssd = SingleShotDetector(ssd_proc, ssd_net)
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            obj_data = ssd.detect(frame)
            persons = ssd.get_objects(frame, obj_data, class_num, tolerance)
            person_queue.put(persons)

class RealtimeVideoDetector():
    def __init__(self, log: Logger, proto, model, proc_frame_size):
        self.log = log
        self.proc_frame_size = proc_frame_size
        self.proto = proto
        self.model = model
        self.frame = None
        self.tracker = ObjectTracker()
        self.face_finder = FaceRecognizer(log, 'models/encodings.pickle', 0.5)
        self.output = Output(log)
        self.statistics = dict()
        self._delay = 0.040 # TODO: research delay for RPi
        self._padding = 20 # for face recognition frame enlarge
        self._person_acc_filter = 100

    def match_faces(self, ids, matching_array):
        names = list()
        for (id_num, id_point) in ids.items():
            name = 'Unknown'
            for match in matching_array:
                match_point, match_name = match
                if match_point == tuple(id_point):
                    name = match_name[0]
                    break
            if id_num in self.statistics:
                if not self.statistics[id_num]['detected']:
                    self.statistics[id_num]['person'].append(name)
                    if len(self.statistics[id_num]['person']) > self._person_acc_filter:
                        self.statistics[id_num]['detected'] = True
                        self.statistics[id_num]['person'] = mode(self.statistics[id_num]['person'])
                        self.output.proceed_attendance(self.statistics[id_num]['person'], self.frame)
                else:
                    name = self.statistics[id_num]['person']
            else:
                self.statistics[id_num] = dict()
                self.statistics[id_num]['detected'] = False
                self.statistics[id_num]['person'] = [name]

            if name == 'Unknown':
                name = 'Stranger'
            names.append(name)
        return names

    def proceed_objects(self, objects):
        if (objects is None) or (len(objects) == 0):
            self.tracker.update([])
            return None

        rects = list()
        box_arr, name_arr, perc_arr = list(), list(), list()
        matching_array = list()
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

            face_result = self.face_finder.find_faces(subframe)
            boxes, names, percents = face_result
            # reframe face boxes to main frame
            for i in range(len(boxes)):
                (sy1, sx2, sy2, sx1) = boxes[i]
                boxes[i] = (
                    sub_y_start + sy1, sub_x_start + sx2,
                    sub_y_start + sy2, sub_x_start + sx1,
                )
            box_arr.extend(boxes)
            name_arr.extend(names)
            perc_arr.extend(percents)

            start_x, start_y, end_x, end_y = rect
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            centroid = (c_x, c_y)
            if names:
                matching_array.append((centroid, names))

        ids = self.tracker.update(rects)
        # TODO: research mismatching from known person to stranger
        name_arr = self.match_faces(ids, matching_array)
        faces = (box_arr, name_arr, perc_arr)
        # verbose output
        #Utils.draw_ids(ids, (0, 255, 0), self.frame) # tracker id point
        #Utils.draw_objects(objects, 'PERSON', (0, 0, 255), self.frame) # person object box
        Utils.draw_face_boxes(faces, self.frame)

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
            args=(self.proto, self.model, self.proc_frame_size, frame_queue, person_queue, class_num, tolerance))
        detect_proc.daemon = True
        detect_proc.start()

        persons = None
        # Capture all frames
        while(True):
            t1 = time.time()

            self.frame = vstream.read()
            if self.frame is None:
                break

            if frame_queue.empty():
                frame_queue.put(self.frame)

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
                self.log.info('Quitting...')
                break

        cv2.destroyAllWindows()
        vstream.stop()

if __name__ == '__main__':
    # TODO: write some short app
    pass
