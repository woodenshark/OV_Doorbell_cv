#!/usr/bin/env python3
import cv2
from pathlib import Path
import pickle


class Utils():
    @staticmethod
    def caffe_model_loader(proto, model):
        net = cv2.dnn.readNetFromCaffe(proto, model)
        return net

    @staticmethod
    def pickle_model_loader(log, model):
        path = Path(model)
        if not path.exists():
            log.error(f'Model encodings file {model} is not found.')
            exit(-1)

        log.info('Loading encodings for face recognizer.')
        with path.open(mode='rb') as f:
            data = pickle.loads(f.read())
        return data

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

    @staticmethod
    def draw_ids(ids, color, frame):
        for (object_id, centroid) in ids.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = f'ID {object_id}'
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

    @staticmethod
    def draw_face_boxes(face_result, frame):
        for ((top, right, bottom, left), name, perc) in zip(*face_result):
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            if name != 'Stranger':
                text = f'{name} {perc}%'
            else:
                text = f'{name}'

            cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 255, 255), 2)
