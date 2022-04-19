from imutils import resize
import numpy as np
import cv2


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

class SingleShotDetector():
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
        padding = 20 # for decrease target box dimensions
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
