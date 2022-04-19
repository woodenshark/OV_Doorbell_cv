#!/usr/bin/env python3
import unittest
import unittest.mock
from face_recognizer import FaceRecognizer
from pathlib import Path
import os
import cv2


class Test_model_with_dataset(unittest.TestCase):

    dataset_folder = '../dataset'

    def setUp(self):
        path = Path(self.dataset_folder)
        self.img_paths = tuple(path.rglob('*.jpg'))
        self.tolerance = 0.6
        log = unittest.mock.Mock()
        self.face_finder = FaceRecognizer(log, '../encodings.pickle', self.tolerance)

    def test_person(self):
        for img_path in self.img_paths:
            with self.subTest(img_path=img_path):
                frame = cv2.imread(str(img_path))
                boxes, faces, tolerances = self.face_finder.find_faces(frame)
                box, face, tolerance = boxes[0], faces[0], tolerances[0]
                name = str(img_path).split(os.path.sep)[-2]
                self.assertEqual(name, face)
                self.assertTrue(tolerance >= self.tolerance)
                self.assertEqual(len(box), 4)


if __name__ == "__main__":
    unittest.main()
