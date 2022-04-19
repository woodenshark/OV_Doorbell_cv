#!/usr/bin/env python3
from pathlib import Path
import face_recognition
import argparse
import pickle
import cv2
import os


def training(dataset_folder: str, output_file: str) -> None:
    path = Path(dataset_folder)
    if not path.is_dir():
        print(f'{dataset_folder} is not found.')
        exit(-1)
    img_paths = tuple(path.rglob('*.jpg'))

    print(f'Start processing faces...')
    known_encodings = []
    known_names = []

    for (i, img_path) in enumerate(img_paths):
        # extract the person name from the image path
        print(f'Processing image {i + 1}/{len(img_paths)}')
        name = str(img_path).split(os.path.sep)[-2]

        # convert image from RGB (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(img_path.as_posix())
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes for each face
        boxes = face_recognition.face_locations(rgb, model='hog')

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding to set of known names and encodings
            known_encodings.append(encoding)
            known_names.append(name)

    print('Serializing encodings...')
    data = {
        'encodings': known_encodings,
        'names': known_names
    }
    with open(output_file, 'wb+') as f:
        f.write(pickle.dumps(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model for known faces dataset",
        add_help=False)
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Provide args for custom dataset folder or output file.')
    parser.add_argument('-o', '--output', help='Target output file', default='../encodings.pickle')
    parser.add_argument('-d', '--dataset', help='Source dataset folder', default='../dataset')
    args = parser.parse_args()
    output = args.output
    dataset = args.dataset
    training(dataset, output)
