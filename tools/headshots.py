#!/usr/bin/env python3
from pathlib import Path
import argparse
import cv2


def headshot(video_source: int, person: str, dataset_folder: str) -> None:
    path = Path(dataset_folder)
    if not path.is_dir():
        print(f'{dataset_folder} is not found.')
        print(f'Please run script from its directory ' \
            f'or create {dataset_folder} manually if it is not present.')
        exit(-1)
    path = path.joinpath(person)
    if not path.is_dir():
        path.mkdir(parents=True)
        img_counter = 0
    else:
        img_counter = len(list(path.glob('*')))

    cam = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cam.read()
        if not ret:
            print(f'Error {ret} while getting a frame.')
            break
        cv2.imshow(f'Take a photo', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 113:
            # esc or q button
            print("Quitting...")
            break
        elif key == 32:
            # space button
            img_path = f'{path}/image_{img_counter}.jpg'
            cv2.imwrite(img_path, frame)
            print(f'{img_path} saved.')
            img_counter += 1

    if not img_counter:
        path.rmdir()

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take dataset photos of new known persons",
        add_help=False)
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Press space to take a photo, escape or q to exit.')
    parser.add_argument('-c', '--camera', help='Camera video source', default=0)
    parser.add_argument('-p', '--person', help='Person name', required=True)
    parser.add_argument('-d', '--dataset', help='Target dataset folder', default='../dataset')
    args = parser.parse_args()
    camera = int(args.camera)
    person = args.person
    dataset = args.dataset
    headshot(camera, person, dataset)
