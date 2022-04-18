#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
from time import strftime
import cv2
import json


class Output():
    def __init__(self, filename='statistics.json'):
        self.stat_file = Path(filename)
        self.stat_file.touch(exist_ok=True)

        self.strangers_folder = Path('detected_strangers')
        if not self.strangers_folder.is_dir():
            try:
                self.strangers_folder.mkdir(parents=True)
            except FileExistsError:
                print(f'Can not create directory - "{self.strangers_folder}" file exists.')
                exit(-1)

    def update_statistics(self, person: str) -> None:
        with self.stat_file.open(mode='r+') as f:
            try:
                data = json.loads(f.read())
                f.seek(0)
            except:
                data = dict()
                data['face_detections'] = 0
                data['known_persons'] = 0
                data['strangers'] = 0

            if person != 'Stranger' and person != 'Unknown':
                data['face_detections'] += 1
                data['known_persons'] += 1
            elif person == 'Stranger':
                data['strangers'] += 1
                data['face_detections'] += 1
            elif person == 'Unknown':
                data['strangers'] += 1

            json_string = json.dumps(data, indent=4)
            f.write(json_string)

    def report_name(self, person: str) -> None:
        print(f'{person}')

    def proceed_attendance(self, person: str, frame):
        self.update_statistics(person)
        if person == 'Stranger' or person == 'Unknown':
            person = 'Stranger'
        self.report_name(person)

        if person == 'Stranger' and frame is not None and frame != []:
            now = datetime.now()
            date = now.strftime('%Y%m%d')
            time = now.strftime('%H_%M_%S_%f')

            path = self.strangers_folder.joinpath(date)
            if not path.is_dir():
                path.mkdir(parents=True)

            img_path = f'{path}/{time}.jpg'
            cv2.imwrite(img_path, frame)
            print(f'{img_path} saved.')

if __name__ == '__main__':
    out = Output()
    out.proceed_attendance('pi', None)
    out.proceed_attendance('Unknown', None)
    out.proceed_attendance('Stranger', None)
