import argparse
from logging import basicConfig, DEBUG, getLogger

from .video_detector import RealtimeVideoDetector

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

    basicConfig(
        level=DEBUG,
        format='%(asctime)s %(name)s %(levelname)s -- %(message)s'
        )
    log = getLogger()

    # MobileNet pretrained model
    proto_file = r'models/mobilenet.prototxt'
    model_file = r'models/mobilenet.caffemodel'
    proc_frame_size = 300
    person_class = 15

    video_detector = RealtimeVideoDetector(log, proto_file, model_file, proc_frame_size)
    video_detector.detect(video, person_class, tolerance)
