import pathlib

from .face_alignment import image_align
from .landmarks_detector import LandmarksDetector

landmarks_detector = LandmarksDetector("data/shape_predictor_68_face_landmarks.dat")


def align_images(image_file_name):
    for face_landmarks in landmarks_detector.get_landmarks(image_file_name):
        aligned_file_name = str(pathlib.PurePath(image_file_name).with_suffix('')) + '_aligned.png'
        image_align(image_file_name, aligned_file_name, face_landmarks)

        return aligned_file_name
