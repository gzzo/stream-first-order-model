import streamlit as st
import cv2
import torch
import imageio
import os
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import subprocess
import gdown
import bz2

from utils import align_images
from utils import load_checkpoints
from animate import normalize_kp

MEDIA_WIDTH = 300

os.makedirs("tmp/frames", exist_ok=True)
os.makedirs("data", exist_ok=True)


def fetch_landmarks():
    output_path = 'data/shape_predictor_68_face_landmarks.dat.bz2'
    extracted_path = output_path[:-4]

    if os.path.exists(extracted_path):
        return

    file_location = 'https://drive.google.com/uc?id=1ONAUiyUsngDOBQFwmnqtxC2uFWlId0n2'
    output_path = 'data/shape_predictor_68_face_landmarks.dat.bz2'
    gdown.download(file_location, output_path)

    data = bz2.BZ2File(output_path).read()
    with open(extracted_path, 'wb') as fp:
        fp.write(data)


def fetch_vox():
    output_path = 'data/vox-cpk.pth.tar'

    if os.path.exists(output_path):
        return

    file_location = 'https://drive.google.com/uc?id=1jvzD5ef6mPHRpISATDLqDMn90V-qNkV0'
    gdown.download(file_location, output_path)


@st.cache
def cached_align_image(photo_file_name):
    return align_images.align_images(photo_file_name)


@st.cache
def make_animation(source_image, driving_video, relative=True, adapt_movement_scale=True, cpu=False):
    generator, kp_detector = load_checkpoints.load_checkpoints(config_path='config/vox-256.yaml',
                                              checkpoint_path='data/vox-cpk.pth.tar', cpu=True)

    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in range(driving.shape[2]):
            progress.progress(frame_idx / float(driving.shape[2]))

            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


uploaded_video = st.file_uploader('Video', type=['mp4'])
video_file_name = 'tmp/video.mp4'
if uploaded_video is not None:
    with open(video_file_name, 'wb+') as video_file:
        video_file.write(uploaded_video.getbuffer())

    st.video(uploaded_video)


uploaded_photo = st.file_uploader('Photo', type=['jpg'])
photo_file_name = 'tmp/photo.jpg'


if uploaded_photo is not None:
    with open(photo_file_name, 'wb+') as photo_file:
        photo_file.write(uploaded_photo.getbuffer())

    aligned_file_name = cached_align_image(photo_file_name)
    st.image(uploaded_photo, caption='Original photo', width=MEDIA_WIDTH)
    st.image(aligned_file_name, caption='Aligned photo', width=MEDIA_WIDTH)


if uploaded_photo is not None and uploaded_video is not None:
    fps_of_video = int(cv2.VideoCapture(video_file_name).get(cv2.CAP_PROP_FPS))
    frames_of_video = int(cv2.VideoCapture(video_file_name).get(cv2.CAP_PROP_FRAME_COUNT))

    source_image = imageio.imread(aligned_file_name)
    source_image = resize(source_image, (256, 256))[..., :3]

    driving_video = imageio.mimread(video_file_name, memtest=False)
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    if st.button('Start!'):
        progress = st.progress(0)

        predictions = make_animation(source_image, driving_video, relative=True, cpu=True)
        video = [img_as_ubyte(frame) for frame in predictions]

        generated_file_name = 'tmp/generated.mp4'
        imageio.mimsave(generated_file_name, video)

        vidcap = cv2.VideoCapture(generated_file_name)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("tmp/frames/frame%09d.jpg" % count, image)
            success, image = vidcap.read()
            count += 1

        frames = []
        img = os.listdir("tmp/frames/")
        img.sort()
        for frame in img:
            frames.append(imageio.imread("tmp/frames/" + frame))
        frames = np.array(frames)
        imageio.mimsave("tmp/final.mp4", frames, fps=fps_of_video)

        subprocess.check_call('ffmpeg -y -i tmp/video.mp4 -vn -ar 44100 -ac 2 -ab 192K -f mp3 tmp/sound.mp3', shell=True)
        subprocess.check_call('ffmpeg -y -i tmp/sound.mp3 -i tmp/final.mp4 tmp/final_audio.mp4', shell=True)

        st.video('tmp/final_audio.mp4')
