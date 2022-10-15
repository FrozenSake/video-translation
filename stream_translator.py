# conda create -n cv2 python=3.9
# pip install numpy
# pip install opencv-python
# pip install ffpyplayer
# pip install pytesseract
# pip isntall textblob
# install tesseract https://github.com/tesseract-ocr/tesseract#installing-tesseract -> windows (https://github.com/UB-Mannheim/tesseract/wiki)
# Make sure you install the languages you need to use this on
# set pytesseract.pytesseract.tesseract_cmd = r'<fullpath_to_tesseract_executable>'

import time

import cv2
import numpy
import pytesseract

from textblob import TextBlob
from ffpyplayer.player import MediaPlayer

video_source = "./shenmue.mp4"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def PlayVideoCV2(video_source):
    cap = cv2.VideoCapture(video_source)
    player = MediaPlayer(video_source)
    start_time = time.time()

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # sleep_ms = int(numpy.round((1/fps)*1000))
        ret, frame = cap.read()
        if not ret:
            break

        if not player.get_frame():
            break

        elapsed = (time.time() - start_time) * 1000  # msec
        play_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        sleep = max(1, int(play_time - elapsed))

        if cv2.waitKey(sleep) & 0xFF == ord('q'):
            break

        pytess_config = r'--oem 3 --psm 6'
        output = pytesseract.image_to_string(frame, config=pytess_config)
        print(output)

        cv2.imshow('Frame', frame)

    cap.release()

    cv2.destroyAllWindows()


def PlayVideoMPlayer(video_source):
    player = MediaPlayer(video_source)
    player.set_size(720, 480)  # resize it
    #player.set_size(400, 300)

    start_time = time.time()
    frame_time = start_time + 0

    while True:
        current_time = time.time()

        # check if it is time to get next frame
        if current_time >= frame_time:

            # get next frame
            frame, val = player.get_frame()

            if val != 'eof' and frame is not None:
                image, pts = frame
                w, h = image.get_size()

                # convert to array width, height
                img = numpy.asarray(image.to_bytearray()[0]).reshape(h, w, 3)

                # convert RGB to BGR because `cv2` need it to display it
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                pytess_config = r'--oem 3 --psm 1'
                output = pytesseract.image_to_string(
                    img, config=pytess_config, lang="jpn")
                #output = TextBlob(output).translate(from_lang="jp", to="eng")
                print(output)

                cv2.imshow('video', img)

                frame_time = start_time + pts

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
    player.close_player()


# PlayVideoCV2(video_source)
PlayVideoMPlayer(video_source)
