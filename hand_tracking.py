import cv2
import mediapipe as mp
import numpy as np
import math
import random

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ===============================
# Load Mediapipe Model
# ===============================

model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# ===============================
# Webcam
# ===============================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_count = 0


# ===============================
# Particle System
# ===============================

particles = []

def create_particles(center):

    x,y = center

    for _ in range(6):

        particles.append([
            x,
            y,
            random.uniform(-3,3),
            random.uniform(-3,3),
            random.randint(4,6)
        ])


def update_particles(frame):

    for p in particles:

        p[0]+=p[2]
        p[1]+=p[3]
        p[4]-=0.15

        if p[4]>0:
            cv2.circle(frame,(int(p[0]),int(p[1])),int(p[4]),(255,240,200),-1)

    particles[:] = [p for p in particles if p[4] > 0]


# ===============================
# Rasengan Renderer
# ===============================

def draw_rasengan(frame, center, frame_count):

    x,y = center

    overlay = frame.copy()

    # blue chakra sphere (smaller + lower opacity)
    cv2.circle(overlay,(x,y),60,(255,120,0),-1)
    cv2.addWeighted(overlay,0.35,frame,0.65,0,frame)

    # chakra core
    cv2.circle(frame,(x,y),30,(255,255,200),-1)
    cv2.circle(frame,(x,y),15,(255,255,255),-1)

    # vortex swirl (bigger)
    for i in range(3):

        angle_offset = frame_count*0.35 + i*2

        for t in np.linspace(0,5*np.pi,160):

            r = 10 + t*6
            angle = t + angle_offset

            px = int(x + r*np.cos(angle))
            py = int(y + r*np.sin(angle))

            if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                frame[py,px] = (255,220,140)

    create_particles(center)


# ===============================
# Hand Open Detection
# ===============================

def is_hand_open(points):

    fingers = 0

    if points[8][1] < points[6][1]:
        fingers +=1

    if points[12][1] < points[10][1]:
        fingers +=1

    if points[16][1] < points[14][1]:
        fingers +=1

    if points[20][1] < points[18][1]:
        fingers +=1

    if points[4][0] < points[3][0]:
        fingers +=1

    return fingers >= 4


# ===============================
# Palm Facing Camera Detection
# ===============================

def palm_facing_camera(points):

    thumb = points[4]
    pinky = points[20]

    return thumb[0] > pinky[0]


# ===============================
# Compute Palm Center
# ===============================

def get_palm_center(points):

    cx = int((points[0][0] + points[5][0] + points[17][0]) / 3)
    cy = int((points[0][1] + points[5][1] + points[17][1]) / 3)

    cy -= 20  # lift Rasengan slightly

    return (cx,cy)


# ===============================
# Main Loop
# ===============================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    frame_count +=1

    if result.hand_landmarks:

        h,w,_ = frame.shape

        for hand_landmarks in result.hand_landmarks:

            points=[]

            for lm in hand_landmarks:

                x = int(lm.x*w)
                y = int(lm.y*h)

                points.append((x,y))

                cv2.circle(frame,(x,y),4,(255,255,0),-1)

            for i in range(1,len(points)):
                cv2.line(frame,points[i-1],points[i],(0,255,255),2)

            if is_hand_open(points) and palm_facing_camera(points):

                palm = get_palm_center(points)

                draw_rasengan(frame,palm,frame_count)

    update_particles(frame)

    cv2.imshow("Anime Rasengan AI",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
