import cv2
import mediapipe as mp
import math
import numpy as np
import imageio as iio

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

theta_space = np.linspace(0.0, 360.0, num=61)[:31]
phi_space = np.linspace(-90.0, 0.0, num=31)
radius_space = np.linspace(3.0, 5.0, num=5)

def find_nearest(theta, phi, radius):
    n_theta = theta_space[np.argmin(np.abs(theta_space - theta))]
    n_phi = phi_space[np.argmin(np.abs(phi_space - phi))]
    n_radius = radius_space[np.argmin(np.abs(radius_space - radius))]
    return n_theta, n_phi, n_radius

def low_pass_filter(current_value, previous_output, alpha):
    filtered_value = alpha * current_value + (1 - alpha) * previous_output
    return filtered_value

alpha = 0.2 
filt_theta = 0
filt_phi = 0
filt_radius = 0
frame_size = 800

# Create a new window with a specific size
window_name = 'Hand Tracking and Image'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 600)  # Adjust the window size as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, C = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[-1]
        for idx, landmark in enumerate(hand_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            if (idx == 4) or (idx == 8):
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        index4_x, index4_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
        index8_x, index8_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

        line_dist = math.sqrt((index4_x-index8_x)**2 + (index4_y-index8_y)**2)

        median_x = int((index4_x + index8_x)/2)
        median_y = int((index4_y + index8_y)/2)

        theta = ((median_x - 0.2*W) / (0.8*W - 0.2*W)) * 180
        theta = max(0, min(180, theta))
        phi = ((median_y - 0.2*H) / (0.8*H - 0.2*H)) * ((-90) - 0) + 0
        phi = max(-90, min(0, phi))
        radius = min(((line_dist/250 * 2.4) + 2.8), 5)

        filt_theta = low_pass_filter(theta, filt_theta, alpha)
        filt_phi = low_pass_filter(phi, filt_phi, alpha)
        filt_radius = low_pass_filter(radius, filt_radius, alpha)

        theta, phi, radius = find_nearest(filt_theta, filt_phi, filt_radius)

        filename = "t{:.1f}_p{:.1f}_r{:.1f}.png".format(theta, phi, radius)
        img = iio.imread("/Users/tylersmithline/Documents/UMich_Git_Local/ROB599-Group4-Final/TinyNeRF/Output_Images/" + filename) # set local image directory here

        # Convert the image to BGR color space (if it's not already in BGR)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw a line between landmarks 4 and 8
        cv2.line(frame, (index4_x, index4_y), (index8_x, index8_y), (255, 0, 0), 2)
        # Draw the median point (should lie on the center of the line)
        cv2.circle(frame, (median_x, median_y), 5, (0, 0, 255), -1)

        #Print theta, phi, line_dist values
        cv2.putText(frame, "Theta = " + str(theta), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2) #Display text -- Theta
        cv2.putText(frame, "Phi = " + str(phi), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2) #Display text -- Phi
        cv2.putText(frame, "Radius = " + str(radius), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2) #Display text -- Line_dist

        # Draw lines to show effective tracking area of screen. (Width: 0*W - 0.8*W) (Height: 0.2*H - 0.8*H)
        cv2.line(frame, (int(0.2*W), int(0.2*H)), (int(0.8*W), int(0.2*H)), (0, 0, 0), 2)     # top horizontal line    @ 0.2*H
        cv2.line(frame, (int(0.2*W), int(0.8*H)), (int(0.8*W), int(0.8*H)), (0, 0, 0), 2)     # bottom horizontal line @ 0.8*H
        cv2.line(frame, (int(0.2*W), int(0.2*H)), (int(0.2*W), int(0.8*H)), (0, 0, 0), 2)     # right vertical line    @ 0.8*W
        cv2.line(frame, (int(0.8*W), int(0.2*H)), (int(0.8*W), int(0.8*H)), (0, 0, 0), 2)     # right vertical line    @ 0.8*W

        # Resize the video frame and the image to fit side by side
        frame_resized = cv2.resize(frame, (frame_size, frame_size))  # Adjust the size as needed
        img_resized = cv2.resize(img, (frame_size, frame_size))  # Adjust the size as needed

        # Concatenate the resized video frame and the image horizontally
        concatenated_frame = cv2.hconcat([frame_resized, img_resized])

        # Display the concatenated frame in the window
        cv2.imshow(window_name, concatenated_frame)

    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        break

cap.release()
cv2.destroyAllWindows()