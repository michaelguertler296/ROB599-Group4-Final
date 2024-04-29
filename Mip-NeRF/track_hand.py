import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize VideoCapture
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    #check for successful frame grab
    if not ret:
        break

    # Flip the frame horizontally to revert it to its original orientation
    frame = cv2.flip(frame, 1)  #makes movement mirrored and not flipped
    H, W, C = frame.shape       #assume unchanged shape. Get frame sizes 

    # Convert the image to RGB --> MediaPipe needs this format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe
    results = hands.process(rgb_frame)

    # If hand(s) are detected, draw landmarks on the frame
    if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:    #this line annotates all visible hands. Indent lines below if kept
        hand_landmarks = results.multi_hand_landmarks[-1]         #ONLY detects/annotates the last hand. Keep either this or above line
        for idx, landmark in enumerate(hand_landmarks.landmark):
            # Convert normalized coordinates to pixel coordinates
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)

            # Draw a circle and number on each landmark
            #Only display dots (and text) for tip of index & thumb. 
            if (idx == 4) or (idx == 8):
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                # cv2.putText(frame, str(idx), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) #Display text to see which indexes needed

        # Get the pixel coordinates of landmarks 4 and 8
        index4_x, index4_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
        index8_x, index8_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

        # calculate length of line between 4,8
        line_dist = math.sqrt((index4_x-index8_x)**2 + (index4_y-index8_y)**2)  

        # Get median point of line between 4,8 --> point that will be tracked
        median_x = int((index4_x + index8_x)/2)
        median_y = int((index4_y + index8_y)/2)

        #Convert median point coordinates to theta, phi, radius values. Range values used from P4
        # theta = min(360, median_x/(0.8*W) * 360)    #Range [0, 360]. Mapping only about 80% of screen to this range, then maxing out at 360.
        theta = ((median_x - 0.2*W) / (0.8*W - 0.2*W)) * 360
        theta = max(0, min(360, theta))
        # phi = (median_y/H * ((-90) - 0)) + 0    # Range [-90, 0]. mapped_value = (x * (range_max - range_min)) + range_min. 
        phi = ((median_y - 0.2*H) / (0.8*H - 0.2*H)) * ((-90) - 0) + 0 # map y_position from 0.2*H-0.8*H to [-90, 0]
        phi = max(-90, min(0, phi))                                 # values < 0.2*H = -90. values > 0.8*H = 0. Prevent hand from leaving screen just to reach these limits
        radius = min( ((line_dist/250 * (5 - 3)) + 3), 5)      # ^^. Map to range [3, 5]

        # Draw a line between landmarks 4 and 8
        cv2.line(frame, (index4_x, index4_y), (index8_x, index8_y), (255, 0, 0), 2)
        # Draw the median point (should lie on the center of the line)
        cv2.circle(frame, (median_x, median_y), 5, (0, 0, 255), -1)

        #Print theta, phi, line_dist values
        cv2.putText(frame, "Theta = " + str(theta), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) #Display text -- Theta
        cv2.putText(frame, "Phi = " + str(phi), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) #Display text -- Phi
        cv2.putText(frame, "Radius = " + str(radius), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) #Display text -- Line_dist

        # Draw lines to show effective tracking area of screen. (Width: 0*W - 0.8*W) (Height: 0.2*H - 0.8*H)
        cv2.line(frame, (int(0.2*W), int(0.2*H)), (int(0.8*W), int(0.2*H)), (0, 0, 0), 2)     # top horizontal line    @ 0.2*H
        cv2.line(frame, (int(0.2*W), int(0.8*H)), (int(0.8*W), int(0.8*H)), (0, 0, 0), 2)     # bottom horizontal line @ 0.8*H
        cv2.line(frame, (int(0.2*W), int(0.2*H)), (int(0.2*W), int(0.8*H)), (0, 0, 0), 2)     # right vertical line    @ 0.8*W
        cv2.line(frame, (int(0.8*W), int(0.2*H)), (int(0.8*W), int(0.8*H)), (0, 0, 0), 2)     # right vertical line    @ 0.8*W


    # Display the annotated frame. 
    cv2.imshow('Hand Landmarks', frame)         #frame = annotated frame
    # cv2.imshow('Hand Landmarks', frame_orig)    #frame_orig = original un-annotated frame

    # Press 'esc' (keycode 27) or 'q' (ASCII keycode 113) to exit
    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
