import cv2
import numpy as np

class CubeGrasp():
    def __init__(self):
        self.gripped = False
        self.task_complete = False
        self.inital_image = None
        self.first_frame = True

    def reset(self):
        self.gripped = False
        self.task_complete = False
        self.inital_image = None
        self.first_frame = True

    def is_goal_achieved(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_initial = cv2.cvtColor(self.inital_image, cv2.COLOR_BGR2HSV)

        brown_lower = np.array([10, 100, 20], np.uint8)
        brown_upper = np.array([20, 255, 200], np.uint8)

        cyan_lower = np.array([80, 50, 50], np.uint8)
        cyan_upper = np.array([100, 255, 255], np.uint8)
        red_lower = np.array([0, 50, 50], np.uint8)
        red_upper = np.array([10, 255, 255], np.uint8)

        brown_mask = cv2.inRange(hsv_initial, brown_lower, brown_upper)
        cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        brown_contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cyan_contours, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_contour = max(brown_contours, key=cv2.contourArea) if brown_contours else None

        if box_contour is None:
            return False, "Brown box not found."

        box_x, box_y, box_w, box_h = cv2.boundingRect(box_contour)

        def is_inside_box(block_contour, box_x, box_y, box_w, box_h):
            x, y, w, h = cv2.boundingRect(block_contour)
            return (box_x < x < box_x + box_w) and (box_y < y < box_y + box_h)
        
        reward = 0
        complete = True

        for contour in cyan_contours:
            if not is_inside_box(contour, box_x, box_y, box_w, box_h):
                complete = False
            else:
                reward += 1

        for contour in red_contours:
            if not is_inside_box(contour, box_x, box_y, box_w, box_h):
                complete = False
            else:
                reward += 1

        return complete, reward, "All red and cyan blocks are inside the brown box."

    def reward(self, image_path):
        reward = 0
        
        image = cv2.imread(image_path)


        if self.first_frame:
            self.inital_image = image
            self.first_frame = False
        

        done, reward, _ = self.is_goal_achieved(image)
        if done and self.task_complete == False:
            self.task_complete = True

        return reward


