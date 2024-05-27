import cv2
import numpy as np


class OpenDoor():
    def __init__(self):
        self.task_1_done = False
        self.task_2_done = False

        self.first_frame = False
        self.initial_image = None

    def reset(self):
        self.task_1_done = False
        self.task_2_done = False

        self.first_frame = False
        self.initial_image = None

    def check_key_disappeared(self, image):
        _, key_found = self.find_key_corrected(image)

        return not key_found

    def find_key_corrected(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) > 4:
                x, y, w, h = cv2.boundingRect(cnt)
                return (x + w // 2, y + h // 2), True

        return (0, 0), False
    def find_agent(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        lower_red = np.array([150, 0, 0])
        upper_red = np.array([255, 100, 100])

        mask = cv2.inRange(image_rgb, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        def is_triangle(c):
            peri = cv2.arcLength(c, True)
            vertices = cv2.approxPolyDP(c, 0.04 * peri, True)
            return len(vertices) == 3

        for cnt in contours:
            if is_triangle(cnt):
                x, y, w, h = cv2.boundingRect(cnt)
                return (x + w//2, y + h//2), True

        return (0, 0), False

    def find_door(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])  # HSV values for yellow
        upper_yellow = np.array([30, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)

                if w * h > 100:  
                    return (x + w // 2, y + h // 2), True

        return (0, 0), False
    def check_task_2_completion(self, initial_image, current_image):
        initial_gray = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

        door_location, door_found = self.find_door(initial_image_path)
        if not door_found:
            return False

        x, y, w, h = cv2.boundingRect(cv2.findContours(cv2.inRange(cv2.cvtColor(initial_image, cv2.COLOR_BGR2HSV),
                                                                   np.array([20, 100, 100]),
                                                                   np.array([30, 255, 255])),
                                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0])
        initial_roi = initial_gray[y:y+h, x:x+w]
        current_roi = current_gray[y:y+h, x:x+w]

        difference = cv2.absdiff(initial_roi, current_roi)
        _, threshold = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

        return cv2.countNonZero(threshold) > (w * h * 0.5) 



    def check_and_progress(self, image_path):
        image = cv2.imread(image_path)

        if self.initial_image:
            self.initial_image = image
            self.first_frame = False

        _, key_gone = self.check_key_disappeared(image)
        door_open = self.check_task_2_completion(image)

        reward = 0

        if self.task_1_done == False and key_gone == True:
            reward = 1
            self.task_1_done = True

        if self.task_1_done == True and self.task_2_done == False and door_open == True:
            self.task_2_done = True

        return reward

