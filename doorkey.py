import cv2
import numpy as np


class DoorKey8x8():
    def __init__(self):
        self.task_1_done = False
        self.task_2_done = False
        self.task_3_done = False

        self.first_frame = False
        self.door_contour = None

    def find_agent_and_key(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        agent_position = None
        for contour in contours_red:
            if cv2.contourArea(contour) > 100:  
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    agent_position = (cx, cy)
                    break  

        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        key_position = None
        for contour in contours_yellow:
            if 100 < cv2.contourArea(contour) < 1000:  
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) > 4:  
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        key_position = (cx, cy)
                        break
        if key_position == None:
            key_not_found = True
        else:
            key_not_found = False

        return agent_position, key_position, key_not_found

    def find_door_by_shape_and_color(self, image, agent_position):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_color_bound = np.array([20, 100, 100])
        upper_color_bound = np.array([30, 255, 255])

        color_mask = cv2.inRange(hsv_image, lower_color_bound, upper_color_bound)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        door_contour = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            if len(approx) == 4:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    door_position = (cx, cy)
                    door_contour = contour

        return True, door_contour, door_position
    


    def check_door_opened(self, door_contour, threshold=10):
        _, _, w, h = cv2.boundingRect(door_contour)
        diff = h - w
        if diff > threshold:
            return True
        else:
            return False

    def check_goal(self, image):
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])

        hsv_goal = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv_goal, lower_green, upper_green)

        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        goal_position = None
        max_area = 0
        for contour in contours_green:
            area = cv2.contourArea(contour)
            if area > max_area and area > 500:  
                max_area = area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    goal_position = (cx, cy)

        return goal_position




    def check_and_progress(self, image_path):
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (1000, 1000))

        reward = 0

        agent_position, key_position, key_not_found = self.find_agent_and_key(frame)

        if key_not_found:
            self.task_1_done = True
            reward = 0.5
        else:
            self.task_1_done = False


        if self.task_1_done and self.task_2_done == False:
            _, door_contour, _ = self.find_door_by_shape_and_color(frame, agent_position)
            self.task_2_done = self.check_door_opened(door_contour)
            reward = 0.5

        if self.task_1_done and self.task_2_done and self.task_3_done == False:
            goal_position = self.check_goal(frame)
        
        return reward







