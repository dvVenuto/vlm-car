import cv2
import numpy as np
import matplotlib.pyplot as plt

class PandaGym():
    def __init__(self) -> None:
        self.task_1_complete = False
        self.first_frame = True
        self.goal_complete = False
        self.initial_image = None
        self.initial_goal_item_location = None
    def reset(self):
        self.task_1_complete = False
        self.first_frame = True
        self.goal_complete = False
        self.initial_image = None
        self.initial_goal_item_location = None

    def find_green_cube(self, image):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([45, 100, 50])  
        upper_green = np.array([75, 255, 255])  

        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            return (x, y, w, h), True
        else:
            return None, False


    def check_completion_of_task_1(self, initial_image, final_image, threshold=10):
        initial_cube_position, initial_found = self.find_green_cube_color_only(initial_image)

        if not initial_found:
            return False, "Green cube not found in the initial image."

        final_cube_position, final_found = self.find_green_cube_color_only(final_image)

        if not final_found:
            return False, "Green cube not found in the final image."

        initial_center = (initial_cube_position[0] + initial_cube_position[2] // 2, initial_cube_position[1] + initial_cube_position[3] // 2)
        final_center = (final_cube_position[0] + final_cube_position[2] // 2, final_cube_position[1] + final_cube_position[3] // 2)
        distance = np.linalg.norm(np.array(initial_center) - np.array(final_center))

        return distance > threshold, f"Cube moved by a distance of {distance:.2f} pixels."



    def find_yellow_cube_color_only(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])  
        upper_yellow = np.array([30, 255, 255])  

        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)

            return (x, y, w, h), True
        else:
            return None, False

    def check_completion_of_task_2(self, initial_image, final_image):
        yellow_area_position, yellow_found = self.find_yellow_cube_color_only(initial_image)

        if not yellow_found:
            return False, "Yellow area not found in the final image.", 0

        green_cube_position, green_found = self.find_green_cube_color_only(final_image)
        green_cube_initial_position, green_initial_found = self.find_green_cube_color_only(initial_image)

        if not green_found or not green_initial_found:
            return False, "Green cube not found in the final image.", 0

        initial_center = (green_cube_initial_position[0] + green_cube_initial_position[2] // 2, green_cube_initial_position[1] + green_cube_initial_position[3] // 2)
        final_center = (green_cube_position[0] + green_cube_position[2] // 2, green_cube_position[1] + green_cube_position[3] // 2)

        goal_center = (yellow_area_position[0] + yellow_area_position[2] // 2, yellow_area_position[1] + yellow_area_position[3] // 2)

        distance = np.linalg.norm(np.array(final_center) - np.array(goal_center))
        initial_distance = np.linalg.norm(np.array(initial_center) - np.array(goal_center))

        reward = (initial_distance - distance) / initial_distance

        green_center_x = green_cube_position[0] + green_cube_position[2] // 2
        green_center_y = green_cube_position[1] + green_cube_position[3] // 2

        within_bounds = (yellow_area_position[0] <= green_center_x <= yellow_area_position[0] + yellow_area_position[2] and
                        yellow_area_position[1] <= green_center_y <= yellow_area_position[1] + yellow_area_position[3])

        return within_bounds, "Green cube is within the yellow area." if within_bounds else "Green cube is not within the yellow area.", reward




    def reward(self, image_path):
        image = cv2.imread(image_path)

        reward = 0

        if self.first_frame:
            self.initial_image = image
            self.first_frame = False

        task1_completed, _ = self.check_completion_of_task_1(self.initial_image, image)

        goal_complete,_, reward_dist = self.check_completion_of_task_2(self.initial_image, image)

        reward = reward_dist
        if self.task_1_complete == False and task1_completed:
            reward = 1
            self.task_1_complete = True

        if self.task_1_complete == True and goal_complete == True and self.goal_complete == False:
            self.goal_complete = True

        return reward