import cv2
import numpy as np

class BowlTask():

    def __init__(self):
        self.first_frame = True
        self.initial_image = None

    def reset(self):
        self.first_frame = True
        self.initial_image = None

    def find_red_cubes(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        red_cubes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    red_cubes.append((cx, cy))
        return red_cubes, bool(red_cubes)

    def find_green_containers(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        refined_contours = []
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
            area = cv2.contourArea(approx)
            perimeter = cv2.arcLength(approx, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            
            if 750 < area < 5000 and circularity > 0.8:  
                refined_contours.append(approx)

        return refined_contours

    def check_goal_completion(self, current_frame_image, initial_image):
        red_cubes_locations, _ = self.find_red_cubes(current_frame_image)
        
        initial_frame_contours = self.find_green_containers(initial_image)
        
        reward = 0
        
        for contour in initial_frame_contours:
            for cube_center in red_cubes_locations:
                is_inside = cv2.pointPolygonTest(contour, cube_center, False)
                if is_inside >= 0:
                    reward += 1

        return reward  




    def reward(self, image_path):
        image = cv2.imread(image_path)

        reward = 0
        if self.first_frame:
            self.initial_image = image
            self.first_frame = False

        reward = self.check_goal_completion(image, self.initial_image)


        return reward