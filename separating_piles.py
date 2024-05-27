import cv2
import numpy as np

class MoveBlocks():

    def __init__(self):
        self.task_complete = False
        self.first_run = True
        self.yellow_square = None
        self.max_reward= 0
    def reset(self):
        self.task_complete = False
        self.first_run = True

        self.yellow_square = None
        self.max_reward = 0

    def find_blue_blocks(self, image):
        # Convert to HSV and threshold for blue color
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 20])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def identify_yellow_square_contours(self, image):
        # Convert to HSV and threshold for yellow color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Assuming the largest contour is the yellow square
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    def create_mask_from_contour(self, image_shape, contour):
        # Create an empty mask with the same dimensions as the image
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        # Fill the contour on the mask with white color
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        return mask

    def check_block_in_yellow_square(self, block_contour, yellow_square_mask):
        # Create a mask from the block contour
        block_mask = self.create_mask_from_contour(yellow_square_mask.shape, block_contour)
        # Perform a bitwise AND to see if the block is completely within the yellow square
        intersection = cv2.bitwise_and(block_mask, yellow_square_mask)
        return np.array_equal(intersection, block_mask)

    def check_goal_completion(self, blue_blocks_contours, yellow_square_mask):
        # Check if all blue blocks are completely within the yellow square mask
        for contour in blue_blocks_contours:
            if not self.check_block_in_yellow_square(contour, yellow_square_mask):
                return False
        return True

    def count_blue_blocks_in_yellow_square(self, blue_blocks_contours, yellow_square_mask):
        count = 0
        for contour in blue_blocks_contours:
            if self.check_block_in_yellow_square(contour, yellow_square_mask):
                count += 1
        return count


    def reward(self, image_path):

        image = cv2.imread(image_path)

        # Load the initial and updated images
        if self.first_run:
            contour = self.identify_yellow_square_contours(image)
            self.yellow_square = self.create_mask_from_contour(image.shape, contour)

            self.first_run = False

        # Identify blue blocks from the updated image
        block_contours = self.find_blue_blocks(image)
        reward = self.count_blue_blocks_in_yellow_square(block_contours, self.yellow_square)

        if reward >= self.max_reward:
            self.max_reward = reward

        return self.max_reward