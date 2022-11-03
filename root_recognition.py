from settings import *

class Root_recognition:
    """Recognizes the root and outputs the information about height and width"""
    root_data = []
    current_root_data = []

    def __init__(self, img, img_name):
        self.img = img
        self.img_name = img_name
        self.height, self.width, _ = img.shape

    def detect_root_contours(self):
        root_mask = cv2.inRange(self.img, lower_red, upper_red)
        contours, _ = cv2.findContours(root_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def do_detection(self):
        self.current_root_data = []
        self.current_root_data.append(self.img_name)
        root_contours = self.detect_root_contours()
        return self.detected(root_contours)

    def detected(self, root_contours):
        root_height_width = self.root_height_width_calculation(root_contours)
        square_mask = self.square_mask()
        area_text = self.calculations(root_height_width, square_mask)
        self.add_text(area_text)
        return self.img

    def root_height_width_calculation(self, root_contours):
        cnt = max(root_contours, key=len)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        #cv2.polylines(root_contours, [box], True, (255, 0, 0), 2)
        return (w, h)

    def square_mask(self):
        blue_square = cv2.inRange(self.img, lower_blue_v3, upper_blue_v3)
        contours, _ = cv2.findContours(blue_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=len)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        return (w, h)

    def calculations(self, root_height_width, square_mask):
        height = root_height_width[1]/square_mask[1]
        width = root_height_width[0]/square_mask[0]
        self.current_root_data.append(height)
        self.current_root_data.append(width)
        return f'Height {height}, width {width}'

    def add_text(self, area_text):
        cv2.putText(self.img, self.img_name, (10, 100), font, 1, (255, 51, 51), 2, 2)
        cv2.putText(self.img, str(area_text), (10, 200), font, 1, (255, 51, 51), 2, 2)
        self.root_data.append(self.current_root_data)