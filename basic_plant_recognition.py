from settings import *

class Basic_plant_recognition:
    """Recognizes the plant and outputs the information about the area"""
    plant_data = []
    current_plant_data = []

    def __init__(self, img, img_name):
        self.img = img
        self.img_name = img_name
        self.height, self.width, _ = img.shape

    def detect_plant_contours(self):
        plant_mask = cv2.inRange(self.img, lower_green, upper_green)
        plant_contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return plant_contours

    def do_detection(self):
        self.current_plant_data = []
        self.current_plant_data.append(self.img_name)
        plant_contours = self.detect_plant_contours()
        return self.detected(plant_contours)

    def detected(self, plant_contours):
        area_plant = self.area_plant_calculation(plant_contours)
        square_mask = self.square_mask()
        self.detect_color(plant_contours)
        convex_hull_area = self.convex_hull(plant_contours)
        area_text = self.calculate_area(convex_hull_area, area_plant, square_mask)
        self.add_text(area_text)
        return self.img

    def area_plant_calculation(self, processed_image):
        area_plant = 0
        for cnt in processed_image:
            if cv2.contourArea(cnt) > 0:
                area_plant += cv2.contourArea(cnt)
        return area_plant

    def square_mask(self):
        mask_blue = cv2.inRange(self.img, lower_blue_v2, upper_blue_v2)
        contours_square, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area_square = 0
        for cnt in contours_square:
            area_square += cv2.contourArea(cnt)
        return area_square

    def detect_color(self, contours_plant):

        mask = np.zeros(self.img.shape[:2], np.uint8)
        cv2.drawContours(mask, contours_plant, -1, 255, -1)
        mean = cv2.mean(self.img, mask=mask)

        self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2RGB)
        cv2.drawContours(self.img, contours_plant, -1, (0, 255, 0), 3)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)

        try:
            plant_color = webcolors.hex_to_name(webcolors.rgb_to_name((int(mean[0]), int(mean[1]), int(mean[2]))))
        except ValueError:
            orig = (int(mean[0]), int(mean[1]), int(mean[2]))
            similarity = {}
            for hex_code, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
                approx = webcolors.hex_to_rgb(hex_code)
                similarity[color_name] = sum(np.subtract(orig, approx) ** 2)
            plant_color = min(similarity, key=similarity.get)

        self.current_plant_data.append(plant_color)

    def convex_hull(self, plant_contours):
        hull_list = []
        for i in range(len(plant_contours)):
            hull = cv2.convexHull(plant_contours[i])
            hull_list.append(hull)
        drawing = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
        cropped_image = cv2.fillPoly(drawing, hull_list, (255, 255, 255))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        convex_hull_area = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 0:
                convex_hull_area += cv2.contourArea(cnt)
        return convex_hull_area

    def calculate_area(self, convex_hull_area, area_plant, square_mask):
        area_plant_cm = round(area_plant / square_mask, 2)
        convex_hull_area_cm = round(convex_hull_area / square_mask, 2)
        self.current_plant_data.append(area_plant_cm)
        self.current_plant_data.append(convex_hull_area_cm)
        return f'Plant area {area_plant_cm}, convex hull area {convex_hull_area_cm}'

    def add_text(self, area_text):
        cv2.putText(self.img, self.img_name, (10, 100), font, 1, (255, 51, 51), 2, 2)
        cv2.putText(self.img, str(area_text), (10, 200), font, 1, (255, 51, 51), 2, 2)
        self.plant_data.append(self.current_plant_data)
