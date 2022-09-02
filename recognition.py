from settings import *


class Recognition:
    """Recognizes the plant and outputs the information about the area"""
    plant_data = []
    class_ids = []
    confidences = []
    current_plant_data = []
    boxes = []

    def __init__(self, img, img_name):
        self.img = img
        self.img_name = img_name
        self.height, self.width, self.channels = img.shape

    def preprocess(self):
        PlantNet.setInput(cv2.dnn.blobFromImage(self.img, 0.00392, (416, 416), (0, 0, 0), True, crop=False))
        return PlantNet.forward(output_layers)

    def detection(self, outs):
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.75:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    self.boxes.append([x, y, w, h])
                    self.confidences.append(float(confidence))
                    self.class_ids.append(class_id)

    def not_detected(self):
        cv2.putText(self.img, 'No plant was detected', (0, self.height - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                    (255, 51, 51), 2)
        return self.img

    def do_detection(self):
        self.current_plant_data = []
        self.current_plant_data.append(self.img_name)
        processed_image = self.preprocess()
        self.detection(processed_image)
        indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
        if len(indexes) == 0:
            return self.not_detected()
        else:
            return self.detected(indexes)

    def detected(self, indexes):
        x, y, w, h = self.bounding_box(indexes)
        plant_mask = self.plant_mask(x, y, w, h)
        square_mask = self.square_mask()
        calculate_area = self.calculate_area(plant_mask, square_mask)
        self.add_text(calculate_area)
        return self.img

    def bounding_box(self, indexes):
        for i in range(len(self.boxes)):
            if i in indexes:
                x, y, w, h = self.boxes[i]
                label = str(classes[self.class_ids[i]])
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(self.img, label, (x, y - 100), font, 2, (255, 51, 51), 2)
                self.current_plant_data.append(label)
        return x, y, w, h

    def plant_mask(self, x, y, w, h):
        cropped_image = self.img[y:y + h, x:x + w]
        img_plant = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(img_plant)
        img_plant = cv2.subtract(B, A)
        _, thresh1 = cv2.threshold(img_plant, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh1, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        contours_plant, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_plant = 0
        for cnt in contours_plant:
            if cv2.contourArea(cnt) > 0:
                area_plant += cv2.contourArea(cnt)
        return area_plant

    def square_mask(self):
        img_square = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        mask_blue = cv2.inRange(img_square, lower_blue, upper_blue)
        contours_square, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area_square = 0
        for cnt in contours_square:
            area_square += cv2.contourArea(cnt)
        return area_square

    def calculate_area(self, plant_mask, square_mask):
        sqrcm_plant = round(plant_mask / square_mask, 2)
        sqrcm_text = f'''{self.img_name}
        Area of leaves: {sqrcm_plant}'''
        self.current_plant_data.append(sqrcm_plant)
        return sqrcm_text

    def add_text(self, calculate_area):
        cv2.putText(self.img, calculate_area, (10, 100), font, 2, (255, 51, 51), 3, 2)
        self.plant_data.append(self.current_plant_data)