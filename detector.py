import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util


class DetectorTF2:
	def __init__(
		self,
		window_width,
		window_height,
		path_to_checkpoint, 
		path_to_labelmap, 
		class_id=None, 
		threshold=0.4, 
		critical_distance=0.3,
		epsilon=0.01
	):
		self.window_width = window_width
		self.window_height = window_height
		self.timeline_height = self.window_height - 25
		self.timeline_start = 50
		self.timeline_end = self.window_width - 50

		self.threshold = threshold
		self.critical_distance = critical_distance
		self.epsilon = epsilon

		self.prev_distance = 1

		self.trace = []
		self.detections = []
		self.warnings = []

		self.class_id = class_id
		label_map = label_map_util.load_labelmap(path_to_labelmap)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)

		tf.keras.backend.clear_session()
		self.detect_fn = tf.saved_model.load(path_to_checkpoint)

	def DetectFromImage(self, img):
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		input_tensor = np.expand_dims(img, 0)
		detections = self.detect_fn(input_tensor)

		bboxes = detections['detection_boxes'][0].numpy()
		bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
		bscores = detections['detection_scores'][0].numpy()
		det_boxes = self.ExtractBBoxes(bboxes, bclasses, bscores)

		return det_boxes

	def ExtractBBoxes(self, bboxes, bclasses, bscores):
		bboxes_filtered = []
		bclasses_filtered = []
		bscores_filtered = []
		for i in range(len(bboxes)):
			if (bclasses[i] not in self.class_id) and (self.class_id is not None):
				continue

			y_min = int(bboxes[i][0] * self.window_height)
			x_min = int(bboxes[i][1] * self.window_width)
			y_max = int(bboxes[i][2] * self.window_height)
			x_max = int(bboxes[i][3] * self.window_width)
			if ((x_max - x_min) * (y_max - y_min) / (self.window_width * self.window_height) > 0.4):
				continue
			if ((x_max - x_min) * (y_max - y_min) / (self.window_width * self.window_height) < 0.00235):
				continue
			if ((x_max - x_min) / self.window_width > 0.8):
				continue
			if ((y_max - y_min) / self.window_height > 0.7):
				continue

			bboxes_filtered.append(bboxes[i])
			bclasses_filtered.append(bclasses[i])
			bscores_filtered.append(bscores[i])

		indices = cv2.dnn.NMSBoxes(bboxes_filtered, bscores_filtered, self.threshold, 0.3)

		bbox = []
		for idx in indices:
			y_min = int(bboxes_filtered[idx][0] * self.window_height)
			x_min = int(bboxes_filtered[idx][1] * self.window_width)
			y_max = int(bboxes_filtered[idx][2] * self.window_height)
			x_max = int(bboxes_filtered[idx][3] * self.window_width)
			class_label = self.category_index[int(bclasses_filtered[idx])]['name']
			bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores_filtered[idx])])

		return bbox

	def DisplayDetections(self, img, boxes_list, video_duration, t, show_trace=False, step_between_traces=10, det_time=None):
		is_warning = False
		for idx in range(len(boxes_list)):
			x_min = boxes_list[idx][0]
			y_min = boxes_list[idx][1]
			x_max = boxes_list[idx][2]
			y_max = boxes_list[idx][3]
			cls = str(boxes_list[idx][4])
			score = str(np.round(boxes_list[idx][-1], 2))
			text = cls + ': ' + score
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
			cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

			if (show_trace):
				x_middle = int((x_min + x_max) / 2)
				y_middle = int((y_min + y_max) / 2)
				self.trace.append((x_middle, y_middle))

			if (is_warning):
				continue
			#if (cls != 'car') and (cls != 'bus') and (cls != 'truck'):
			# 	continue
			distance = round((1 - (x_max - x_min) / self.window_width) ** 4, 3)
			if (distance > self.critical_distance):
				continue
			relative_x_middle = round((x_min + x_max) / 2 / self.window_width, 2)
			if (relative_x_middle <= 0.25) or (relative_x_middle >= 0.75):
				continue
			if (distance + self.epsilon < self.prev_distance):
				is_warning = True
			self.prev_distance = distance

		if (show_trace):
			for i in range(0, len(self.trace), step_between_traces):
				cv2.rectangle(img, self.trace[i], self.trace[i], (0, 150, 0), 5)

		timeline_point = int(round(t * (self.timeline_end - self.timeline_start) / video_duration + self.timeline_start))

		# Обводка таймлайна (черный)
		cv2.rectangle(img, (self.timeline_start - 6, self.timeline_height - 3), (self.timeline_end + 6, self.timeline_height + 3), (0, 0, 0), 1)

		# Обнаружения (зеленый)
		if (len(boxes_list) > 0):
			self.detections.append(timeline_point)
		for i in range(len(self.detections)):
			cv2.rectangle(img, (self.detections[i] - 5, self.timeline_height - 10), (self.detections[i] + 5, self.timeline_height + 10), (0, 255, 0), -1)

		# Предупреждения (красный)
		if (is_warning):
			self.warnings.append(timeline_point)
			cv2.putText(img, 'Warning!', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
		for i in range(len(self.warnings)):
			cv2.rectangle(img, (self.warnings[i] - 5, self.timeline_height - 10), (self.warnings[i] + 5, self.timeline_height + 10), (0, 0, 255), -1)

		# Таймлайн (светло-серый)
		cv2.rectangle(img, (self.timeline_start - 5, self.timeline_height - 2), (self.timeline_end + 5, self.timeline_height + 2), (100, 100, 100), -1)

		# Курсор (темно-серый)
		cv2.rectangle(img, (timeline_point - 2, self.timeline_height - 8), (timeline_point + 2, self.timeline_height + 8), (30, 30, 30), -1)

		if (det_time != None):
			fps = str(round(1000 / det_time, 1)) + ' FPS'
			cv2.putText(img, fps, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

		return img