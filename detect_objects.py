import os
import cv2
import time
import argparse
from detector import DetectorTF2


def DetectFromVideo(detector, video_path, show_trace=False, step_between_traces=10, save_output=False, output_dir='output/'):
	cap = cv2.VideoCapture(video_path)
	video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * 1000

	if save_output:
		output_path = os.path.join(output_dir, 'out.mp4')
		out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (detector.window_width, detector.window_height))
		frames = []

	while (cap.isOpened()):
		ret, img = cap.read()
		if not ret: break
		img = cv2.resize(img, (detector.window_width, detector.window_height))

		timestamp1 = time.time()
		det_boxes = detector.DetectFromImage(img)
		elapsed_time = round((time.time() - timestamp1) * 1000)
		t = cap.get(cv2.CAP_PROP_POS_MSEC)
		img = detector.DisplayDetections(img, det_boxes, video_duration, t, show_trace, step_between_traces, det_time=elapsed_time)

		cv2.imshow('TF2 Detection', img)

		if cv2.waitKey(1) == 27: break

		if save_output:
			frames.append((img, t))
			#out.write(img)

	cap.release()
	if save_output:
		frames.reverse()
		while (len(frames) > 0):
			img, t = frames.pop()

			for i in range(len(detector.detections)):
				cv2.rectangle(img, (detector.detections[i] - 5, detector.timeline_height - 10), (detector.detections[i] + 5, detector.timeline_height + 10), (0, 255, 0), -1)

			for i in range(len(detector.warnings)):
				cv2.rectangle(img, (detector.warnings[i] - 5, detector.timeline_height - 10), (detector.warnings[i] + 5, detector.timeline_height + 10), (0, 0, 255), -1)

			cv2.rectangle(img, (detector.timeline_start - 5, detector.timeline_height - 2), (detector.timeline_end + 5, detector.timeline_height + 2), (100, 100, 100), -1)

			timeline_point = int(round(t * (detector.timeline_end - detector.timeline_start) / video_duration + detector.timeline_start))
			cv2.rectangle(img, (timeline_point - 2, detector.timeline_height - 8), (timeline_point + 2, detector.timeline_height + 8), (30, 30, 30), -1)

			out.write(img)
		out.release()

def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):
	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')):
			image_path = os.path.join(images_dir, file.name)
			print(image_path)
			img = cv2.imread(image_path)
			det_boxes = detector.DetectFromImage(img)
			img = detector.DisplayDetections(img, det_boxes)

			cv2.imshow('TF2 Detection', img)
			cv2.waitKey(0)

			if save_output:
				img_out = os.path.join(output_dir, file.name)
				cv2.imwrite(img_out, img)


if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description='Object Detection from Images or Video')
	parser.add_argument('--model_path', help='Path to frozen detection model')
	parser.add_argument('--path_to_labelmap', help='Path to labelmap (.pbtxt) file', default='models/mscoco_label_map.pbtxt')
	parser.add_argument('--class_ids', help='id of classes to detect, expects string with ids delimited by ,', type=str, default='3,6,8')
	parser.add_argument('--threshold', help='Detection threshold', type=float, default=0.4)
	parser.add_argument('--critical_distance', help='Minimum distance for alarm', type=float, default=0.3)
	parser.add_argument('--epsilon', help='Threshold value for comparing distances', type=float, default=0.07)
	parser.add_argument('--show_trace', help='Flag for object`s trace visualization, default: False', action='store_true')
	parser.add_argument('--step_between_traces', help='Number of traces not shown between each trace visualization', type=int, default=10)
	parser.add_argument('--images_dir', help='Directory to input images)', default='images/')
	parser.add_argument('--video_path', help='Path to input video')
	parser.add_argument('--output_directory', help='Path to output images and video', default='output/')
	parser.add_argument('--video_input', help='Flag for video input, default: False', action='store_true')
	parser.add_argument('--save_output', help='Flag for save images and video with detections visualized, default: False', action='store_true')
	args = parser.parse_args()

	id_list = None
	if args.class_ids is not None:
		id_list = [int(item) for item in args.class_ids.split(',')]

	if args.save_output:
		if not os.path.exists(args.output_directory):
			os.makedirs(args.output_directory)

	detector = DetectorTF2(window_width=1280, window_height=720, path_to_checkpoint=args.model_path, path_to_labelmap=args.path_to_labelmap, class_id=id_list, threshold=args.threshold)

	if args.video_input:
		DetectFromVideo(detector, args.video_path, args.show_trace, args.step_between_traces, save_output=args.save_output, output_dir=args.output_directory)
	else:
		DetectImagesFromFolder(detector, args.images_dir, save_output=args.save_output, output_dir=args.output_directory)

	print('Done..')
	cv2.destroyAllWindows()
