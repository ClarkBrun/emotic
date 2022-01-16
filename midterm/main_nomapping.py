import argparse 
import tensorflow as tf
import numpy as np
import yolo_inference
from yolo_inference import get_bbox
import cv2
import os

import torch
from torchvision import transforms

from yolo_utils import prepare_yolo, rescale_boxes, non_max_suppression

from inference import process_images, infer, inference_emotic

from emotic import Emotic 

from model import predict, image_to_tensor, deepnn

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--experiment_path', type=str, default='./debug', help='Path of experiment files (results, models, logs)') # ,required=True
	parser.add_argument('--model_dir', type=str, default='models', help='Folder to access the models')
	parser.add_argument('--video_model_dir', type=str, default='./ckpt', help='Path to model file.')
	parser.add_argument('--result_dir', type=str, default='results', help='Path to save the results')
	parser.add_argument('--inference_file', type=str, default='faces_list.txt', help='Text file containing image context paths and bounding box')
	parser.add_argument('--video_file', type=str, default='Recoreded.mp4', help='Test video file')
	parser.add_argument('--mode', type=str, default = 'image', help='To choose the mode to detect image or video')
	parser.add_argument('--emojis_path', type=str, default = './data/emojis_Large/', help='To find to emoji pictures')
	parser.add_argument('--showbox', type=bool, default=True, help='To show bounding_boxes of faces')
	# Generate args
	args = parser.parse_args()
	return args

def format_image(image):
	if len(image.shape) > 2 and image.shape[2] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = cascade_classifier.detectMultiScale(
		image,
		scaleFactor = 1.3,
		minNeighbors = 5
  	)
	# None is no face found in image
	if not len(faces) > 0:
		return None, None
	max_are_face = faces[0]
	for face in faces:
		if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
			max_are_face = face
	# face to image
	face_coor =  max_are_face
	image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
	# Resize image to network size
	try:
		image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
	except Exception:
		print("[+} Problem during resize")
		return None, None
	return  image, face_coor

def get_bbox(args, image_context, model_path, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):
	device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
	yolo_model = prepare_yolo(model_path)
	yolo_model = yolo_model.to(device)
	yolo_model.eval()
	
	test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
	image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

	with torch.no_grad():
		detections = yolo_model(image_yolo)
		nms_det  = non_max_suppression(detections, conf_thresh, nms_thresh)[0]
		det = rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))

	bboxes = []

	for x1, y1, x2, y2, _, _, cls_pred in det:
		if cls_pred == 0:  # checking if predicted_class = persons. 
			x1 = int(min(image_context.shape[1], max(0, x1)))
			x2 = int(min(image_context.shape[1], max(x1, x2)))
			y1 = int(min(image_context.shape[0], max(15, y1)))
			y2 = int(min(image_context.shape[0], max(y1, y2)))
			bboxes.append([x1, y1, x2, y2])

	return np.array(bboxes)

def main(args, EMOTIONS, CAT, EMO2CAT, model_path, result_path, faces_path, context_norm, body_norm, ind2cat, ind2vad):
	catograry = None
	device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")

	thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy'))).to(device) 
	model_context = torch.load(os.path.join(model_path,'model_context1.pth')).to(device)
	model_body = torch.load(os.path.join(model_path,'model_body1.pth')).to(device)
	emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)
	models = [model_context, model_body, emotic_model]

	emoji_faces = []
	for index, emotion in enumerate(CAT):
		emoji = cv2.imread(args.emojis_path + emotion + '.png', -1)
		emoji = cv2.resize(emoji, (600, 600), interpolation = cv2.INTER_CUBIC)
		emoji_faces.append(emoji)
	
	if args.mode == 'image':
		with open(faces_path, 'r') as f:
			lines = f.readlines()
		for idx, line in enumerate(lines):
			image_context_path = line.split('\n')[0].split(' ')[0]
			image_name = image_context_path.split('/')[-1]
			# print("image_context_path", image_context_path)
			# exit(10)
			org_frame = cv2.imread(image_context_path)
			if org_frame is None:
				print('None, cannot reach the specified pictures')
				exit(100)
			else:
				frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2RGB)
				cv2.imshow('frame', org_frame)
				cv2.waitKey(2000)
				cv2.destroyAllWindows()

				bboxes = get_bbox(args, frame, model_path=model_path)

				'''
				only select the first face in a picture
				'''
				print(bboxes[0])
				catograry = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context=frame, bbox=bboxes[0], to_print=True)
				if catograry is not None:
						index = CAT.index(catograry)
						emoji_face = emoji_faces[index]
						for c in range(0, 3):
							# org_frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + org_frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
							org_frame[200:800, 200:800, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + org_frame[200:800, 200:800, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
						
						cv2.imshow('frame_emoji', org_frame)
						cv2.waitKey(2000)
						cv2.destroyAllWindows()
						
						saved_path = "./data/output/SavedImage-" + image_name + ".jpg"
						cv2.imwrite(saved_path, org_frame)

						print("Done~")
				else:
					print("Cannot detect any emotions~")
				
				'''
				detect several faces in a picture
				'''

				# for boxid in range(len(bboxes)):
				# 	print(bboxes[boxid])
				# 	catograry = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context=frame, bbox=bboxes[boxid], to_print=True)
				
				# 	if catograry is not None:
				# 		index = CAT.index(catograry)
				# 		emoji_face = emoji_faces[index]
				# 		for c in range(0, 3):
				# 			# org_frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + org_frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
				# 			org_frame[200:800, 200:800, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + org_frame[200:800, 200:800, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
						
				# 		cv2.imshow('frame_emoji', org_frame)
				# 		cv2.waitKey(2000)
				# 		cv2.destroyAllWindows()
						
				# 		saved_path = "SavedImage" + str(boxid) + ".jpg"
				# 		cv2.imwrite(saved_path, org_frame)

				# 		print("Done~")
				# 	else:
				# 		print("Cannot detect any emotions~")
		f.close()
					
	elif args.mode == 'video':
		print('Testing on video')

		face_x = tf.placeholder(tf.float32, [None, 2304])
		y_conv = deepnn(face_x)
		probs = tf.nn.softmax(y_conv)

		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(args.video_model_dir)
		sess = tf.Session()
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')

		emoji_faces = []
		for index, emotion in enumerate(EMOTIONS):
			emoji = cv2.imread('./data/emojis_Small/' + emotion + '.png', -1)
			emoji = cv2.resize(emoji, (600, 600), interpolation = cv2.INTER_CUBIC)
			emoji_faces.append(emoji)

		video_captor = cv2.VideoCapture(0) # using the inner webcam

		while True:
			ret, frame = video_captor.read()
			detected_face, face_coor = format_image(frame)
			if args.showBox:
				if face_coor is not None:
					[x,y,w,h] = face_coor
					cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

			if cv2.waitKey(1) & 0xFF == ord('f'):
				if detected_face is not None:
					tensor = image_to_tensor(detected_face)
					result = sess.run(probs, feed_dict={face_x: tensor})
				if result is not None:
					for index, emotion in enumerate(EMOTIONS):
						cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
						cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
										(255, 0, 0), -1)
						emoji_face = emoji_faces[np.argmax(result[0])]

				for c in range(0, 3):
					frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
				cv2.imshow('face', frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			

if __name__=='__main__':
	args = parse_args()

	model_path = os.path.join(args.experiment_path, args.model_dir)
	result_path = os.path.join(args.experiment_path, args.result_dir)
	if args.mode == 'image':
		faces_path = os.path.join('./', args.inference_file)
		# print('the image path is:', faces_path)
		# exit(10)
	else:
		faces_path = None

	CAT = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
			'Disquietment', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
			'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'] # 'Doubt/Confusion', 
	
	EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']


	EMO2CAT = {
		'angry' : ['Anger'],
		'disgusted' : ['Annoyance', 'Aversion', 'Disapproval', 'Esteem', 'Sensitivity'],
		'fearful' : ['Disquietment', 'Fear'],
		'sad' : ['Disconnection', 'Doubt/Confusion', 'Embarrassment', 'Fatigue', 'Pain', 'Sadness', 'Suffering'],
		'happy': ['Affection', 'Excitement', 'Happiness', 'Pleasure'],
		'surprised' : ['Anticipation', 'Surprise'],
		'neutral' : ['Confidence', 'Peace', 'Sympathy', 'Yearning']
	}

	cat2ind = {}
	ind2cat = {}
	for idx, emotion in enumerate(CAT):
		cat2ind[emotion] = idx
		ind2cat[idx] = emotion

	vad = ['Valence', 'Arousal', 'Dominance']
	ind2vad = {}
	for idx, continuous in enumerate(vad):
		ind2vad[idx] = continuous

	context_mean = [0.4690646, 0.4407227, 0.40508908]
	context_std = [0.2514227, 0.24312855, 0.24266963]
	body_mean = [0.43832874, 0.3964344, 0.3706214]
	body_std = [0.24784276, 0.23621225, 0.2323653]
	context_norm = [context_mean, context_std]
	body_norm = [body_mean, body_std]

	'''
	get the image name in a folder
	'''
	if args.mode is 'image':
		file = open('./faces_list.txt','w')
		for imagename in os.listdir(r'./data/faces'):
			images_path = os.path.join('./data/faces', imagename)
			file.write(images_path)
			file.write('\n')
		file.close()

	main(args, EMOTIONS, CAT, EMO2CAT, model_path, result_path, faces_path, context_norm, body_norm, ind2cat, ind2vad)
	# print("the catogray is:", cat)