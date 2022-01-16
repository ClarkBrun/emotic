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

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')
	parser.add_argument('--experiment_path', type=str, default='./debug', help='Path of experiment files (results, models, logs)') # ,required=True
	parser.add_argument('--model_dir', type=str, default='models', help='Folder to access the models')
	parser.add_argument('--result_dir', type=str, default='results', help='Path to save the results')
	parser.add_argument('--inference_file', type=str, default='faces_list.txt', help='Text file containing image context paths and bounding box')
	parser.add_argument('--video_file', type=str, default='Recoreded.mp4', help='Test video file')
	parser.add_argument('--mode_imagesORvideo', type=str, default = 'image', help='To choose the mode to detect image or video')
	# Generate args
	args = parser.parse_args()
	return args

def get_bbox(args, image_context, model_path, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):
	''' Use yolo to obtain bounding box of every person in context image. 
	:param yolo_model: Yolo model to obtain bounding box of every person in context image. 
	:param device: Torch device. Used to send tensors to GPU (if available) for faster processing. 
	:yolo_image_size: Input image size for yolo model. 
	:conf_thresh: Confidence threshold for yolo model. Predictions with object confidence > conf_thresh are returned. 
	:nms_thresh: Non-maximal suppression threshold for yolo model. Predictions with IoU > nms_thresh are returned. 
	:return: Numpy array of bounding boxes. Array shape = (no_of_persons, 4). 
	'''
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

def main(args, EMOTIONS, EMO2CAT, model_path, result_path, context_norm, body_norm, ind2cat, ind2vad):
	catograry = None
	device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")

	thresholds = torch.FloatTensor(np.load(os.path.join(result_path, 'val_thresholds.npy'))).to(device) 
	model_context = torch.load(os.path.join(model_path,'model_context1.pth')).to(device)
	model_body = torch.load(os.path.join(model_path,'model_body1.pth')).to(device)
	emotic_model = torch.load(os.path.join(model_path,'model_emotic1.pth')).to(device)
	models = [model_context, model_body, emotic_model]

	emoji_faces = []
	for index, emotion in enumerate(EMOTIONS):
		emoji = cv2.imread('./data/emojis/' + emotion + '.png', -1)
		emoji = cv2.resize(emoji, (600, 600), interpolation = cv2.INTER_CUBIC)
		emoji_faces.append(emoji)
	
	with open('./faces_list.txt', 'r') as f:
		lines = f.readlines()
	for idx, line in enumerate(lines):
		image_context_path = line.split('\n')[0].split(' ')[0]
		org_frame = cv2.imread(image_context_path)
		frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2RGB)

		bboxes = get_bbox(args, frame, model_path=model_path)
		for boxid in range(len(bboxes)):
			print(bboxes[boxid])
			cat_emotions = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context=frame, bbox=bboxes[boxid], to_print=True)
			for key, value in EMO2CAT.items():
				if cat_emotions in value:
					catograry = key
					break
				else:
					continue
		
		if catograry is not None:
			index = EMOTIONS.index(catograry)
			emoji_face = emoji_faces[index]
			for c in range(0, 3):
				# org_frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + org_frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
				org_frame[200:800, 200:800, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + org_frame[200:800, 200:800, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
				
			cv2.imwrite("SavedImage.jpg", org_frame)

			print("Done~")
		else:
			print("Cannot detect any emotions~")

if __name__=='__main__':
	args = parse_args()

	model_path = os.path.join(args.experiment_path, args.model_dir)
	result_path = os.path.join(args.experiment_path, args.result_dir)

	EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

	CAT = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
		  'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
		  'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

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

	main(args, EMOTIONS, EMO2CAT, model_path, result_path, context_norm, body_norm, ind2cat, ind2vad)
	# print("the catogray is:", cat)