import argparse
from torchvision import transforms
import cv2
import torch
import numpy as np
from numpy.linalg import norm as l2norm
import sys
from PIL import Image
from matplotlib import cm
import torch.nn as nn
from mmdet.apis import inference_detector, init_detector
sys.path.append('/home/maxim/Age_detection_pipline/insightface/recognition/arcface_torch')
from backbones import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--config',default = '/home/maxim/Age_detection_pipline/insightface/detection/scrfd/configs/scrfd/scrfd_500m.py', help='test config file path')
    parser.add_argument('--checkpoint',default = '/home/maxim/Age_detection_pipline/insightface/detection/scrfd/model.pth', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    args = parser.parse_args()
    return args




def inference(net, img):
    
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
 
    net.eval()
    feat = net(img).detach().numpy()
    return feat



def main():
    
    anger = ['anger','/home/maxim/Age_detection_pipline/Expressions/references/anger/066_y_m_a_a.jpg']

    disgust = ['disgust','/home/maxim/Age_detection_pipline/Expressions/references/disgust/140_y_f_d_a.jpg']

    fear = ['fear','/home/maxim/Age_detection_pipline/Expressions/references/fear/116_m_m_f_a.jpg']

    happiness = ['happiness','/home/maxim/Age_detection_pipline/Expressions/references/happiness/079_o_f_h_a.jpg']

    neutral = ['neutral','/home/maxim/Age_detection_pipline/Expressions/references/neutral/116_m_m_n_a.jpg']

    sadness = ['sadness','/home/maxim/Age_detection_pipline/Expressions/references/sadness/140_y_f_s_a.jpg']

    reference_list = [anger,disgust,fear,happiness,neutral,sadness]
    args = parse_args()
    device = torch.device(args.device)
    expression_estimator = get_model('r18', fp16=False)
    expression_estimator.load_state_dict(torch.load('/home/maxim/Age_detection_pipline/Expressions/r18_expression_recognition_weights.pth',map_location = device))

    detector = init_detector(args.config, args.checkpoint, device=device)
    
    expression_reference_logits=[]
    for expression_list in reference_list:
        logits_reference = inference(expression_estimator,cv2.imread(expression_list[1]))
        logits_reference = np.mean(logits_reference, axis=0)
        logits_reference /= l2norm(logits_reference)
        expression_reference_logits.append((expression_list[0],logits_reference))

    camera = cv2.VideoCapture(args.camera_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()
        result = inference_detector(detector, img)
        

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
        for bbox in result[0]:
            if bbox[4] >= args.score_thr:
                img_rgb = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                logits_val_img = inference(expression_estimator,img_rgb)
                logits_val_img = np.mean(logits_val_img, axis=0)
                logits_val_img /= l2norm(logits_val_img)
                
                expression_reference_scores=[]
   
                for logits in expression_reference_logits:
                    score = np.dot(logits_val_img,logits[1])
                    expression_reference_scores.append(score)
                
                max_score = max(expression_reference_scores)
                expression = reference_list[expression_reference_scores.index(max_score)][0]

                
                img = cv2.putText(img, expression,(int(bbox[0]), int(bbox[1])+25), font, fontScale, color, thickness, cv2.LINE_AA)#str(int(regress_age.squeeze(0).cpu().detach().numpy()))
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.imshow('Video', img)
	

if __name__ == '__main__':
    main()
