import argparse
from torchvision import transforms
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
import torch.nn as nn
from mmdet.apis import inference_detector, init_detector


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

class MultiTaskModel(nn.Module):
    """
    Creates a MTL model with the encoder from "model_backbone" 
    """
    def __init__(self, model_backbone):
        super(MultiTaskModel,self).__init__()
        self.encoder = model_backbone       #fastai function that creates an encoder given an architecture
        self.fc1 = nn.Linear(in_features=1432, out_features=2, bias=True)    #fastai function that creates a head
        self.fc2 = nn.Linear(in_features=1432, out_features=90, bias=True)
        self.fc3 = nn.Linear(in_features=1432, out_features=7, bias=True)
        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.encoder(x)
        gender = self.fc1(x)
        age = self.fc2(self.relu(x))
        emotions = self.fc3(x)

        return [age, gender, emotions]
    
    
    
class MultiTaskModel_grouped_age_head(nn.Module):
    """
    Creates a MTL model with the encoder from "model_backbone" 
    """
    def __init__(self, model):
        super(MultiTaskModel_grouped_age_head,self).__init__()
        self.encoder = model     
        self.idx_tensor = torch.from_numpy(np.array([idx for idx in range(31)])).to('cpu')
        self.age_group_head = nn.Linear(in_features=1400, out_features=31, bias=True)
        self.Softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
    def forward(self,x):

        age,gender,emotions = self.encoder(x)

        grouped_age = self.age_group_head(self.relu(age))
        regression_age = torch.sum(self.Softmax(grouped_age) * self.idx_tensor, axis=1)*3
  

        return [gender, (grouped_age,regression_age),  emotions]
   



def main():
    args = parse_args()
    device = torch.device(args.device)
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trans_tensor = transforms.ToTensor()

    age_estimator = torch.hub.load('mit-han-lab/ProxylessNAS', "proxyless_cpu" , pretrained=True)
    age_estimator.classifier = nn.Sequential(*list(age_estimator.classifier.children())[:-3])
    age_estimator = MultiTaskModel(age_estimator)
    age_estimator.fc2 = nn.Linear(in_features=1432, out_features=1400, bias=True)
    age_estimator = MultiTaskModel_grouped_age_head(age_estimator)
    age_estimator.load_state_dict(torch.load('/home/maxim/Age_detection_pipline/proxyless-cpu_age_trained_3_heads.pth',map_location=device))
    age_estimator.to(device)
    age_estimator.eval()
  
    



    detector = init_detector(args.config, args.checkpoint, device=device)

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
                frame = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb = Image.fromarray(np.uint8(img_rgb))
                inputs = val_transforms(img_rgb)
 
                img_rgb = trans_tensor(img_rgb).unsqueeze(0)
                inputs = inputs.to(device)

                group,regress_age = age_estimator(img_rgb)[1]
        
            
                img = cv2.putText(img, str(int(regress_age.squeeze(0).cpu().detach().numpy())),(int(bbox[0]), int(bbox[1])+25), font, fontScale, color, thickness, cv2.LINE_AA)#
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.imshow('Video', img)
	

if __name__ == '__main__':
    main()
