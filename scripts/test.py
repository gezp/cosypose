
from PIL import Image
import numpy as np
from cosypose.cosypose_detector import getModel,inference

def main():
    detector,pose_predictor=getModel()
    print("start...........................................")
    #预测目标
    path="test.png"
    img = Image.open(path) 
    img = np.array(img)
    camera_k = np.array([[585.75607,    0,      320.5 ],\
                        [  0,      585.75607, 240.5],\
                        [  0,        0,        1,     ]])
    #预测
    pred=inference(detector,pose_predictor,img,camera_k)
    #poses,poses_input,K_crop,boxes_rend,boxes_crop
    print("num of pred:",len(pred))
    for i in range(len(pred)):
        print("object ",i,":",pred.infos.iloc[i].label,"------\n  pose:",pred.poses[i].numpy(),"\n  detection score:",pred.infos.iloc[i].score)

if __name__ == '__main__':
    main()
