from PIL import Image
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse

from cosypose.datasets.datasets_cfg import make_object_dataset
# Pose estimator
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor

# Detection
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from cosypose.integrated.detector import Detector

from cosypose.config import EXP_DIR

def load_detector(run_id):
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model

def load_pose_predictor(coarse_run_id, refiner_run_id=None):
    run_dir = EXP_DIR / coarse_run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config_pose(cfg)
    #object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    object_ds = make_object_dataset(cfg.object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=1)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,refiner_model=refiner_model)
    return model

def getModel(): 
    #加载模型
    detector_run_id='detector-bop-tless-pbr--873074'
    coarse_run_id='coarse-bop-tless-pbr--506801'
    refiner_run_id='refiner-bop-tless-pbr--233420'
    detector = load_detector(detector_run_id)
    pose_predictor = load_pose_predictor(coarse_run_id=coarse_run_id,refiner_run_id=refiner_run_id)
    return detector,pose_predictor


def inference(detector,pose_predictor,image,camera_k):
    #[1,540,720,3]->[1,3,540,720]
    images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    #[1,3,3]
    cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)
    #2D detector 
    #print("start detect object.")
    box_detections = detector.get_detections(images=images, one_instance_per_class=False, 
                    detection_th=0.8,output_masks=False, mask_th=0.9)
    #pose esitimition
    if len(box_detections) == 0:
        return None
    #print("start estimate pose.")
    final_preds, all_preds=pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
                        n_coarse_iterations=1,n_refiner_iterations=4)
    #print("inference successfully.")
    #result: this_batch_detections, final_preds
    return final_preds.cpu()


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


#####################################################
import transforms3d

class CosyposeDetector():
    def __init__(self, camera_k, one_instance_per_class = False, 
                 detection_th = 0.8, output_masks = False, mask_th = 0.9, 
                 n_coarse_iterations = 1, n_refiner_iterations = 4):
        # init model
        self.detector,self.pose_predictor=getModel()
        # init parameter
        self.camera_k = camera_k
        self.one_instance_per_class = one_instance_per_class
        self.detection_th = detection_th
        self.output_masks = output_masks
        self.mask_th = mask_th
        self.n_coarse_iterations = n_coarse_iterations
        self.n_refiner_iterations = n_refiner_iterations
        # data
        self.results = []
        print("CosyposeDetector initialised successfuly")

    def inference(self, image):
        self.results = []
        # [1,540,720,3]->[1,3,540,720]
        images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
        images = images.permute(0, 3, 1, 2) / 255
        # [1,3,3]
        cameras_k = torch.from_numpy(self.camera_k).cuda().float().unsqueeze_(0)
        # 2D detector 
        # print("start detect 2d object.")
        box_detections = self.detector.get_detections(images=images,
                        one_instance_per_class = self.one_instance_per_class, 
                        detection_th = self.detection_th,
                        output_masks = self.output_masks,
                        mask_th = self.mask_th)
        # pose esitimition
        if len(box_detections) == 0:
            # print("detect 2d: nothing.")
            return
        # print("start estimate pose.")
        final_preds, all_preds = self.pose_predictor.get_predictions(images, cameras_k, detections=box_detections,
                            n_coarse_iterations = self.n_coarse_iterations,
                            n_refiner_iterations = self.n_refiner_iterations)
        # print("inference successfully.")
        pred = final_preds.cpu()
        if pred is None:
            # print("predict pose: nothing.")
            return
        for i in range(len(pred)):
            # (label, score, t, r)
            rt = pred.poses[i].numpy()
            r = rt[0:3, 0:3]
            t = rt[0:3, 3]
            result = (pred.infos.iloc[i].label, pred.infos.iloc[i].score, t, r)
            self.results.append(result)

    def get_all_results(self):
        return self.results

    def get_model_results(self, model):
        model_results = []
        for result in self.results:
            if result[0] == model:
                model_results.append(result)
        return model_results

    def get_max_model_result(self, model):
        max_socre = 0
        max_result = None
        # get model with max socre
        for result in self.results:
            if result[0] == model:
                if result[1] > max_socre:
                    max_socre = result[1]
                    max_result = result
        return max_result

    def print_result(self, result):
        r,q,y = transforms3d.euler.mat2euler(result[3])
        r,p,y = r*180/3.14159,q*180/3.14159,y*180/3.14159
        print("label ",result[0],"---  score:",result[1],"------\n",
            "position:",result[2], ",rpy:", (r,p,y),"\n")

    def print_all(self):
        for result in self.results:
            self.print_result(result)


if __name__ == '__main__':
    main()
