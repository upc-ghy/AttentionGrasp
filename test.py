import os
import random
import sys
import numpy as np
import argparse
import time
import open3d as o3d
import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from PSAGrasp import PSAGrasp, pred_decode


from dataset_processed import GraspDataset, collate_fn, load_grasp_labels
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--checkpoint_path', required=True)
parser.add_argument('--dump_dir', required=True)
parser.add_argument('--camera', required=True)
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
TEST_DATASET = GraspDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split='test', camera=cfgs.camera,
                               num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False,use_fine=False)

print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TEST_DATALOADER))
net = PSAGrasp(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                    cylinder_radius=0.08, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def inference():
    batch_interval = 10
    stat_dict = {}
    net.eval()
    tic = time.time()
    sum_time = 0
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        

        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds, save_preds = pred_decode(end_points)
        if batch_idx % batch_interval == 0:
            toc = time.time()
            sum_time += (toc - tic)
            print('Eval batch: %d, time: %fs'%(batch_idx, (toc-tic)/batch_interval))
            tic = time.time()

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)
            spreds = save_preds[i].detach().cpu().numpy()

            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, 'result.npy')
            save_path2 = os.path.join(save_dir, 'save.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)
            np.save(save_path2,spreds)


def evaluate():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    res, ap = ge.eval_all(cfgs.dump_dir, proc=cfgs.num_workers)
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res)


if __name__=='__main__':
    torch.manual_seed(822)
    torch.cuda.manual_seed(822)
    np.random.seed(822)
    inference()
