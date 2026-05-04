import sys
import os
sys.path.append(sys.path[0]+r"/../")
import numpy as np
import torch
from datasets import get_dataset_motion_loader, get_motion_loader
from datetime import datetime
from models import *
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils.utils import *
from configs import get_config
from os.path import join as pjoin
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')


class ComMDMDataset(Dataset):
    def __init__(self, npy_path, mode="gen"):
        data = np.load(npy_path, allow_pickle=True).item()
        self.data = data
        self.mode = mode
        self.len = len(data["motion"])
        
    def __getitem__(self, idx):
        text = self.data["text"][idx]
        lengths = self.data["lengths"][idx]
        
        if self.mode == "gt":
            m1 = self.data["motion_gt"][idx]
            m2 = self.data["other_motion_gt"][idx]
            return "ground truth", text, m1, m2, lengths
        else:
            m1 = self.data["motion"][idx]
            m2 = self.data["other_motion"][idx]
            return "comMDM", text, m1, m2, lengths

    def __len__(self):
        return self.len

def get_comMDM_loader(npy_path, batch_size=96, mode="gen"):
    dataset = ComMDMDataset(npy_path, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(motion_loader), desc=f"MM Dist ({motion_loader_name})"):
                
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                

                mm_dist_sum += dist_mat.trace()
                argsmax = np.argsort(dist_mat, axis=1)

                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            mm_dist = mm_dist_sum / all_size
            R_precision = top_k_count / all_size
            
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict

def evaluate_fid(gt_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(gt_loader), desc="FID GT Embeddings"):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
            
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict

def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def evaluation(log_file, eval_wrapper, npy_path, batch_size):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({
            'MM Distance': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({})
        })
        
        all_rep_data = []
        for replication in range(replication_times):
            current_npy = base_npy_path.format(replication)
            if not os.path.exists(current_npy):
                print(f"Warning: {current_npy} not found, skipping.")
                continue
            data = np.load(current_npy, allow_pickle=True).item()
            all_rep_data.append(data)

            gen_loader = get_comMDM_loader(current_npy, batch_size, mode="gen")
            gt_loader = get_comMDM_loader(current_npy, batch_size, mode="gt")
            
            motion_loaders = {
                "ground truth": gt_loader,
                "comMDM": gen_loader
            }

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)
            div_score_dict = evaluate_diversity(acti_dict, f)
            print(f'!!! DONE !!!\n', file=f, flush=True)

            for key, item in mat_score_dict.items():
                all_metrics['MM Distance'].setdefault(key, []).append(item)
            for key, item in R_precision_dict.items():
                all_metrics['R_precision'].setdefault(key, []).append(item)
            for key, item in fid_score_dict.items():
                all_metrics['FID'].setdefault(key, []).append(item)
            for key, item in div_score_dict.items():
                all_metrics['Diversity'].setdefault(key, []).append(item)

        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values))
                if isinstance(mean, (np.float64, np.float32)):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}] '
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f; ' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)


if __name__ == '__main__':
    mm_num_samples = 47
    mm_num_repeats = 30
    mm_num_times = 10
    diversity_times = 30
    replication_times = 20
    batch_size = 48

    print("Loading Evaluator...")
    base_npy_path = "./generate_result_comMDM/eval/model_results/comMDM_interclip_rep{}.npy"
    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    
    evalmodel_cfg = get_config("configs/eval_model.yaml")
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)

    log_file = f'./evaluation_1.log'
    evaluation(log_file, eval_wrapper, base_npy_path , batch_size)