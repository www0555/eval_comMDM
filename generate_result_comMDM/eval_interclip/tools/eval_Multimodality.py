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
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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


def evaluate_multimodality_grouped(all_rep_data, eval_wrapper, file):
    print('========== Evaluating MultiModality (Grouped) ==========')

    num_groups = len(all_rep_data) // mm_num_repeats  
    results = []

    for g in range(num_groups):
        group_data = all_rep_data[g * mm_num_repeats:(g + 1) * mm_num_repeats]

        num_samples = len(group_data[0]["motion"])
        mm_motion_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(num_samples), desc=f"Group {g}"):
                motions = []

                for rep_data in group_data:
                    text = rep_data["text"][i]
                    lengths = rep_data["lengths"][i]

                    m1 = rep_data["motion"][i]
                    m2 = rep_data["other_motion"][i]

                    motion1 = torch.tensor(m1).unsqueeze(0).float()
                    motion2 = torch.tensor(m2).unsqueeze(0).float()
                    lengths_tensor = torch.tensor([lengths])

                    batch = (
                        ["comMDM"],
                        [text],
                        motion1,
                        motion2,
                        lengths_tensor
                    )

                    emb = eval_wrapper.get_motion_embeddings(batch)
                    motions.append(emb.squeeze(0))

                motions = torch.stack(motions, dim=0)  # (30, D)
                mm_motion_embeddings.append(motions.unsqueeze(0))

        mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()

        mm_score = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        results.append(mm_score)

        print(f'Group {g} Multimodality: {mm_score:.4f}')
        print(f'Group {g} Multimodality: {mm_score:.4f}', file=file, flush=True)

    results = np.array(results)
    mean = results.mean()
    std = results.std()

    print('========== MultiModality Summary ==========')
    print(f'Mean: {mean:.4f}, Std: {std:.4f}')
    print(f'Mean: {mean:.4f}, Std: {std:.4f}', file=file, flush=True)

    return mean, std


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def evaluation(log_file, eval_wrapper, npy_path, batch_size):
    with open(log_file, 'w') as f:
        for replication in range(replication_times):

            current_npy = base_npy_path.format(replication)
            if not os.path.exists(current_npy):
                print(f"Warning: {current_npy} not found, skipping.")
                continue
        all_rep_data = []
        for i in range(400):   
            path = base_npy_path.format(i)
            if not os.path.exists(path):
                continue
            data = np.load(path, allow_pickle=True).item()
            all_rep_data.append(data)
        evaluate_multimodality_grouped(all_rep_data, eval_wrapper, f)
        print(f"Loaded {len(all_rep_data)} replications")
   


if __name__ == '__main__':
    mm_num_samples = 47
    mm_num_repeats = 20
    mm_num_times = 10

    diversity_times = 30
    replication_times = 20


    batch_size = 48

    print("Loading Evaluator...")
    base_npy_path = "./generate_result_comMDM/eval/model_results/comMDM_interclip_rep{}.npy"
    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    
    evalmodel_cfg = get_config("configs/eval_model.yaml")
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)

    log_file = f'./evaluation_2.log'
    evaluation(log_file, eval_wrapper, base_npy_path , batch_size)