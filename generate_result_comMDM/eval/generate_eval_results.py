import os
import numpy as np
import torch
from tqdm import tqdm

from utils.fixseed import fixseed
from utils.parser_util import generate_multi_args
from utils.model_util import load_model
from utils import dist_util

from model.comMDM import ComMDM
from data_loaders.get_data import get_dataset_loader

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils.rotation_conversions import *
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils.utils import process_motion_np, qrot_np

def hml_to_interclip_pair(m0_hml, m1_hml):

    T = m0_hml.shape[0]
    n_joints = 22
    
    rot0 = m0_hml[:, 67:193]
    rot1 = m1_hml[:, 67:193]
    
    pos0 = recover_from_ric(torch.from_numpy(m0_hml).float(), n_joints).numpy() # (T, 22, 3)
    pos1 = recover_from_ric(torch.from_numpy(m1_hml).float(), n_joints).numpy() # (T, 22, 3)
    
    raw0 = np.concatenate([pos0.reshape(T, -1), rot0], axis=-1)
    feat0, quat0, trans0 = process_motion_np(raw0, feet_thre=0.002, prev_frames=0, n_joints=n_joints)
    
    trans_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, -1.0, 0.0]])
    pos1 = np.einsum("mn, tjn->tjm", trans_mat, pos1)
    
    floor_height = pos1.min(axis=0).min(axis=0)[1]
    pos1[:, :, 1] -= floor_height
    

    pos1 = pos1 - trans0  
    T, V, _ = pos1.shape  
    quat0_expand = np.tile(quat0.reshape(1, 1, 4), (T, V, 1))
    q_flat = quat0_expand.reshape(-1, 4)
    v_flat = pos1.reshape(-1, 3)
    pos1 = qrot_np(q_flat, v_flat).reshape(T, V, 3)
    
    fid_l, fid_r = [7, 10], [8, 11]
    velfactor, heightfactor = np.array([0.002, 0.002]), np.array([0.12, 0.05])
    
    feet_l_x = (pos1[1:, fid_l, 0] - pos1[:-1, fid_l, 0]) ** 2
    feet_l_y = (pos1[1:, fid_l, 1] - pos1[:-1, fid_l, 1]) ** 2
    feet_l_z = (pos1[1:, fid_l, 2] - pos1[:-1, fid_l, 2]) ** 2
    feet_l_h = pos1[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

    feet_r_x = (pos1[1:, fid_r, 0] - pos1[:-1, fid_r, 0]) ** 2
    feet_r_y = (pos1[1:, fid_r, 1] - pos1[:-1, fid_r, 1]) ** 2
    feet_r_z = (pos1[1:, fid_r, 2] - pos1[:-1, fid_r, 2]) ** 2
    feet_r_h = pos1[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
    
    joint_positions1 = pos1.reshape(len(pos1), -1)
    joint_vels1 = pos1[1:] - pos1[:-1]
    joint_vels1 = joint_vels1.reshape(len(joint_vels1), -1)
    
    feat1 = joint_positions1[:-1]
    feat1 = np.concatenate([feat1, joint_vels1], axis=-1)
    feat1 = np.concatenate([feat1, rot1[:-1]], axis=-1)
    feat1 = np.concatenate([feat1, feet_l, feet_r], axis=-1)
    
    return feat0, feat1  

def pad_to_196(arr, max_len=196):
    T, D = arr.shape
    if T >= max_len:
        return arr[:max_len, :]
    pad = np.zeros((max_len - T, D), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def main(args, rep_idx=0):
    fixseed(args.seed + rep_idx) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    dist_util.setup_dist(device)

    print("Loading dataset...")
    data = get_dataset_loader(
        name="pw3d",
        batch_size=args.batch_size,
        num_frames=None,
        split='test',
        load_mode='text'
    )

    print("Loading model...")
    model, diffusion = load_model(args, data, device, ModelClass=ComMDM)
    model.to(device)
    model.eval()
    diffusion.num_timesteps = 50

    all_motion0, all_motion1 = [], []
    all_gt_motion0, all_gt_motion1 = [], []
    all_lengths = []
    all_text = []

    print("Start sampling...")


    for i, batch in enumerate(tqdm(data)):
        if i * args.batch_size > 1000:
            break

        motion, model_kwargs = batch
        B = motion.shape[0]
        n_frames = int(max(model_kwargs['y']['lengths']))

        model_kwargs['y'] = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in model_kwargs['y'].items()
        }

        if 'scale' not in model_kwargs['y']:
            model_kwargs['y']['scale'] = torch.ones(B, device=device)

        with torch.no_grad():
            sample, sample1 = diffusion.p_sample_loop(
                model,
                (B, model.njoints, model.nfeats, n_frames + 1),
                model_kwargs=model_kwargs,
                progress=False,
                predict_two_person=True,
            )


        canon0, sample = torch.split(sample, [1, sample.shape[-1] - 1], dim=-1)
        canon1, sample1 = torch.split(sample1, [1, sample1.shape[-1] - 1], dim=-1)
        

        sample_raw = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1))
        sample1_raw = data.dataset.t2m_dataset.inv_transform(sample1.cpu().permute(0, 2, 3, 1))

        if isinstance(sample_raw, torch.Tensor): sample_raw = sample_raw.numpy()
        if isinstance(sample1_raw, torch.Tensor): sample1_raw = sample1_raw.numpy()

        sample_raw = sample_raw.squeeze(1)   # [B, T, 263]
        sample1_raw = sample1_raw.squeeze(1) # [B, T, 263]

        # GT Person 0
        _, gt0_feat = torch.split(motion[:, :263, 0:1, :], [1, motion.shape[-1] - 1], dim=-1)
        gt0_raw = data.dataset.t2m_dataset.inv_transform(gt0_feat.cpu().permute(0, 2, 3, 1))
        if isinstance(gt0_raw, torch.Tensor): gt0_raw = gt0_raw.numpy()
        gt0_raw = gt0_raw.squeeze(1)

        # GT Person 1
        if 'other_motion' in model_kwargs['y']:
            other_m = model_kwargs['y']['other_motion']
            _, gt1_feat = torch.split(other_m[:, :263, 0:1, :], [1, other_m.shape[-1] - 1], dim=-1)
        else:
            _, gt1_feat = torch.split(motion[:, :263, 1:2, :], [1, motion.shape[-1] - 1], dim=-1)
            
        gt1_raw = data.dataset.t2m_dataset.inv_transform(gt1_feat.cpu().permute(0, 2, 3, 1))
        if isinstance(gt1_raw, torch.Tensor): gt1_raw = gt1_raw.numpy()
        gt1_raw = gt1_raw.squeeze(1)

        for b in range(B):
            valid_len = int(model_kwargs['y']['lengths'][b].item())

            m0_inter, m1_inter = hml_to_interclip_pair(sample_raw[b, :valid_len], sample1_raw[b, :valid_len])
            g0_inter, g1_inter = hml_to_interclip_pair(gt0_raw[b, :valid_len], gt1_raw[b, :valid_len])

            all_motion0.append(pad_to_196(m0_inter))
            all_motion1.append(pad_to_196(m1_inter))
            all_gt_motion0.append(pad_to_196(g0_inter))
            all_gt_motion1.append(pad_to_196(g1_inter))
            all_lengths.append(min(valid_len - 1, 196))
            
            current_text = model_kwargs['y']['text'][b]
            all_text.append(current_text)

    all_lengths = np.array(all_lengths)

    motion_hml = np.stack(all_motion0, axis=0).astype(np.float32)
    other_motion_hml = np.stack(all_motion1, axis=0).astype(np.float32)
    gt0_hml = np.stack(all_gt_motion0, axis=0).astype(np.float32)
    gt1_hml = np.stack(all_gt_motion1, axis=0).astype(np.float32)
    
    print("\n" + "=" * 50)
    print(f"Collected {len(all_text)} texts")
    print("=" * 50 + "\n")
    save_dict = {
        "motion": motion_hml,
        "other_motion": other_motion_hml,
        "motion_gt": gt0_hml,
        "other_motion_gt": gt1_hml,
        "lengths": all_lengths,
        "text": all_text
    }


    save_dir = "./generate_result_comMDM/eval/model_results"
    save_path = os.path.join(save_dir, f"comMDM_interclip_rep{rep_idx}.npy")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    np.save(save_path, save_dict)
    print(f"Saved Replication {rep_idx} to: {save_path}")


if __name__ == "__main__":
    args = generate_multi_args()
    for i in range(20):
        print(f"\n>>> Starting Sampling Replication {i}")
        main(args, rep_idx=i)