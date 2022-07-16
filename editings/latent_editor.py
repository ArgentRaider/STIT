import os.path
import pickle

import numpy as np
import torch

from configs.paths_config import interfacegan_folder, stylespace_folder
from editings import styleclip_global_utils
from utils.edit_utils import get_affine_layers, load_stylespace_std, to_styles, w_to_styles


class LatentEditor:
    def __init__(self):
        interfacegan_directions = {
            os.path.splitext(file)[0]: np.load(os.path.join(interfacegan_folder, file), allow_pickle=True)
            for file in os.listdir(interfacegan_folder) if file.endswith('.npy')}

        self.interfacegan_directions_tensors = {name: torch.from_numpy(arr).cuda()
                                                for name, arr in interfacegan_directions.items()}


    def get_interfacegan_edits(self, orig_w, edit_names, edit_range, edit_layers_start=None, edit_layers_end=None):
        edits = []
        for edit_name, direction in self.interfacegan_directions_tensors.items():
            if edit_name not in edit_names:
                continue
            for factor in np.linspace(*edit_range):
                w_edit = self._apply_interfacegan(orig_w, direction, factor / 2, edit_layers_start, edit_layers_end)
                edits.append((w_edit, edit_name, factor))

        return edits, False

    def get_amplification_edits(self, orig_w, edit_names, edit_range, edit_layers_start=None, edit_layers_end=None, origin_pivot_type={'type': 'first'}):
        assert(type(origin_pivot_type) == dict and origin_pivot_type.get('type') in ['first', 'mean', 'min', 'min_weight'])

        edits = []
        if len(edit_names) == 0 or not edit_names[0] in self.interfacegan_directions_tensors.keys():
            direction = None
        else:
            direction = self.interfacegan_directions_tensors[edit_names[0]].float().clone()  # [18, 512]
            for i in range(direction.shape[0]):
                direction[i] /= torch.norm(direction[i], 2).item()

        yaw = self.interfacegan_directions_tensors['yaw'][0].float()
        pitch = self.interfacegan_directions_tensors['pitch'][0].float()
        yaw /= torch.norm(yaw, 2)
        pitch -= torch.dot(pitch, yaw) * yaw
        pitch /= torch.norm(pitch, 2)
        for factor in np.linspace(*edit_range):
            w_edit = self._apply_amplification(orig_w, direction, factor, yaw=yaw, pitch=pitch, edit_layers_start=edit_layers_start, edit_layers_end=edit_layers_end, origin_pivot_type=origin_pivot_type)
            if direction is None:
                edits.append((w_edit, f'amplification_{edit_layers_start}_{edit_layers_end}_{factor}', factor))
            else:
                edits.append((w_edit, f'amplification_{edit_names[0]}_{edit_layers_start}_{edit_layers_end}_{factor}', factor))

        return edits, False

    def get_style_clip_amplification_edits(self, orig_w, edit_names, edit_range, generator, neutral_class='face', origin_pivot_type={'type': 'first'}, 
                                            beta=0.1, use_stylespace_std=False):
        affine_layers = get_affine_layers(generator.synthesis)
        if edit_names is None:
            edit = None
        else:
            edit_directions = styleclip_global_utils.get_direction(neutral_class, edit_names[0], beta)
            
            if use_stylespace_std:
                s_std = load_stylespace_std()
                edit_directions = to_styles(edit_directions, affine_layers)
                edit = [(s * std).float() for s, std in zip(edit_directions, s_std)]
            else:
                edit = to_styles(edit_directions, affine_layers)
                edit = [edit_i.float() for edit_i in edit]

        factors = np.linspace(*edit_range)
        styles = w_to_styles(orig_w, affine_layers)
        final_edits = []

        for factor in factors:
            edited_styles = self._apply_amplification_style(orig_w, affine_layers, edit, factor, origin_pivot_type)
            final_edits.append((edited_styles, edit_names[0], f'{factor}_{beta}'))
        return final_edits, True
    
    def get_transfer_edits(self, transfer_src_w, transfer_dst_w_list, edit_names, edit_layers_start=None, edit_layers_end=None, origin_pivot_type={'type': 'first'}):
        assert(type(origin_pivot_type) == dict and origin_pivot_type.get('type') in ['first', 'mean', 'min', 'min_weight'])

        edits = []
        if len(edit_names) == 0 or not edit_names[0] in self.interfacegan_directions_tensors.keys():
            direction = None
        else:
            direction = self.interfacegan_directions_tensors[edit_names[0]].float().clone()
            for i in range(direction.shape[0]):
                direction[i] /= torch.norm(direction[i], 2).item()

        for transfer_dst_w in transfer_dst_w_list:
            w_edit = self._apply_transfer(transfer_src_w, transfer_dst_w, direction, edit_layers_start, edit_layers_end, origin_pivot_type)
            edits.append((w_edit, f'transfer_{edit_layers_start}_{edit_layers_end}', 1))

        return edits, False
    
    def get_removal_edits(self, orig_w, edit_names, edit_layers_start=None, edit_layers_end=None, origin_pivot_type={'type': 'first'}):
        assert(type(origin_pivot_type) == dict and origin_pivot_type.get('type') in ['first', 'mean', 'min', 'min_weight'])

        edits = []
        for edit_name, direction in self.interfacegan_directions_tensors.items():
            if edit_name not in edit_names:
                continue
            direction = direction[0]
            w_edit = self._apply_removal(orig_w, direction, edit_layers_start, edit_layers_end, origin_pivot_type)
            edits.append((w_edit, f'removal_{edit_name}_{edit_layers_start}_{edit_layers_end}', -1))

        return edits, False

    def get_amplification_edits_with_pose(self, orig_w, edit_range, edit_layers_start=None, edit_layers_end=None, origin_pivot_type=False):
        edits = []
        for factor in np.linspace(*edit_range):
            w_edit = self._apply_amplification_with_pose(orig_w, factor, edit_layers_start=edit_layers_start, edit_layers_end=edit_layers_end, origin_pivot_type=origin_pivot_type)
            edits.append((w_edit, f'amplification_with_pose_{edit_layers_start}_{edit_layers_end}', factor))

        return edits, False

    @staticmethod
    def get_styleclip_global_edits(orig_w, neutral_class, target_class, beta, edit_range, generator, edit_name, use_stylespace_std=False):
        affine_layers = get_affine_layers(generator.synthesis)
        edit_directions = styleclip_global_utils.get_direction(neutral_class, target_class, beta)
        edit_disentanglement = beta
        if use_stylespace_std:
            s_std = load_stylespace_std()
            edit_directions = to_styles(edit_directions, affine_layers)
            edit = [s * std for s, std in zip(edit_directions, s_std)]
        else:
            edit = to_styles(edit_directions, affine_layers)

        direction = edit_name[0]
        factors = np.linspace(*edit_range)
        styles = w_to_styles(orig_w, affine_layers)
        final_edits = []

        for factor in factors:
            edited_styles = [style + factor * edit_direction for style, edit_direction in zip(styles, edit)]
            final_edits.append((edited_styles, direction, f'{factor}_{edit_disentanglement}'))
        return final_edits, True

    @staticmethod
    def _apply_interfacegan(latent, direction, factor=1, edit_layers_start=None, edit_layers_end=None, factor_range=None):
        edit_latents = []
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latent + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent.clone()
            edit_latents[:, edit_layers_start:edit_layers_end] += factor * direction[edit_layers_start:edit_layers_end]
        return edit_latents

    @staticmethod
    def _apply_amplification(latent, direction=None, factor=2, yaw=None, pitch=None, edit_layers_start=None, edit_layers_end=None, 
                            origin_pivot_type={'type': 'first'}, clamp=False):
        if origin_pivot_type['type'] == 'mean':
            origin = latent.mean(0).unsqueeze(0)
        elif origin_pivot_type['type'] == 'first':
            origin = latent[0, None]
        elif origin_pivot_type['type'] == 'min':
            min_index = origin_pivot_type['min_index']
            origin = latent[min_index, None]
        elif origin_pivot_type['type'] == 'min_weight':
            weight = origin_pivot_type['weight']
            origin = torch.zeros([latent.shape[1], latent.shape[2]]).float().cuda()
            for i in range(len(weight)):
                origin += weight[i] * latent[i]
            origin.unsqueeze(0)

        dist = latent - origin
        yaw_2d = yaw.reshape(yaw.shape[0], 1)
        pitch_2d = pitch.reshape(pitch.shape[0], 1)

        yaw_direction_vec = torch.matmul(dist, yaw_2d) * yaw
        pitch_direction_vec = torch.matmul(dist, pitch_2d) * pitch
        dist -= (yaw_direction_vec + pitch_direction_vec)

        if not direction is None:
            for i in range(direction.shape[0]):
                direction_2d = direction[i].reshape(direction[i].shape[0], 1) # direction_2d -> [512, 1]
                if clamp:
                    dist[:, i] = torch.clamp(torch.matmul(dist[:, i], direction_2d), min=0) * direction[i] # dist[:, i] -> [frame_num, 1, 512]
                else:
                    dist[:, i] = torch.matmul(dist[:, i], direction_2d) * direction[i] # dist[:, i] -> [frame_num, 1, 512]

        edit_latents = latent.clone()
        edit_latents[:, edit_layers_start:edit_layers_end] += (factor - 1) * dist[:, edit_layers_start:edit_layers_end]

        return edit_latents
    
    @staticmethod
    def _apply_amplification_style(latent, affine_layers, style_direction=None, factor=2, origin_pivot_type={'type': 'first'}, clamp=False):
        if origin_pivot_type['type'] == 'mean':
            origin = latent.mean(0).unsqueeze(0)
        elif origin_pivot_type['type'] == 'first':
            origin = latent[0, None]
        elif origin_pivot_type['type'] == 'min':
            min_index = origin_pivot_type['min_index']
            origin = latent[min_index, None]
        elif origin_pivot_type['type'] == 'min_weight':
            weight = origin_pivot_type['weight']
            origin = torch.zeros([latent.shape[1], latent.shape[2]]).float().cuda()
            for i in range(len(weight)):
                origin += weight[i] * latent[i]
            origin.unsqueeze(0)
        
        styles = w_to_styles(latent, affine_layers)
        styles_origin = w_to_styles(origin, affine_layers)
        
        edit_styles = []
        for i in range(len(styles)):
            edit_style_i = styles[i].clone()    # styles[i] -> [frame_num, style_dim]
            dist = styles[i] - styles_origin[i] # styles_origin[i] -> [1, style_dim], dist -> [frame_num, style_dim]

            if not style_direction is None:
                length = torch.norm(style_direction[i], 2)
                if length < 1e-4:
                    dist = 0
                else:
                    style_direction[i] /= length
                    style_direction_2d = style_direction[i].reshape(-1, 1) # style_direction_2d -> [style_dim, 1]
                    if clamp:
                        dist = torch.clamp(torch.matmul(dist, style_direction_2d), min=0) * style_direction[i]
                    else:
                        dist = torch.matmul(dist, style_direction_2d) * style_direction[i]

            edit_style_i += (factor - 1) * dist
            edit_styles.append(edit_style_i)
        return edit_styles

    @staticmethod
    def _apply_amplification_with_pose(latent, factor=2, edit_layers_start=None, edit_layers_end=None, origin_pivot_type={'type': 'first'}):
        if origin_pivot_type['type'] == 'mean':
            origin = latent.mean(0).unsqueeze(0)
        elif origin_pivot_type['type'] == 'first':
            origin = latent[0, None]
        elif origin_pivot_type['type'] == 'min':
            min_index = origin_pivot_type['min_index']
            origin = latent[min_index, None]
        elif origin_pivot_type['type'] == 'min_weight':
            weight = origin_pivot_type['weight']
            origin = torch.zeros([latent.shape[1], latent.shape[2]]).float().cuda()
            for i in range(len(weight)):
                origin += weight[i] * latent[i]
            origin.unsqueeze(0)

        dist = latent - origin
        edit_latents = latent.clone()
        edit_latents[:, edit_layers_start:edit_layers_end] += (factor - 1) * dist[:, edit_layers_start:edit_layers_end]
            
        return edit_latents
    
    @staticmethod
    def _apply_transfer(src_latent, dst_latent, direction=None, edit_layers_start=None, edit_layers_end=None, origin_pivot_type={'type': 'first'}):
        if origin_pivot_type['type'] == 'mean':
            src_origin = src_latent.mean(0).unsqueeze(0)
            dst_origin = dst_latent.mean(0).unsqueeze(0)
        elif origin_pivot_type['type'] == 'first':
            src_origin = src_latent[0, None]
            dst_origin = dst_latent[0, None]
        elif origin_pivot_type['type'] == 'min':
            src_min_index = origin_pivot_type['src_min_index']
            dst_min_index = origin_pivot_type['dst_min_index']
            src_origin = src_latent[src_min_index, None]
            dst_origin = dst_latent[dst_min_index, None]
        elif origin_pivot_type['type'] == 'min_weight':
            src_weight = origin_pivot_type['src_weight']
            dst_weight = origin_pivot_type['dst_weight']
            src_origin = torch.zeros([src_latent.shape[1], src_latent.shape[2]]).float().cuda()
            dst_origin = torch.zeros([dst_latent.shape[1], dst_latent.shape[2]]).float().cuda()
            for i in range(len(src_weight)):
                src_origin += src_weight[i] * src_latent[i]
            for i in range(len(dst_weight)):
                dst_origin += dst_weight[i] * dst_latent[i]
            src_origin.unsqueeze(0)
            dst_origin.unsqueeze(0)
        
        src_dist = src_latent - src_origin
        if not direction is None:
            for i in range(direction.shape[0]):
                direction_2d = direction[i].reshape(direction[i].shape[0], 1)
                src_dist[i] = torch.matmul(src_dist[i], direction_2d) * direction[i]

        edit_latents = dst_origin.repeat(src_dist.shape[0], 1, 1)
        edit_latents[:, edit_layers_start:edit_layers_end] += src_dist[:, edit_layers_start:edit_layers_end]

        return edit_latents
    
    @staticmethod
    def _apply_removal(latent, direction, edit_layers_start=None, edit_layers_end=None, origin_pivot_type={'type': 'first'}):
        if origin_pivot_type['type'] == 'mean':
            origin = latent.mean(0).unsqueeze(0)
        elif origin_pivot_type['type'] == 'first':
            origin = latent[0, None]
        elif origin_pivot_type['type'] == 'min':
            min_index = origin_pivot_type['min_index']
            origin = latent[min_index, None]
        elif origin_pivot_type['type'] == 'min_weight':
            weight = origin_pivot_type['weight']
            origin = torch.zeros([latent.shape[1], latent.shape[2]]).float().cuda()
            for i in range(len(weight)):
                origin += weight[i] * latent[i]
            origin.unsqueeze(0)
        
        dist = latent - origin
        remove_direction_2d = direction.reshape(direction.shape[0], 1)
        remove_direction_vec = torch.matmul(dist, remove_direction_2d) * direction
        dist -= remove_direction_vec

        edit_latents = origin.repeat(dist.shape[0], 1, 1)
        edit_latents[:, edit_layers_start:edit_layers_end] += dist[:, edit_layers_start:edit_layers_end]

        return edit_latents

    @staticmethod
    def _save_latent(latent, name):
        latent_np = latent.cpu().numpy()
        np.save(name, latent_np)