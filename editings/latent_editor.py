import os.path

import numpy as np
import torch

from configs.paths_config import interfacegan_folder
from editings import styleclip_global_utils
from utils.edit_utils import get_affine_layers, load_stylespace_std, to_styles, w_to_styles


class LatentEditor:
    def __init__(self):
        interfacegan_directions = {
            os.path.splitext(file)[0]: np.load(os.path.join(interfacegan_folder, file), allow_pickle=True)
            for file in os.listdir(interfacegan_folder) if file.endswith('.npy')}
        self.interfacegan_directions_tensors = {name: torch.from_numpy(arr).cuda()[0, None]
                                                for name, arr in interfacegan_directions.items()}


    def get_interfacegan_edits(self, orig_w, edit_names, edit_range):
        edits = []
        for edit_name, direction in self.interfacegan_directions_tensors.items():
            if edit_name not in edit_names:
                continue
            for factor in np.linspace(*edit_range):
                w_edit = self._apply_interfacegan(orig_w, direction, factor / 2)
                edits.append((w_edit, edit_name, factor))

        return edits, False

    def get_amplification_edits(self, orig_w, edit_range, edit_layers=0):
        edits = []
        yaw = self.interfacegan_directions_tensors['yaw'][0].float()
        pitch = self.interfacegan_directions_tensors['pitch'][0].float()
        yaw /= torch.norm(yaw, 2)
        pitch /= torch.norm(pitch, 2)
        for factor in np.linspace(*edit_range):
            w_edit = self._apply_amplification(orig_w, factor, yaw=yaw, pitch=pitch, edit_layers=edit_layers)
            edits.append((w_edit, f'amplification_{edit_layers}', factor))

        return edits, False

    def get_amplification_edits_with_pose(self, orig_w, edit_range, edit_layers=0):
        edits = []
        for factor in np.linspace(*edit_range):
            w_edit = self._apply_amplification_with_pose(orig_w, factor, edit_layers=edit_layers)
            edits.append((w_edit, f'amplification_with_pose_{edit_layers}', factor))

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
    def _apply_interfacegan(latent, direction, factor=1, factor_range=None):
        edit_latents = []
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latent + f * direction
                edit_latents.append(edit_latent)
            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = latent + factor * direction
        return edit_latents

    @staticmethod
    def _apply_amplification(latent, factor=2, yaw=None, pitch=None, edit_layers=0):
        dist = latent[1:] - latent[0]
        yaw_2d = yaw.reshape(yaw.shape[0], 1)
        pitch_2d = pitch.reshape(pitch.shape[0], 1)

        yaw_direction_vec = torch.matmul(dist, yaw_2d) * yaw
        pitch_direction_vec = torch.matmul(dist, pitch_2d) * pitch
        dist -= (yaw_direction_vec + pitch_direction_vec)

        edit_latents = latent
        if edit_layers <= 0:
            edit_latents[1:, edit_layers:] += (factor - 1) * dist[:, edit_layers:]
        else:
            edit_latents[1:, :edit_layers] += (factor - 1) * dist[:, :edit_layers]


        return edit_latents

    @staticmethod
    def _apply_amplification_with_pose(latent, factor=2, edit_layers=0):
        dist = latent[1:] - latent[0]
        edit_latents = latent
        if edit_layers <= 0:
            edit_latents[1:, edit_layers:] += (factor - 1) * dist[:, edit_layers:]
        else:
            edit_latents[1:, :edit_layers] += (factor - 1) * dist[:, :edit_layers]
        return edit_latents

    @staticmethod
    def _save_latent(latent, name):
        latent_np = latent.cpu().numpy()
        np.save(name, latent_np)