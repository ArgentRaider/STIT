from editings.latent_editor import LatentEditor
import numpy as np
import torch
from editings import styleclip_global_utils
from utils.edit_utils import get_affine_layers, load_stylespace_std, to_styles, w_to_styles

class ConditionalLatentEditor(LatentEditor):
    def __init__(self):
        super().__init__()
    
    def get_conditional_amplification_edits(self, orig_w, edit_names, conditional_name, edit_range, edit_layers_start=None, edit_layers_end=None, origin_pivot_type={'type': 'first'}):
        assert(type(origin_pivot_type) == dict and origin_pivot_type.get('type') in ['first', 'mean', 'min', 'min_weight'])

        edits = []
        if len(edit_names) == 0 or not edit_names[0] in self.interfacegan_directions_tensors.keys():
            direction = None
        else:
            direction = self.interfacegan_directions_tensors[edit_names[0]].float().clone()  # [18, 512]
            for i in range(direction.shape[0]):
                direction[i] /= torch.norm(direction[i], 2).item()
            if not conditional_name is None:
                conditional_direction = self.interfacegan_directions_tensors[conditional_name].float().clone()  # [18, 512]
                for i in range(conditional_direction.shape[0]):
                    conditional_direction[i] /= torch.norm(conditional_direction[i], 2).item()
                    direction[i] -= torch.dot(direction[i], conditional_direction[i]) * conditional_direction[i]
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

    def get_conditional_style_clip_amplification_edits(self, orig_w, edit_names, conditional_name, edit_range, generator, neutral_class='face', origin_pivot_type={'type': 'first'}, beta=0.1, use_stylespace_std=False):
        affine_layers = get_affine_layers(generator.synthesis)
        
        edit = None
        conditional_edit = None
        if not edit_names is None:
            edit_directions = styleclip_global_utils.get_direction(neutral_class, edit_names[0], beta)
            if not conditional_name is None:
                conditional_edit_directions = styleclip_global_utils.get_direction(neutral_class, conditional_name, beta)
            
            if use_stylespace_std:
                s_std = load_stylespace_std()
                edit_directions = to_styles(edit_directions, affine_layers)
                edit = [(s * std).float() for s, std in zip(edit_directions, s_std)]
                if not conditional_name is None:
                    conditional_edit_directions = to_styles(conditional_edit_directions, affine_layers)
                    conditional_edit = [(s * std).float() for s, std in zip(conditional_edit_directions, s_std)]
            else:
                edit = to_styles(edit_directions, affine_layers)
                edit = [edit_i.float() for edit_i in edit]
                if not conditional_name is None:
                    conditional_edit = to_styles(conditional_edit_directions, affine_layers)
                    conditional_edit = [edit_i.float() for edit_i in conditional_edit]
        
        if not edit is None and not conditional_edit is None:
            # subtract the components on conditional_edit direction
            for i in range(len(edit)):
                edit_i_len = torch.norm(edit[i], 2)
                conditional_edit_i_len = torch.norm(conditional_edit[i], 2)
                if edit_i_len > 1e-4 and conditional_edit_i_len > 1e-4:
                    edit[i] /= edit_i_len
                    conditional_edit[i] /= conditional_edit_i_len
                    edit[i] -= torch.dot(edit[i], conditional_edit[i]) * conditional_edit[i]
        
        factors = np.linspace(*edit_range)
        styles = w_to_styles(orig_w, affine_layers)
        final_edits = []

        for factor in factors:
            edited_styles = self._apply_amplification_style(orig_w, affine_layers, edit, factor, origin_pivot_type)
            final_edits.append((edited_styles, edit_names[0], f'{factor}_{beta}'))
        return final_edits, True