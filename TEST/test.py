import sys

sys.path.append('./')
sys.path.append('../')

import torch
import numpy as np
from editings.latent_editor import LatentEditor

def is_parallel(v1, v2, return_cosine=False):
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)

    if np.abs(l1) > 1e-6 and np.abs(l2) > 1e-6:
        v1_norm = v1 / l1
        v2_norm = v2 / l2

        dot_product = np.dot(v1_norm, v2_norm)
        ret = np.abs(dot_product - 1) < 1e-6
    else:
        ret = True
        dot_product = 0

    if not return_cosine:
        return ret
    else:
        return ret, dot_product


def test_diff_direction_parallel(w0, w1, direction):
    diff = w1 - w0
    for i in range(diff.shape[0]):
        is_para, cosine = is_parallel(direction[i], diff[i], True)

        if not is_para:
            print(f"Test didn't pass at layer {i+1}: Not Parallel. -> Cosine: {cosine}")
            return False

    print("Test Passed!")
    return True



def test_latent_amplification(test_data_path, direction_name, test_times):
    w_vectors = np.load(test_data_path)
    direction = np.load(f'editings/w_directions/{direction_name}.npy')
    w_vectors_cuda = torch.from_numpy(w_vectors).cuda()
    latent_editor = LatentEditor()
    edit_range = (1.5, 1.5, 1)
    edits, _ = latent_editor.get_amplification_edits(w_vectors_cuda, [direction_name], edit_range)
    edited_w_vectors_cuda, _, _ = edits[0]
    edited_w_vectors = edited_w_vectors_cuda.detach().cpu().numpy()

    frame_num = w_vectors.shape[0]
    layer_num = 18
    test_num = 0
    passed_test_num = 0
    test_frames = np.random.choice(frame_num-1, size=test_times, replace=False)+1
    for ti in range(test_times):
        print(f"================== TEST AMPLIFICATOIN ({ti+1}/{test_times}) ==================")
        print(f"TEST DATA: {test_data_path}")
        print(f"TEST DIRECTION: {direction_name}")
        fi = test_frames[ti]
        print(f"TEST FRAME: {fi+1}")
        test_passed = test_diff_direction_parallel(w_vectors[fi], edited_w_vectors[fi], direction)
        test_num += 1
        if test_passed:
            passed_test_num += 1

    print(f"================== SUMMARY ==================")
    print(f"Passed Tests: {passed_test_num}/{test_num}")
    pass

def _main():
    test_latent_amplification('w_vectors/e4e_yukun_happy_anger.npy', 'e4e_Happy_4', 10)

if __name__ == "__main__":
    _main()