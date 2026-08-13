import torch

from embodi.canonical import matrix_to_rotation_6d, reanchor_canonical_actions, rotation_6d_to_matrix


def test_rotation_6d_round_trip() -> None:
    matrix = torch.eye(3).unsqueeze(0)
    encoded = matrix_to_rotation_6d(matrix)
    torch.testing.assert_close(encoded, torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))
    torch.testing.assert_close(rotation_6d_to_matrix(encoded), matrix)


def test_future_local_actions_are_reanchored_to_first_state() -> None:
    actions = torch.zeros(1, 2, 25)
    actions[:, :, 3] = 1
    actions[:, :, 7] = 1
    actions[:, :, 13] = 1
    actions[:, :, 17] = 1
    actions[0, 0, 0] = 0.01
    actions[0, 1, 0] = 0.02
    states = torch.zeros(1, 2, 4, 10)
    states[:, :, 0, 3] = 1
    states[:, :, 0, 7] = 1
    states[:, :, 1, 3] = 1
    states[:, :, 1, 7] = 1
    states[0, 1, 0, 0] = 0.1
    anchored, initial_state = reanchor_canonical_actions(actions, states)
    torch.testing.assert_close(anchored[0, :, 0], torch.tensor([0.01, 0.12]))
    torch.testing.assert_close(initial_state, states[:, 0])
