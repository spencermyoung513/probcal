import torch


def extract_state_dict(model_chkp_path: str, state_dict_path: str):
    """
    Extracts the state dictionary from a saved PyTorch model checkpoint and saves it to a specified path.

    Args:
        model_chkp_path (str): The path to the saved PyTorch model checkpoint.
        state_dict_path (str): The path where the extracted state dictionary should be saved.

    Example:
        extract_state_dict('path/to/model/checkpoint.pth', 'path/to/save/state_dict.pth')

    Note:
        The saved state dictionary can be loaded with `torch.load('path/to/save/state_dict.pth')`.
    """
    m = torch.load(model_chkp_path, map_location="cpu")
    state_dict = m["state_dict"]
    torch.save(state_dict, state_dict_path)
