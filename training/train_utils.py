def to_list(tensor):
    return tensor.detach().cpu().tolist()


