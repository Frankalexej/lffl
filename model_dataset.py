import torch





def extract_last_from_packed(packed, lengths:torch.Tensor): 
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    ))
    sorted_lengths = lengths[packed.sorted_indices]
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0))
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items