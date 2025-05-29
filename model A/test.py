import torch





tnsrs = [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1]), torch.tensor([0, 1, 2, 3, 4, 5, 6])]

n_tensrs = zero_fills(tnsrs)

print(n_tensrs[0])
print(n_tensrs[1])