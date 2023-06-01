from torchsummary import summary
from u2net import Block9


rbc = Block9().to("cuda")
print(summary(rbc, (128,288,288)))

# import torch
# a = torch.tensor([[1,2],[1,2]])
# b = torch.tensor([[3,4], [3,4]])
# print(a)
# print(b)
# print(a+b)
# import torch 
# a = torch.ones(1,1,1)
# b = torch.ones(1,1,1)
# c = torch.cat((a,b),1)
# print(c.shape)