import torch

z = torch.randn((3*2, 2))
print(z)
labels = torch.cat([torch.arange(3) for _ in range(2)])

labels = (labels.unsqueeze(0) == labels.unsqueeze(1))

mask = torch.eye(labels.shape[0], dtype=torch.bool)

labels = labels[~mask].view(labels.shape[0], -1)


similarity_matrix = torch.matmul(z, z.T)
print(similarity_matrix)
similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
print(similarity_matrix)
pos = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
neg = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
print(pos)
print(neg)
logits = torch.cat([pos, neg], dim=1)
labels = torch.zeros(logits.shape[0], dtype=torch.long)