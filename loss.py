import torch.nn.functional as F
import torch
import args


# depth_contrastive_encoder: NT-Xent loss
def nt_xent_loss(z, B, views, temperature):
    labels = torch.cat([torch.arange(B) for _ in range(views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
    labels = labels.to(args.device)

    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)
    print(similarity_matrix.shape)
    # remove diag
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    print(similarity_matrix.shape)
    # select pos example and neg example
    pos = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    neg = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # cal logits
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)
    logits = logits / temperature
    return logits, labels
