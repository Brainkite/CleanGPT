# import torch

# class JEST:
#     def __init__(self, model, ref_scores, B, T, batch_size, super_batch_size, n_chunks=16, proc_rank=0, num_proc=1):
#         self.model = model
#         self.ref_scores = ref_scores
#         self.B = B
#         self.T = T
#         self.batch_size = batch_size
#         self.super_batch_size = super_batch_size
#         self.n_chunks = n_chunks
#         self.proc_rank = proc_rank
#         self.num_proc = num_proc
#         assert batch_size % (B * num_proc) == 0, print(batch_size , B, num_proc)
#         assert super_batch_size % batch_size == 0, print(super_batch_size, batch_size)
#         self.super_batch_steps = super_batch_size

#     def compute_learnability_scores(self, train_loader):
#         with torch.no_grad():
#             learner_loss = self.model.loss(super_batch)
#             reference_loss = self.reference_model.loss(super_batch)
#         return learner_loss - reference_loss

#     def select_batch(self, super_batch):
#         scores = self.compute_learnability_scores(super_batch)
#         selected_indices = []
#         chunk_size = self.batch_size // self.n_chunks

#         for _ in range(self.n_chunks):
#             available_indices = set(range(self.super_batch_size)) - set(selected_indices)
#             chunk_scores = scores[list(available_indices)]
#             chunk_indices = torch.topk(chunk_scores, chunk_size).indices
#             selected_indices.extend(chunk_indices)

#         return super_batch[selected_indices]

#     def train_step(self, super_batch, optimizer):
#         selected_batch = self.select_batch(super_batch)
#         optimizer.zero_grad()
#         loss = self.model.loss(selected_batch)
#         loss.backward()
#         optimizer.step()
#         return loss.item()