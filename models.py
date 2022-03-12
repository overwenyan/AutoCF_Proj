import torch
import torch.nn as nn
import torch.nn.functional as F


CF_MODEL = ['ui', 'ur', 'ri', 'rr'] # Input Encoding
CF_EMB = ['mat', 'mlp'] # Embedding Function
CF_IFC = ['max', 'min', 'plus', 'mul', 'concat'] # Interacton
CF_PRED = ['i', 'h', 'mlp'] # Prediction Function

SPACE = (1 + 2 + 2 + 4) * len(CF_IFC) * len(CF_PRED)

OPS = {
    'plus': lambda p, q: p + q,
    'multiply': lambda p, q: p * q,
    'max': lambda p, q: torch.max(torch.stack((p, q)), dim=0)[0],
    'min': lambda p, q: torch.min(torch.stack((p, q)), dim=0)[0],
    'concat': lambda p, q: torch.cat([p, q], dim=-1),
    'norm_0': lambda p: torch.ones_like(p),
    'norm_0.5': lambda p: torch.sqrt(torch.abs(p) + 1e-7),
    'norm_1': lambda p: torch.abs(p),
    'norm_2': lambda p: p ** 2,
    'I': lambda p: torch.ones_like(p),
    '-I': lambda p: -torch.ones_like(p),
    'sign': lambda p: torch.sign(p),
}# 表示不同的算符

SPACE_CF = len(CF_MODEL) * len(CF_EMB) * len(CF_IFC) * len(CF_PRED) # =120

OPS_CF = {
    'plus': lambda p, q: p + q,
    'mul': lambda p, q: p * q,
    'max': lambda p, q: torch.max(torch.stack((p, q)), dim=0)[0],
    'min': lambda p, q: torch.min(torch.stack((p, q)), dim=0)[0],
    'concat': lambda p, q: torch.cat([p, q], dim=-1)
}


def constrain(p):
    c = torch.norm(p, p=2, dim=1, keepdim=True)
    c[c < 1] = 1.0
    p.data.div_(c)


class Virtue_CF(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(Virtue_CF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.reg = reg
        self._UsersEmbedding = nn.Embedding(num_users, embedding_dim)
        self._ItemsEmbedding = nn.Embedding(num_items, embedding_dim)

        self._UsersRatingsEmbedding = nn.Embedding(
            num_items+1, embedding_dim, padding_idx=0)
        self._ItemsRatingsEmbedding = nn.Embedding(
            num_users+1, embedding_dim, padding_idx=0)
        self.pairwise_loss = nn.BCEWithLogitsLoss(size_average=False)

    def compute_loss(self, inferences, labels, regs):
        labels = torch.reshape(labels, [-1, 1])
        loss = F.mse_loss(inferences, labels)
        return loss + regs
    
    # modified
    def forward_pair(self, users, items, negs, users_sparse_ratings, items_sparse_ratings):
        pos_inferences, pos_regs = self.forward(users, items, users_sparse_ratings, items_sparse_ratings)
        neg_inferences, neg_regs = self.forward(users, negs, users_sparse_ratings, items_sparse_ratings)
        inferences = pos_inferences - neg_inferences
        return inferences, pos_regs + neg_regs

    def compute_loss_pair(self, inferences, regs):
        labels = torch.ones(inferences.size()[0], device=inferences.device)
        labels = torch.reshape(labels, [-1, 1])
        loss = self.pairwise_loss(inferences, labels) / (inferences.size()[0])
        return loss


class BaseModel(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.reg = reg
        self._UsersEmbedding = nn.Embedding(num_users, embedding_dim)
        self._ItemsEmbedding = nn.Embedding(num_items, embedding_dim)

        self._UsersRatingsEmbedding = nn.Parameter(torch.FloatTensor(num_items, embedding_dim))
        nn.init.xavier_normal_(self._UsersRatingsEmbedding)
        self._ItemsRatingsEmbedding = nn.Parameter(torch.FloatTensor(num_users, embedding_dim))
        nn.init.xavier_normal_(self._ItemsRatingsEmbedding)
        self.pairwise_loss = nn.BCEWithLogitsLoss(size_average=False)

    def compute_loss(self, inferences, labels, regs):
        labels = torch.reshape(labels, [-1, 1])
        loss = F.mse_loss(inferences, labels)
        return loss + regs
    
    # modified
    def forward_pair(self, users, items, negs, users_sparse_ratings, items_sparse_ratings):
        pos_inferences, pos_regs = self.forward(users, items, users_sparse_ratings, items_sparse_ratings)
        neg_inferences, neg_regs = self.forward(users, negs, users_sparse_ratings, items_sparse_ratings)
        inferences = pos_inferences - neg_inferences
        return inferences, pos_regs + neg_regs

    def compute_loss_pair(self, inferences, regs):
        labels = torch.ones(inferences.size()[0], device=inferences.device)
        labels = torch.reshape(labels, [-1, 1])
        loss = self.pairwise_loss(inferences, labels) / (inferences.size()[0])
        return loss

# inference, regs
class SVD_plus_plus(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(SVD_plus_plus, self).__init__(
            num_users, num_items, embedding_dim, reg)
        self._W1, self._W4 = 1, 1

    def forward(self, users, items, users_ratings, items_ratings):
        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        users_ratings_embedding = self._UsersRatingsEmbedding(users_ratings)
        users_ratings_embedding_normed = torch.mean(
            users_ratings_embedding, dim=1)

        self._out1 = users_embedding * items_embedding
        self._out4 = users_ratings_embedding_normed * items_embedding
        inference = self._W1 * \
            self._out1.sum(dim=1, keepdim=True) + self._W4 * \
            self._out4.sum(dim=1, keepdim=True)
        regs = self.reg * (torch.norm(items_embedding) +
                           torch.norm(users_embedding))
        return inference, regs


class FISM_Old(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(FISM_Old, self).__init__(num_users, num_items, embedding_dim, reg)
        self._W4 = 1

    def forward(self, users, items, users_ratings, items_ratings):
        items_embedding = self._ItemsEmbedding(items)
        users_ratings_embedding = self._UsersRatingsEmbedding(users_ratings)
        users_ratings_embedding_normed = torch.mean(
            users_ratings_embedding, dim=1)

        self._out4 = users_ratings_embedding_normed * items_embedding
        inference = self._W4 * self._out4.sum(dim=1, keepdim=True)
        regs = self.reg * torch.norm(items_embedding)
        return inference, regs

class FISM(BaseModel):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(FISM, self).__init__(num_users, num_items, embedding_dim, reg)
        self._W4 = 1

    def forward(self, users, items, users_ratings, items_ratings):
        items_embedding = self._ItemsEmbedding(items)
        users_ratings = users_ratings.cuda() # modified
        
        alluser_sparse_embeddings = torch.sparse.mm(users_ratings, self._UsersRatingsEmbedding)
        users_ratings_embedding = alluser_sparse_embeddings[users]
        users_ratings_embedding_normed = torch.mean(
            users_ratings_embedding, dim=1, keepdim=True)

        self._out4 = users_ratings_embedding_normed * items_embedding
        inference = self._W4 * self._out4.sum(dim=1, keepdim=True)
        regs = self.reg * torch.norm(items_embedding)
        return inference, regs

class JNCF_Dot(BaseModel):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(JNCF_Dot, self).__init__(num_users, num_items, embedding_dim, reg)
        self._W4 = 1
        self._FC1 = nn.Sequential(
            nn.Linear(num_items, embedding_dim, bias=True),
        )
        self._FC2 = nn.Sequential(
            nn.Linear(num_users, embedding_dim, bias=True),
        )
        self._W1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))

        self.num_users = num_users
        self.num_items = num_items

    def forward(self, users, items, users_ratings, items_ratings):
        items_embedding = self._ItemsEmbedding(items)
        alluser_sparse_embeddings = self._FC1(users_ratings)
        users_ratings_embedding_normed = alluser_sparse_embeddings[users]
        allitem_sparse_embeddings = self._FC2(items_ratings)
        items_ratings_embedding_normed = allitem_sparse_embeddings[items]

        _out1 = users_ratings_embedding_normed * items_ratings_embedding_normed
        inference = self._W1(_out1).sum(dim=1, keepdim=True)
        regs = self.reg * 0
        return inference, regs

class SVD(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(SVD, self).__init__(num_users, num_items, embedding_dim, reg)
        self._W1 = 1
        self.pairwise_loss = nn.BCEWithLogitsLoss(size_average=False)

    def forward(self, users, items, users_ratings, items_ratings):
        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        self._out1 = users_embedding * items_embedding
        inference = self._W1 * self._out1.sum(dim=1, keepdim=True)
        regs = self.reg * (torch.norm(users_embedding) +
                           torch.norm(items_embedding))
        return inference, regs

    def forward_pair(self, users, items, negs, users_sparse_ratings, items_sparse_ratings):
        pos_inferences, pos_regs = self.forward(users, items, users_sparse_ratings, items_sparse_ratings)
        neg_inferences, neg_regs = self.forward(users, negs, users_sparse_ratings, items_sparse_ratings)
        inferences = pos_inferences - neg_inferences
        return inferences, pos_regs + neg_regs

    def compute_loss_pair(self, inferences, regs):
        labels = torch.ones(inferences.size()[0], device=inferences.device)
        labels = torch.reshape(labels, [-1, 1])
        loss = self.pairwise_loss(inferences, labels) / (inferences.size()[0])
        return loss


class GMF(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(GMF, self).__init__(num_users, num_items, embedding_dim, reg)
        self._W1 = nn.Linear(embedding_dim, 1, bias=False)
        self.pairwise_loss = nn.BCEWithLogitsLoss(size_average=False)

    def forward(self, users, items, users_ratings, items_ratings):
        constrain(next(self._W1.parameters()))

        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        self._out1 = users_embedding * items_embedding
        inference = self._W1(self._out1).sum(dim=1, keepdim=True)
        regs = self.reg * (torch.norm(users_embedding) +
                           torch.norm(items_embedding))
        return inference, regs
    # modified
    def forward_pair(self, users, items, negs, users_sparse_ratings, items_sparse_ratings):
        pos_inferences, pos_regs = self.forward(users, items, users_sparse_ratings, items_sparse_ratings)
        neg_inferences, neg_regs = self.forward(users, negs, users_sparse_ratings, items_sparse_ratings)
        inferences = pos_inferences - neg_inferences
        return inferences, pos_regs + neg_regs

    def compute_loss_pair(self, inferences, regs):
        labels = torch.ones(inferences.size()[0], device=inferences.device)
        labels = torch.reshape(labels, [-1, 1])
        loss = self.pairwise_loss(inferences, labels) / (inferences.size()[0])
        return loss


class MLP(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(MLP, self).__init__(num_users, num_items, embedding_dim, reg)
        self._W1 = nn.Sequential(
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))

    def forward(self, users, items, users_ratings, items_ratings):
        constrain(next(self._W1.parameters()))

        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        mlp_out = torch.cat([users_embedding, items_embedding], dim=-1)
        inference = self._W1(F.tanh(mlp_out)).sum(dim=1, keepdim=True)
        regs = self.reg * (torch.norm(users_embedding) +
                           torch.norm(items_embedding))
        return inference, regs


class DMF(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(DMF, self).__init__(num_users, num_items, embedding_dim, reg)
        self._FC1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))
        self._FC2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))
        self._W1 = 1

    def forward(self, users, items, users_ratings, items_ratings):

        users_ratings_embedding = self._UsersRatingsEmbedding(users_ratings)
        users_ratings_embedding_normed = self._FC1(users_ratings_embedding)

        items_ratings_embedding = self._ItemsRatingsEmbedding(items_ratings)
        items_ratings_embedding_normed = self._FC2(items_ratings_embedding)

        self._out1 = users_ratings_embedding_normed * items_ratings_embedding_normed
        inference = self._W1 * self._out1.sum(dim=1, keepdim=True)
        regs = self.reg * 0
        return inference, regs


class CML(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(CML, self).__init__(num_users, num_items, embedding_dim, reg)
        self._W1 = 1

    def forward(self, users, items, users_ratings, items_ratings):

        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        self._out1 = users_embedding - items_embedding
        inference = self._W1 * torch.norm(self._out1, dim=1, keepdim=True)
        regs = self.reg * (torch.norm(users_embedding) +
                           torch.norm(items_embedding))
        return inference, regs


class JNCF_Cat(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(JNCF_Cat, self).__init__(
            num_users, num_items, embedding_dim, reg)
        self._FC1 = nn.Sequential(
            nn.Linear(num_items, embedding_dim),
        )
        self._FC2 = nn.Sequential(
            nn.Linear(num_users, embedding_dim),
        )
        self._W1 = nn.Sequential(
            nn.Linear(2*embedding_dim, 1, bias=False))
        self.num_users = num_users
        self.num_items = num_items

    def forward(self, users, items, users_sparse_ratings, items_sparse_ratings):
        constrain(next(self._FC1.parameters()))
        constrain(next(self._FC2.parameters()))
        constrain(next(self._W1.parameters()))

        users_ratings_embedding_normed = self._FC1(
            users_sparse_ratings.float())
        items_ratings_embedding_normed = self._FC2(
            items_sparse_ratings.float())

        _out1 = torch.cat([users_ratings_embedding_normed,
                           items_ratings_embedding_normed], dim=-1)
        inference = self._W1(_out1).sum(dim=1, keepdim=True)
        regs = self.reg * 0
        return inference, regs


class JNCF_Dot_Old(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(JNCF_Dot_Old, self).__init__(
            num_users, num_items, embedding_dim, reg)
        self._FC1 = nn.Sequential(
            nn.Linear(num_items, embedding_dim, bias=True),
        )
        self._FC2 = nn.Sequential(
            nn.Linear(num_users, embedding_dim, bias=True),
        )
        self._W1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))

        self.num_users = num_users
        self.num_items = num_items

    def forward(self, users, items, users_sparse_ratings, items_sparse_ratings):

        users_ratings_embedding_normed = self._FC1(
            users_sparse_ratings.float())
        items_ratings_embedding_normed = self._FC2(
            items_sparse_ratings.float())
        _out1 = users_ratings_embedding_normed * items_ratings_embedding_normed
        inference = self._W1(_out1).sum(dim=1, keepdim=True)
        regs = self.reg * 0
        return inference, regs


class DELF(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, reg):
        super(DELF, self).__init__(num_users, num_items, embedding_dim, reg)
        self._W1 = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))
        self._W2 = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))
        self._W3 = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))
        self._W4 = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1, bias=False))

    def forward(self, users, items, users_ratings, items_ratings):
        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)
        # users_embedding = users_embedding.to_sparse().to_dense()
        users_ratings_embedding = self._UsersRatingsEmbedding(users_ratings)
        items_ratings_embedding = self._ItemsRatingsEmbedding(items_ratings)
        users_ratings_embedding_normed = torch.mean(
            users_ratings_embedding, dim=1)
        items_ratings_embedding_normed = torch.mean(
            items_ratings_embedding, dim=1)

        self._out1 = torch.cat([users_embedding, items_embedding], dim=-1)
        self._out2 = torch.cat(
            [users_embedding, items_ratings_embedding_normed], dim=-1)
        self._out3 = torch.cat(
            [users_ratings_embedding_normed, items_embedding], dim=-1)
        self._out4 = torch.cat(
            [users_ratings_embedding_normed, items_ratings_embedding_normed], dim=-1)
        inference = (self._W1(self._out1) + self._W2(self._out2) +
                     self._W3(self._out3) + self._W4(self._out4)).sum(dim=1, keepdim=True)
        regs = self.reg * (torch.norm(items_embedding) +
                           torch.norm(users_embedding))
        return inference, regs


class single_model_old(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, arch, reg):
        super(single_model_old, self).__init__(
            num_users, num_items, embedding_dim, reg)
        self.arch = arch
        self._FC1 = nn.Sequential(
            nn.Linear(num_items, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False))
        self._FC2 = nn.Sequential(
            nn.Linear(num_users, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False))

        if arch['ifc'] == 'concat':
            w_size = 2 * embedding_dim
        else:
            w_size = embedding_dim
        if arch['pred'] == 'i':
            self._W = torch.ones(w_size, 1)
        elif arch['pred'] == 'h':
            self._W = nn.Linear(w_size, 1, bias=False)
        else:
            self._W = nn.Sequential(
                nn.Linear(w_size, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1, bias=False))

    def forward(self, users, items, users_ratings, items_ratings, users_sparse_ratings, items_sparse_ratings):
        constrain(next(self._FC1.parameters()))
        constrain(next(self._FC2.parameters()))
        # constrain(next(self._W.parameters()))
        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        if self.arch['emb']['u'] == 'mlp':
            users_ratings_embedding_normed = self._FC1(users_sparse_ratings)
        else:
            users_ratings_embedding = self._UsersRatingsEmbedding(
                users_ratings)
            users_ratings_embedding_normed = torch.mean(
                users_ratings_embedding, dim=1)

        if self.arch['emb']['i'] == 'mlp':
            items_ratings_embedding_normed = self._FC2(items_sparse_ratings)
        else:
            items_ratings_embedding = self._ItemsRatingsEmbedding(
                items_ratings)
            items_ratings_embedding_normed = torch.mean(
                items_ratings_embedding, dim=1)

        if self.arch['cf'] == 'ui':
            _out = OPS_CF[self.arch['ifc']](users_embedding, items_embedding)
        elif self.arch['cf'] == 'ur':
            _out = OPS_CF[self.arch['ifc']](
                users_embedding, items_ratings_embedding_normed)
        elif self.arch['cf'] == 'ri':
            _out = OPS_CF[self.arch['ifc']](
                users_ratings_embedding_normed, items_embedding)
        else:
            _out = OPS_CF[self.arch['ifc']](
                users_ratings_embedding_normed, items_ratings_embedding_normed)

        if self.arch['pred'] == 'i':
            inferences = _out.sum(dim=1, keepdim=True)
        else:
            inferences = self._W(_out)
        regs = self.reg * 0

        return inferences, regs


class single_model(BaseModel):
    def __init__(self, num_users, num_items, embedding_dim, arch, reg):
        super(single_model, self).__init__(
            num_users, num_items, embedding_dim, reg)
        self.pairwise_loss = nn.BCEWithLogitsLoss(size_average=False)
        self.arch = arch
        self.data_type = 'explicit'
        self._FC1 = nn.Sequential(
            nn.Linear(num_items, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False))
        self._FC2 = nn.Sequential(
            nn.Linear(num_users, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=False))

        if arch['ifc'] == 'concat':
            w_size = 2 * embedding_dim
        else:
            w_size = embedding_dim
        if arch['pred'] == 'i':
            self._W = torch.ones(w_size, 1)
        elif arch['pred'] == 'h':
            self._W = nn.Linear(w_size, 1, bias=False)
        else:
            self._W = nn.Sequential(
                nn.Linear(w_size, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1, bias=False))

    def compute_loss(self, inferences, labels, regs):
        labels = torch.reshape(labels, [-1, 1])
        if self.data_type == 'explicit':
            loss = F.mse_loss(inferences, labels)
        else:
            loss = F.binary_cross_entropy_with_logits(inferences, labels)
        return loss + regs

    def forward(self, users, items, users_sparse_ratings, items_sparse_ratings):
        constrain(next(self._FC1.parameters()))
        constrain(next(self._FC2.parameters()))
        users_embedding = self._UsersEmbedding(users)
        items_embedding = self._ItemsEmbedding(items)

        if self.arch['emb']['u'] == 'mlp':
            alluser_sparse_embeddings = self._FC1(users_sparse_ratings)
            users_ratings_embedding_normed = alluser_sparse_embeddings[users]
        else:
            alluser_sparse_embeddings = torch.sparse.mm(users_sparse_ratings, self._UsersRatingsEmbedding)
            users_ratings_embedding_normed = alluser_sparse_embeddings[users]
        if self.arch['emb']['i'] == 'mlp':
            allitem_sparse_embeddings = self._FC2(items_sparse_ratings)
            items_ratings_embedding_normed = allitem_sparse_embeddings[items]
        else:
            allitem_sparse_embeddings = torch.sparse.mm(items_sparse_ratings, self._ItemsRatingsEmbedding)
            items_ratings_embedding_normed = allitem_sparse_embeddings[items]

        if self.arch['cf'] == 'ui':
            _out = OPS_CF[self.arch['ifc']](users_embedding, items_embedding)
        elif self.arch['cf'] == 'ur':
            _out = OPS_CF[self.arch['ifc']](
                users_embedding, items_ratings_embedding_normed)
        elif self.arch['cf'] == 'ri':
            _out = OPS_CF[self.arch['ifc']](
                users_ratings_embedding_normed, items_embedding)
        else:
            _out = OPS_CF[self.arch['ifc']](
                users_ratings_embedding_normed, items_ratings_embedding_normed)

        if self.arch['pred'] == 'i':
            inferences = _out.sum(dim=1, keepdim=True)
        else:
            inferences = self._W(_out)
        regs = self.reg * 0

        return inferences, regs

    def forward_pair(self, users, items, negs, users_sparse_ratings, items_sparse_ratings):
        pos_inferences, pos_regs = self.forward(users, items, users_sparse_ratings, items_sparse_ratings)
        neg_inferences, neg_regs = self.forward(users, negs, users_sparse_ratings, items_sparse_ratings)
        inferences = pos_inferences - neg_inferences
        return inferences, pos_regs + neg_regs

    def compute_loss_pair(self, inferences, regs):
        labels = torch.ones(inferences.size()[0], device=inferences.device)
        labels = torch.reshape(labels, [-1, 1])
        loss = self.pairwise_loss(inferences, labels) / (inferences.size()[0])
        return loss


class Network_Single_CF_Signal(Virtue_CF):
    def __init__(self, num_users, num_items, embedding_dim, arch_list, reg):
        super(Network_Single_CF_Signal, self).__init__(
            num_users, num_items, embedding_dim, reg)
        self._param_list = dict()
        self._arch_list = arch_list
        self._weight = self._arch_list['weight']
        for cf_signal in self._arch_list:
            if cf_signal == 'weight':
                continue
            self.arch = self._arch_list[cf_signal]
            self.param = dict()
            self.param['_FC1'] = nn.Sequential(
                nn.Linear(num_items, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim, bias=False)).cuda()
            self.param['_FC2'] = nn.Sequential(
                nn.Linear(num_users, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim, bias=False)).cuda()
            if self.arch['ifc'] == 'concat':
                self.w_size = 2 * embedding_dim
            else:
                self.w_size = embedding_dim
            if self.arch['pred'] == 'i':
                self.param['_W'] = torch.ones(self.w_size, 1).cuda()
            elif self.arch['pred'] == 'h':
                self.param['_W'] = nn.Linear(self.w_size, 1, bias=False).cuda()
            else:
                self.param['_W'] = nn.Sequential(
                    nn.Linear(self.w_size, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, 1, bias=False)).cuda()
            self._param_list[cf_signal] = self.param

    def forward(self, users, items, users_ratings, items_ratings, users_sparse_ratings, items_sparse_ratings):
        self.users_embedding = self._UsersEmbedding(users)
        self.items_embedding = self._ItemsEmbedding(items)

        self.inferences_total = []
        self.signal_list = ['ui', 'ur', 'ri', 'rr']

        for i in range(4):
            self.arch = self._arch_list[self.signal_list[i]]
            self.param = self._param_list[self.signal_list[i]]
            if self.arch['emb']['u'] == 'mlp':
                self.users_ratings_embedding_normed = self.param['_FC1'](
                    users_sparse_ratings)
            else:
                self.users_ratings_embedding = self._UsersRatingsEmbedding(
                    users_ratings)
                self.users_ratings_embedding_normed = torch.mean(
                    self.users_ratings_embedding, dim=1)

            if self.arch['emb']['i'] == 'mlp':
                self.items_ratings_embedding_normed = self.param['_FC2'](
                    items_sparse_ratings)
            else:
                self.items_ratings_embedding = self._ItemsRatingsEmbedding(
                    items_ratings)
                self.items_ratings_embedding_normed = torch.mean(
                    self.items_ratings_embedding, dim=1)

            if self.arch['cf'] == 'ui':
                self._out = OPS_CF[self.arch['ifc']](
                    self.users_embedding, self.items_embedding)
            elif self.arch['cf'] == 'ur':
                self._out = OPS_CF[self.arch['ifc']](
                    self.users_embedding, self.items_ratings_embedding_normed)
            elif self.arch['cf'] == 'ri':
                self._out = OPS_CF[self.arch['ifc']](
                    self.users_ratings_embedding_normed, self.items_embedding)
            else:
                self._out = OPS_CF[self.arch['ifc']](
                    self.users_ratings_embedding_normed, self.items_ratings_embedding_normed)

            if self.arch['pred'] == 'i':
                self.inferences = self._out.sum(dim=1, keepdim=True)
            else:
                self.inferences = self.param['_W'](self._out)
            self.inferences_total.append(self.inferences)
        self.regs = self.reg * \
            (torch.norm(self.users_embedding) + torch.norm(self.items_embedding))
        self.inferences_total = self.inferences_total[0] * self._weight[0] + self.inferences_total[1] * self._weight[1] \
            + self.inferences_total[2] * self._weight[2] + \
            self.inferences_total[3] * self._weight[3]
        return self.inferences_total, self.regs


if __name__ == '__main__':
    pass
