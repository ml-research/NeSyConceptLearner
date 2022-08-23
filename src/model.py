import torch
import torch.nn.functional as F
import numpy as np
import math
from torch import nn

class SlotAttention(nn.Module):
    """
    Implementation from https://github.com/lucidrains/slot-attention by lucidrains.
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        # self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim)).abs().to(device='cuda')

        self.project_q = nn.Linear(dim, dim)
        self.project_k = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_inputs = nn.LayerNorm(dim, eps=1e-05)
        self.norm_slots = nn.LayerNorm(dim, eps=1e-05)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-05)

        # dummy initialisation
        self.attn = 0

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_inputs(inputs)
        k, v = self.project_k(inputs), self.project_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_mlp(slots))

        self.attn = attn

        return slots


class SlotAttention_encoder(nn.Module):
    """
    Slot attention encoder for CLEVR as in Locatello et al. 2020 according to the set prediction architecture.
    """
    def __init__(self, in_channels, hidden_channels):
        """
        Builds the Encoder for the set prediction architecture
        :param in_channels: Integer, input channel dimensions to encoder
        :param hidden_channels: Integer, hidden channel dimensions within encoder
        """
        super(SlotAttention_encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class MLP(nn.Module):
    """
    MLP for CLEVR as in Locatello et al. 2020 according to the set prediction architecture.
    """
    def __init__(self, hidden_channels):
        """
        Builds the MLP
        :param hidden_channels: Integer, hidden channel dimensions within encoder, is also equivalent to the input
        channel dims here.
        """
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x):
        return self.network(x)


def build_grid(resolution):
    """
    Builds the grid for the Posisition Embedding.
    :param resolution: Tuple of Ints, in the dimensions of the latent space of the encoder.
    :return: 2D Float meshgrid representing th x y position.
    """
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(nn.Module):
    """
    Adds soft positional embedding with learnable projection.
    """
    def __init__(self, hidden_size, resolution, device="cuda"):
        """Builds the soft position embedding layer.
        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
          device: String specifiying the device, cpu or cuda
        """
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.grid = torch.FloatTensor(build_grid(resolution))
        self.grid = self.grid.to(device)
        self.resolution = resolution[0]
        self.hidden_size = hidden_size

    def forward(self, inputs):
        return inputs + self.dense(self.grid).view((-1, self.hidden_size, self.resolution, self.resolution))


class SlotAttention_classifier(nn.Module):
    """
    The classifier of the set prediction architecture of Locatello et al. 2020
    """
    def __init__(self, in_channels, out_channels):
        """
        Builds the classifier for the set prediction architecture.
        :param in_channels: Integer, input channel dimensions
        :param out_channels: Integer, output channel dimensions
        """
        super(SlotAttention_classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class SlotAttention_model(nn.Module):
    """
    The set prediction slot attention architecture for CLEVR as in Locatello et al 2020.
    """
    def __init__(self, n_slots, n_iters, n_attr, category_ids,
                 in_channels=3,
                 encoder_hidden_channels=64,
                 attention_hidden_channels=128,
                 device="cuda"):
        """
        Builds the set prediction slot attention architecture.
        :param n_slots: Integer, number of slots
        :param n_iters: Integer, number of attention iterations
        :param n_attr: Integer, number of attributes per object to predict
        :param category_ids: List of Integers, specifying the boundaries of each attribute group, e.g. color
        attributes are variables 10 to 17
        :param in_channels: Integer, number of input channel dimensions
        :param encoder_hidden_channels: Integer, number of encoder hidden channel dimensions
        :param attention_hidden_channels: Integer, number of hidden channel dimensions for slot attention module
        :param device: String, either 'cpu' or 'cuda'
        """
        super(SlotAttention_model, self).__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.n_attr = n_attr
        self.category_ids = category_ids
        self.n_attr = n_attr + 1  # additional slot to indicate if it is a object or empty slot
        self.device = device

        self.encoder_cnn = SlotAttention_encoder(in_channels=in_channels, hidden_channels=encoder_hidden_channels)
        self.encoder_pos = SoftPositionEmbed(encoder_hidden_channels, (32, 32), device=device)
        self.layer_norm = nn.LayerNorm(encoder_hidden_channels, eps=1e-05)
        self.mlp = MLP(hidden_channels=encoder_hidden_channels)
        self.slot_attention = SlotAttention(num_slots=n_slots, dim=encoder_hidden_channels, iters=n_iters, eps=1e-8,
                                            hidden_dim=attention_hidden_channels)
        self.mlp_classifier = SlotAttention_classifier(in_channels=encoder_hidden_channels, out_channels=self.n_attr)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_pos(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = self.mlp(x)
        x = self.slot_attention(x)
        x = self.mlp_classifier(x)
        return x

    def _transform_attrs(self, attrs):
        """
        receives the attribute predictions and binarizes them by computing the argmax per attribute group
        :param attrs: 3D Tensor, [batch, n_slots, n_attrs] attribute predictions for a batch.
        :return: binarized attribute predictions
        """
        presence = attrs[:, :, 0]
        attrs_trans = attrs[:, :, 1:]

        # threshold presence prediction, i.e. where is an object predicted
        presence = presence < 0.5

        # flatten first two dims
        attrs_trans = attrs_trans.view(1, -1, attrs_trans.shape[2]).squeeze()
        # binarize attributes
        # set argmax per attr to 1, all other to 0, s.t. only zeros and ones are contained within graph
        # NOTE: this way it is not differentiable!
        bin_attrs = torch.zeros(attrs_trans.shape, device=self.device)
        for i in range(len(self.category_ids) - 1):
            # find the argmax within each category and set this to one
            bin_attrs[range(bin_attrs.shape[0]),
                      # e.g. x[:, 0:(3+0)], x[:, 3:(5+3)], etc
                      (attrs_trans[:,
                       self.category_ids[i]:self.category_ids[i + 1]].argmax(dim=1) + self.category_ids[i]).type(
                          torch.LongTensor)] = 1

        # reshape back to batch x n_slots x n_attrs
        bin_attrs = bin_attrs.view(attrs.shape[0], attrs.shape[1], attrs.shape[2] - 1)

        # add coordinates back
        bin_attrs[:, :, :3] = attrs[:, :, 1:4]

        # redo presence zeroing
        bin_attrs[presence, :] = 0

        return bin_attrs


############
# Transformers #
############
"""
Code largely from https://github.com/juho-lee/set_transformer by yoonholee.
"""
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    """
    Set Transformer used for the Neuro-Symbolic Concept Learner.
    """
    def __init__(self, dim_input=3, dim_output=40, dim_hidden=128, num_heads=4, ln=False):
        """
        Builds the Set Transformer.
        :param dim_input: Integer, input dimensions
        :param dim_output: Integer, output dimensions
        :param dim_hidden: Integer, hidden layer dimensions
        :param num_heads: Integer, number of attention heads
        :param ln: Boolean, whether to use Layer Norm
        """
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=dim_input, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
            SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim=dim_hidden, num_heads=num_heads, num_seeds=1, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze()

############
# NeSyConceptLearner #
############
class NeSyConceptLearner(nn.Module):
    """
    The Neuro-Symbolic Concept Learner of Stammer et al. 2021 based on Slot Attention and Set Transformer.
    """
    def __init__(self, n_classes, n_slots=1, n_iters=3, n_attr=18, n_set_heads=4, set_transf_hidden=128,
                 category_ids=[3, 6, 8, 10, 17], device='cuda'):
        """

        :param n_classes: Integer, number of classes
        :param n_slots: Integer, number of slots for slot attention module
        :param n_iters: Integer, number of attention iterations for slot attentions
        :param n_attr: Integer, number of attributes per object
        :param n_set_heads: Integer, number of attention heads for set transformer
        :param set_transf_hidden: Integer, hidden dim of set transformer
        :param category_ids: List of Integers, specifying the starting ids of the attribute groups for the attribute
        prediction
        :param device: String, eihter 'cpu' or 'cuda'
        """
        super().__init__()
        self.device = device
        # Concept Embedding Module
        self.img2state_net = SlotAttention_model(n_slots, n_iters, n_attr, encoder_hidden_channels=64,
                                                 attention_hidden_channels=128, category_ids=category_ids,
                                                 device=device)
        # Reasoning module
        self.set_cls = SetTransformer(dim_input=n_attr, dim_hidden=set_transf_hidden, num_heads=n_set_heads,
                                      dim_output=n_classes, ln=True)

    def forward(self, img):
        """
        Receives an image, passes it through the concept embedding module and the reasoning module. For simplicity we
        here binarize the continuous output of the concept embedding module before passing it to the reasoning module.
        The ouputs of both modules are returned, i.e. the final class prediction and the latent symbolic representation.
        :param img: 4D Tensor [batch, channels, width, height]
        :return: Tuple of outputs of both modules, [batch, n_classes] classification/ reasoning module output,
        [batch, n_slots, n_attr] concept embedding module output/ symbolic representation
        """
        attrs = self.img2state_net(img)
        # binarize slot attention output, apart from coordinate output
        attrs_trans = self.img2state_net._transform_attrs(attrs)
        # run through classifier via set transformer
        cls = self.set_cls(attrs_trans)

        return cls.squeeze(), attrs_trans


if __name__ == "__main__":
    x = torch.rand(20, 3, 128, 128)
    net = NeSyConceptLearner(n_classes=3, n_slots=10, n_iters=3, n_attr=18, n_set_heads=4, set_transf_hidden=128,
                             category_ids = [3, 6, 8, 10, 17], device='cpu')
    output = net(x)
