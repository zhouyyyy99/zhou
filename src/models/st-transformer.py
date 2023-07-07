from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    # print('------------- in model factory ------------')
    # print(data.feature_df)
    # print('-------------------------------------------')
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print(
                "Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "imputation") or (task == "transduction"):
        if config['model'] == 'LINEAR':
            pass
            # return DummyTSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
            #                                  config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
            #                                  pos_encoding=config['pos_encoding'], activation=config['activation'],
            #                                  norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'transformer':
            pass
            return TSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                        config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                        pos_encoding=config['pos_encoding'], activation=config['activation'],
                                        norm=config['normalization_layer'], freeze=config['freeze'])

    if (task == "classification") or (task == "regression"):
        num_labels = len(data.class_names) if task == "classification" else data.labels_df.shape[
            1]  # dimensionality of labels
        if config['model'] == 'LINEAR':
            pass
            # return DummyTSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
            #                                                 config['num_heads'],
            #                                                 config['num_layers'], config['dim_feedforward'],
            #                                                 num_classes=num_labels,
            #                                                 dropout=config['dropout'], pos_encoding=config['pos_encoding'],
            #                                                 activation=config['activation'],
            #                                                 norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'transformer':
            print('feat-dim:{}'.format(feat_dim))
            print('max_seq_len:{}'.format(max_seq_len))
            print('num_classes=num_labels=:{}'.format(num_labels))
            print('freeze:{}'.format(config['freeze']))
            return TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                       config['num_heads'],
                                                       config['num_layers'], config['dim_feedforward'],
                                                       num_classes=num_labels,
                                                       dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                       activation=config['activation'],
                                                       norm=config['normalization_layer'], freeze=config['freeze'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding  (1024, 116)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # （1024, 1）

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

        # pe.size (1024, 1, 116)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024, seq_length=116):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        # self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        self.pe = nn.Parameter(torch.empty(max_len, 1, seq_length))  # s-transformer use
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        print('position emcodeing ------')
        print(x.shape)
        print(self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class Diffusion(nn.Module):
    def __init__(self, d_model, num_subset=3, diffusion=1, attentiondrop=0):
        super(Diffusion, self).__init__()

        self.theta = torch.nn.Parameter(torch.zeros((1, num_subset, 1, 1)))
        self.num_subset = num_subset
        self.diffusion = diffusion  # 设置diffusion的阶数
        self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        self.tan = nn.Tanh()
        self.attention0s = nn.Parameter(torch.ones(1, num_subset, d_model, d_model) / d_model,
                                        requires_grad=True)
        self.dfw2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dfw3 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dfw4 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dfw5 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.drop = nn.Dropout(attentiondrop)

        self.out_nets = nn.Sequential(
            nn.Conv2d(num_subset, 1, 1, bias=True),
            nn.BatchNorm2d(1),
        )

        self.ff_nets = nn.Sequential(
            nn.Conv2d(1, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        # inp = inp.permute(1, 0, 2)  # [batch_size, seq_length, d_model]
        x = x.permute(1, 2, 0)  # [batch_size, d_model, seq_length]

        dis = torch.cdist(x, x, p=2)  # L2正则化计算矩阵相似度

        dism = dis
        # dism.shape : [batch_size, seq_length, seq_length]
        # dism.shape : [batch_size, d_model, d_model]
        # print('dism.shape')
        # print(dism.shape)

        dism = torch.pow(dism / x.shape[2], 2)  # 计算指数 首先将 dism 中的每个数除以c*t，再平方
        # shape : 不变
        # print('dism.shape, dism = torch.pow(dism / inp.shape[2], 2)')
        # print(dism.shape)

        dsexp = torch.exp(self.theta)  # shape:[1, 3, 1, 1]
        # print('dsexp.shape, dsexp = torch.exp(self.theta)')
        # print(dsexp.shape)
        dism = torch.unsqueeze(dism, 1).repeat(1, self.num_subset, 1, 1)
        kernel_atten = torch.exp(-1 * dsexp * dism)  # 得到 kernel attention 的矩阵
        # kernel_atten.shape: [batch_size, num_subset, seq_length, seq_length]
        # kernel_atten.shape: [batch_size, num_subset, d_model, d_model]
        # print('kelnel_atten.shape')
        # print(kernel_atten.shape)

        if self.diffusion == 2:
            attention_df2 = torch.matmul(kernel_atten, kernel_atten) / kernel_atten.size()[3]  # torch.matmul tensor乘法
            kernel_atten = 0.5 * kernel_atten + attention_df2 * self.dfw2 * 0.5
        elif self.diffusion == 3:
            attention_df2 = torch.matmul(kernel_atten, kernel_atten) / kernel_atten.size()[3]
            attention_df3 = torch.matmul(attention_df2, kernel_atten) / kernel_atten.size()[3]
            kernel_atten = 0.4 * kernel_atten + attention_df2 * self.dfw2 * 0.3 + attention_df3 * self.dfw3 * 0.3
        elif self.diffusion == 4:
            attention_df2 = torch.matmul(kernel_atten, kernel_atten) / kernel_atten.size()[3]
            attention_df3 = torch.matmul(attention_df2, kernel_atten) / kernel_atten.size()[3]
            attention_df4 = torch.matmul(attention_df3, kernel_atten) / kernel_atten.size()[3]
            kernel_atten = 0.25 * kernel_atten + attention_df2 * self.dfw2 * 0.25 + attention_df3 * self.dfw3 * 0.25 + attention_df4 * self.dfw4 * 0.25
        elif self.diffusion == 5:
            attention_df2 = torch.matmul(kernel_atten, kernel_atten) / kernel_atten.size()[3]
            attention_df3 = torch.matmul(attention_df2, kernel_atten) / kernel_atten.size()[3]
            attention_df4 = torch.matmul(attention_df3, kernel_atten) / kernel_atten.size()[3]
            attention_df5 = torch.matmul(attention_df3, kernel_atten) / kernel_atten.size()[3]
            kernel_atten = 0.2 * kernel_atten + attention_df2 * self.dfw2 * 0.2 + attention_df3 * self.dfw3 * 0.2 + attention_df4 * self.dfw4 * 0.2 + attention_df5 * self.dfw5 * 0.2

        attention = self.tan(kernel_atten) * self.alphas
        attention = attention + self.attention0s.repeat(dism.shape[0], 1, 1, 1)
        # attention.shape: [batch_size, num_subset, seq_length, seq_length]
        # attention.shape: [batch_size, num_subset, d_model, d_model]

        attention = self.drop(attention)
        y = self.out_nets(attention)
        # y.shape:[batch_size, 1, seq_length, seq_length]
        # y.shape:[batch_size, 1, d_model, d_model]

        y = y.squeeze(1)
        # y.shape:[batch_size, seq_length, seq_length]
        # y.shape:[batch_size, d_model, d_model]

        y = self.relu(dis + y)
        # y.shape:[batch_size, seq_length, seq_length]
        # y.shape:[batch_size, d_model, d_model]

        # need 4-dimension input
        # y = self.ff_nets(y)
        # # y.shape:[batch_size, d_model, d_model]
        # print('after ff_net:')
        # print(y.shape)

        # y =self.relu(dis + y)
        # # y.shape:[batch_size, d_model, d_model]
        # print('afer relu and res, the last y shape:')
        # print(y.shape)

        return y


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        if max_len < d_model:
            self.max_len = d_model
        else:
            self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        print('----------------in TSTransformerEncoder -----------------')
        print('input x.shape is:')
        print(X.shape)

        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp)
        print('self.project_inp(inp)')
        print(inp.shape)
        inp = inp * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        print('inp * math.sqrt')
        print(inp.shape)

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 num_subset=3, diffusion=2,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        # if max_len < d_model:
        #     self.max_len = d_model
        # else:
        #     self.max_len = max_len
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        # ---- diffusion module setting ----------
        self.diffusion_net = nn.ModuleList()
        self.diffusion_net.append(Diffusion(d_model=d_model, num_subset=num_subset, diffusion=diffusion))
        # -----------------------------------------------

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, self.max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]

        # print('x.shape:{}'.format(X.shape))  # [32, 116, 116]
        inp1 = X.permute(1, 0, 2)
        inp1 = self.project_inp(inp1)

        inp1 = inp1 * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space

        # ------------------------------------ diffusion 机制 ----------------------------------------------
        # for i, m in enumerate(self.diffusion_net):
        #     inp = m(inp)
        # inp.shape:[batch_size, seq_length, seq_length]
        # inp.shape:[batch_size, d_model, d_model]
        # ---------------------------------------------------------------------------------------------------
        # print('inp.shape:{}'.format(inp.shape))

        # inp_T = inp.permute(1, 0, 2)  # [batch_size, seq_length, d_model]
        # inp_T = self.pos_enc(inp_T)  # add positional encoding
        # # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # output_T = self.transformer_encoder(inp_T,
        #                                     src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        # output_T = self.act(output_T)  # the output transformer encoder/decoder embeddings don't include non-linearity
        # output_T = output_T.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        # output_T = self.dropout1(output_T)

        inp_S = inp1.permute(1, 2, 0)  # [batch_size, d_model, seq_length]
        inp_S = self.pos_enc(inp_S)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output_S = self.transformer_encoder(inp_S,
                                            src_key_padding_mask=~padding_masks)  # (d_model, batch_size, seq_length)
        output_S = self.act(output_S)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output_S = output_S.permute(1, 0, 2)  # (batch_size, d_model, seq_length)
        output_S = self.dropout1(output_S)

        # Output
        output = output_S * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output


if __name__ == '__main__':
    diffusion_net = Diffusion(128)
    x = torch.randn([32, 140, 116])
    y = diffusion_net(x)
