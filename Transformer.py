import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning.pytorch as pl
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class MLLM(pl.LightningModule):
    def __init__(self, vocab, dim, pad_idx, max_pos, QKV_dim, heads, num_layers,
        dropout_percentage, learning_rate, wd, ls, pct_start,act, ffn_internal):

        super().__init__()
        self.QKV_dim = QKV_dim
        self.head = heads
        self.pad_idx = pad_idx
        self.num_layers = num_layers
        self.dropout_percentage = dropout_percentage
        self.learning_rate = learning_rate
        self.wd = wd
        self.ls = ls
        self.pct_start = pct_start
        self.act = nn.ReLU() if act == "relu" else nn.GELU()
        self.ffn_internal = ffn_internal

        self.embedtoken = nn.Embedding(
            num_embeddings=vocab, embedding_dim=dim, padding_idx=pad_idx
        )
        self.embedpos = nn.Embedding(
            num_embeddings=max_pos, embedding_dim=dim
        )

        self.dropout = nn.Dropout(self.dropout_percentage)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "Q": nn.Linear(dim, QKV_dim * heads),
                "K": nn.Linear(dim, QKV_dim * heads),
                "V": nn.Linear(dim, QKV_dim * heads),
                "headout": nn.Linear(QKV_dim * heads, dim),
                "norm1": nn.LayerNorm(dim),
                "norm2": nn.LayerNorm(dim),
                "FFN": nn.Sequential(
                    nn.Linear(dim, dim * self.ffn_internal),
                    self.act,
                    nn.Linear(dim * self.ffn_internal, dim)
                ),
            }) for _ in range(num_layers)
        ])

        self.model_head = nn.Linear(dim, vocab, bias=False)
        self.model_head.weight = self.embedtoken.weight

        self.loss = nn.CrossEntropyLoss(
            ignore_index=pad_idx, label_smoothing=self.ls
        )

    def setup(self, stage=None):
        self.model_head.weight = self.embedtoken.weight

    def forward(self, input):
        tok_embed = self.embedtoken(input)

        B, T = input.size()
        positions = torch.arange(T, device=input.device).unsqueeze(0).expand(B, T)
        pos_embed = self.embedpos(positions)

        causal = torch.tril(torch.ones(T, T, device=input.device, dtype=torch.bool))
        pad_mask = (input != self.pad_idx).unsqueeze(1).unsqueeze(2)
        mask = pad_mask & causal

        contextual_embed = tok_embed + pos_embed

        for layers in self.layers:
            norm_1 = layers["norm1"](contextual_embed)

            Q = layers["Q"](norm_1)
            K = layers["K"](norm_1)
            V = layers["V"](norm_1)

            Q = Q.view(B, T, self.head, self.QKV_dim).transpose(1, 2)
            K = K.view(B, T, self.head, self.QKV_dim).transpose(1, 2)
            V = V.view(B, T, self.head, self.QKV_dim).transpose(1, 2)

            score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.QKV_dim)
            score = score.masked_fill(~mask, float("-inf"))

            final_score = F.softmax(score, dim=-1)
            final_score = torch.matmul(final_score, V)
            final_score = self.dropout(final_score)

            attention = final_score.transpose(1, 2).contiguous().view(B, T, self.head * self.QKV_dim)

            out = layers["headout"](attention)
            out = self.dropout(out)
            x = contextual_embed + out

            norm_2 = layers["norm2"](x)
            ffn = layers["FFN"](norm_2)
            ffn = self.dropout(ffn)

            x = x + ffn
            contextual_embed = x

        x = self.model_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x_train, y_train = batch
        preds = self(x_train)
        loss = self.loss(preds.permute(0, 2, 1), y_train)
        self.log("Train_Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val = batch
        preds = self(x_val)
        loss = self.loss(preds.permute(0, 2, 1), y_val)
        self.log("Val_Loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=self.wd,
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=self.pct_start,
                div_factor=25,
                final_div_factor=10
            ),
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]
    
    
    def on_train_batch_end(self, *_):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, logger=False)
