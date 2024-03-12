import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, Precision
from torch.nn import functional as F
from scripts.net import DTML
#from torchmetrics import MatthewsCorrcoef

class TrainingModule(pl.LightningModule):
    def __init__(self, beta_hyp, global_context_index, input_dim, hidden_dim, num_stocks, num_heads, num_layers, pos_weight_factor,
                 learning_rate=1e-3, lambda_reg=1e-1):
        super(TrainingModule, self).__init__()
        self.model = DTML(beta_hyp, global_context_index, input_dim, hidden_dim, num_stocks, num_heads, num_layers)
        self.pos_weight_factor = pos_weight_factor 
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        # Initialize metrics with the 'task' argument
        self.accuracy_metric = Accuracy(task='binary', average='none')
        self.precision_metric = Precision(task='binary', average='none', num_classes=2)
        #self.mcc_metric = MatthewsCorrcoef(num_classes=2)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        return self.model(x)
    
    
    def custom_loss(self, y_pred, y_true):
        
        y_pred = y_pred.view(y_pred.shape[0]*y_pred.shape[1], 1)
        y_true = y_true.view(y_true.shape[0]*y_true.shape[1], 1)
        
        # Weights for the positive class
        pos_weight = torch.tensor([self.pos_weight_factor]).to(y_pred.device)
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = bce_loss(y_pred, y_true)


        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)  # Assuming y_pred is logits with shape [batch_size, num_stocks, 2]
        loss = self.custom_loss(y_pred, y)

        # Compute and log metrics
        preds_binary = y_pred > 0.5  # Binary predictions based on threshold
        acc_per_stock = self.accuracy_metric(preds_binary, y.int())
        prec_per_stock = self.precision_metric(preds_binary, y.int())
        avg_acc = torch.mean(acc_per_stock)
        avg_prec = torch.mean(prec_per_stock)
        #mcc = self.mcc_metric(preds_binary, y.int())
        
        #self.log('train_mcc', mcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', avg_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_prec', avg_prec, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y_pred, y)
        
        preds_binary = y_pred > 0.5  # Binary predictions based on threshold
        acc_per_stock = self.accuracy_metric(preds_binary, y.int())
        prec_per_stock = self.precision_metric(preds_binary, y.int())
        
        # Average metrics across stocks
        avg_acc = torch.mean(acc_per_stock)
        avg_prec = torch.mean(prec_per_stock)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', avg_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_prec', avg_prec, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.custom_loss(y_pred, y)
        
        preds_binary = y_pred > 0.5  # Binary predictions based on threshold
        acc_per_stock = self.accuracy_metric(preds_binary, y.int())
        prec_per_stock = self.precision_metric(preds_binary, y.int())
        
        # Average metrics across stocks
        avg_acc = torch.mean(acc_per_stock)
        avg_prec = torch.mean(prec_per_stock)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', avg_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_prec', avg_prec, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer