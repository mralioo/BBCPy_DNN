import mlflow
import pytorch_lightning as pl

class GradientLogger(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        # Ensure we're in the training phase
        if trainer.train_loop._epoch_ended:
            return

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                mlflow.log_metric(f"grad_{name}_norm", param.grad.norm().item())