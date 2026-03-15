import lightning as L
from lightning.pytorch.callbacks import Callback


class SeedCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["global_step"] = trainer.global_step

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        global_step = checkpoint["global_step"]
        # seed = global_step
        seed = global_step + trainer.global_rank  #! this is an old one
        L.seed_everything(seed)
        print(f"Seeding rank {trainer.global_rank} with seed {seed}")


class NewSeedCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["global_step"] = trainer.global_step

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        global_step = checkpoint["global_step"]
        seed = global_step
        # seed = global_step + trainer.global_rank #! this is an old one
        L.seed_everything(seed)
        print(f"Seeding rank {trainer.global_rank} with seed {seed}")
