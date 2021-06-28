class ConfigCheckpoints:
    def __init__(self, checkpoint_every_epoch: int, start_fresh: bool, no_checkpoints: bool):
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.start_fresh = start_fresh
        self.no_checkpoints = no_checkpoints

    def has_to_save_checkpoint(self, epoch : int) -> bool:
        if self.no_checkpoints: return False
        return epoch % self.checkpoint_every_epoch == 0
