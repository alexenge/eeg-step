from dataclasses import dataclass
from warnings import warn


@dataclass
class AverageConfig:
    """The configuration for the averaging pipeline."""

    name: str
    query: str


class AveragePipeline:
    """The averaging pipeline for computing by-participant average amplitudes."""

    def __init__(self, config: AverageConfig):
        self.config = config

    def run(self, epochs, bad_ixs):
        """Run the averaging pipeline."""

        self.epochs = epochs
        self.bad_ixs = bad_ixs

        self._get_good_ixs()

        self._compute_average()

    def _get_good_ixs(self):
        """Get the indices of good epochs."""

        self.good_ixs = [ix for ix in range(len(self.epochs)) if ix not in self.bad_ixs]

        if not self.good_ixs:
            warn("No good epochs found after excluding bad indices.")

    def _compute_average(self):
        """Compute the average amplitude for the specified query."""

        self.evoked = self.epochs[self.good_ixs][self.config.query].average(
            picks=["eeg", "misc"]
        )
        self.evoked.comment = self.config.name
