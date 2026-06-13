from copy import deepcopy
from dataclasses import dataclass
from warnings import warn

from pandas.api.types import is_list_like


@dataclass
class Average:
    """The definition of a single by-participant average."""

    name: str
    query: str


class AveragePipeline:
    """The pipeline for computing a single by-participant average."""

    def __init__(self, average: Average):
        self.average = average

    def copy(self):
        """Create a copy of the AveragePipeline instance."""

        return deepcopy(self)

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

        self.evoked = self.epochs[self.good_ixs][self.average.query].average(
            picks=["eeg", "misc"]
        )
        self.evoked.comment = self.average.name


class AveragesPipeline:
    """The pipeline for computing multiple by-participant averages."""

    def __init__(self, averages: list[Average] | Average):
        self.averages = averages

        self.averages_ = averages if is_list_like(averages) else [averages]

        self.average_pipelines = {}
        for average in self.averages_:
            self.average_pipelines[average.name] = AveragePipeline(average)

    def copy(self):
        """Create a copy of the AveragesPipeline instance."""

        return deepcopy(self)

    def run(self, epochs, bad_ixs):
        for average_pipeline in self.average_pipelines.values():
            average_pipeline.run(epochs, bad_ixs)
