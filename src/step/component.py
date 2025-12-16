from dataclasses import dataclass

import numpy as np
from mne import pick_channels
from mne.channels import combine_channels
from pandas.api.types import is_list_like


@dataclass
class ComponentConfig:
    """The configuration for the component pipeline."""

    name: str
    tmin: float
    tmax: float
    roi: str | list[str]
    compute_se: bool = False


class ComponentPipeline:
    """The component pipeline for computing single trial amplitudes."""

    def __init__(self, config: ComponentConfig):
        self.config = config

    def run(self, epochs, bad_ixs):
        """Run the component pipeline."""

        self.epochs = epochs
        self.bad_ixs = bad_ixs

        if is_list_like(self.config.roi):
            self.roi = self.config.roi
        else:
            self.roi = [self.config.roi]

        self._add_roi_channel()

        self._get_data()

        self._compute_amplitudes()

        if self.config.compute_se:
            self.name_se = f"{self.config.name}_se"
            self._compute_standard_errors()

    def _add_roi_channel(self):
        """Add a new virtual channel by averaging over the region of interest."""

        roi_dict = {self.config.name: pick_channels(self.epochs.ch_names, self.roi)}
        epochs_roi = combine_channels(self.epochs, roi_dict)

        self.epochs.add_channels([epochs_roi], force_update_info=True)
        self.epochs.set_channel_types({self.config.name: "misc"})

    def _get_data(self):
        """Extract the time series data for the time window and region of interest."""

        self.data = (
            self.epochs.copy()
            .pick_channels([self.config.name])
            .crop(self.config.tmin, self.config.tmax)
            .get_data(units="uAU")  # Arbitrary Units, actually microvolts
        )

    def _compute_amplitudes(self):
        """Compute single-trial mean amplitudes by averaging over the time window in
        the region of interest."""

        self.amplitudes = self.data.mean(axis=(1, 2))
        self.amplitudes[self.bad_ixs] = np.nan
        self.epochs.metadata[self.config.name] = self.amplitudes

    def _compute_standard_errors(self):
        """Compute single-trial standard errors by computing the standard error
        over the time window in the region of interest."""

        self.standard_deviations = self.data.std(axis=(1, 2), ddof=1)
        self.standard_deviations[self.bad_ixs] = np.nan

        n_samples = self.data.shape[1] * self.data.shape[2]

        self.standard_errors = self.standard_deviations / np.sqrt(n_samples)
        self.epochs.metadata[self.name_se] = self.standard_errors
