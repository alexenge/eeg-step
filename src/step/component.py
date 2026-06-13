from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from mne import pick_channels
from mne.channels import combine_channels
from pandas.api.types import is_list_like


@dataclass
class Component:
    """The definition of a single ERP component."""

    name: str
    tmin: float
    tmax: float
    roi: str | list[str]


class ComponentPipeline:
    """The pipeline for computing single trial amplitudes for a single ERP component."""

    def __init__(self, component: Component, compute_se: bool = False):
        self.component = component
        self.compute_se = compute_se

    def copy(self):
        """Create a copy of the ComponentPipeline instance."""

        return deepcopy(self)

    def run(self, epochs, bad_ixs):
        """Run the component pipeline."""

        self.epochs = epochs
        self.bad_ixs = bad_ixs

        self.roi = (
            self.component.roi
            if is_list_like(self.component.roi)
            else [self.component.roi]
        )

        self._add_roi_channel()

        self._get_data()

        self._compute_amplitudes()

        if self.compute_se:
            self.name_se = f"{self.component.name}_se"
            self._compute_standard_errors()

    def _add_roi_channel(self):
        """Add a new virtual channel by averaging over the region of interest."""

        roi_dict = {self.component.name: pick_channels(self.epochs.ch_names, self.roi)}
        epochs_roi = combine_channels(self.epochs, roi_dict)

        self.epochs.add_channels([epochs_roi], force_update_info=True)
        self.epochs.set_channel_types({self.component.name: "misc"}, verbose="ERROR")

    def _get_data(self):
        """Extract the time series data for the time window and region of interest."""

        self.data = (
            self.epochs.copy()
            .pick_channels([self.component.name])
            .crop(self.component.tmin, self.component.tmax)
            .get_data(units="uAU")  # Arbitrary Units, actually microvolts
        )

    def _compute_amplitudes(self):
        """Compute single-trial mean amplitudes by averaging over the time window in
        the region of interest."""

        self.amplitudes = self.data.mean(axis=(1, 2))
        self.amplitudes[self.bad_ixs] = np.nan
        self.epochs.metadata[self.component.name] = self.amplitudes

    def _compute_standard_errors(self):
        """Compute single-trial standard errors by computing the standard error
        over the time window in the region of interest."""

        self.standard_deviations = self.data.std(axis=(1, 2), ddof=1)
        self.standard_deviations[self.bad_ixs] = np.nan

        n_samples = self.data.shape[1] * self.data.shape[2]

        self.standard_errors = self.standard_deviations / np.sqrt(n_samples)
        self.epochs.metadata[self.name_se] = self.standard_errors


class ComponentsPipeline:
    """The pipeline for computing single trial amplitudes for multiple ERP
    components."""

    def __init__(
        self, components: list[Component] | Component, compute_se: bool = False
    ):
        self.components = components
        self.compute_se = compute_se

        self.components_ = components if is_list_like(components) else [components]

        self.component_pipelines = {}
        for component in self.components_:
            self.component_pipelines[component.name] = ComponentPipeline(
                component, compute_se
            )

    def copy(self):
        """Create a copy of the ComponentsPipeline instance."""

        return deepcopy(self)

    def run(self, epochs, bad_ixs):
        for component_pipeline in self.component_pipelines.values():
            component_pipeline.run(epochs, bad_ixs)
