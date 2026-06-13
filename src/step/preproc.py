from copy import deepcopy
from os import PathLike
from pathlib import Path
from warnings import warn

import pandas as pd
from mne import set_bipolar_reference
from mne.channels import make_standard_montage, read_custom_montage
from mne.channels.montage import DigMontage, get_builtin_montages
from mne.io import BaseRaw
from mne.preprocessing import ICA


class PreprocPipeline:
    """The preprocessing pipeline for cleaning the raw EEG data."""

    def __init__(
        self,
        downsample_sfreq: float = None,
        heog_channels: list[str] | str = "auto",
        veog_channels: list[str] | str = "auto",
        montage: str | PathLike = "easycap-M1",
        bad_channels: list[str] | str = None,
        ref_channels: list[str] | str = "average",
        ica_method: str = "fastica",
        ica_n_components: int | float = None,
        ica_eog_channels: list[str] | str = "auto",
        highpass_freq: float = 0.1,
        lowpass_freq: float = 40.0,
    ):
        self.downsample_sfreq = downsample_sfreq
        self.heog_channels = heog_channels
        self.veog_channels = veog_channels
        self.montage = montage
        self.bad_channels = bad_channels
        self.ref_channels = ref_channels
        self.ica_method = ica_method
        self.ica_n_components = ica_n_components
        self.ica_eog_channels = ica_eog_channels
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq

    def copy(self):
        """Create a copy of the PreprocPipeline instance."""

        return deepcopy(self)

    def run(self, raw, besa=None):
        """Run the preprocessing pipeline."""

        assert isinstance(raw, BaseRaw), (
            "`raw` must be an instance of the `mne.io.BaseRaw` class"
        )

        if besa is not None and self.ica_method is not None:
            warn(
                "Disabling ICA correction since BESA/MSEC correction is enabled. "
                "You can disable this warning by setting `ica_method=None`."
            )
            self.ica_method_ = None
        else:
            self.ica_method_ = self.ica_method

        self.raw = raw.copy()

        if self.downsample_sfreq is not None:
            self._resample()

        if self.heog_channels is not None:
            self._add_heog()

        if self.veog_channels is not None:
            self._add_veog()

        self._adjust_channel_types()

        self._apply_montage()

        if self.bad_channels not in [None, "auto", []]:
            self._interpolate_bad_channels()

        self._set_eeg_reference()

        if besa is not None:
            assert isinstance(besa, pd.DataFrame), "`besa` must be a `pandas.DataFrame`"
            self.besa = besa
            self._correct_besa()

        if self.ica_method is not None:
            if self.ica_eog_channels == "auto":
                self.ica_eog_channels_ = ["HEOG", "VEOG"]
            else:
                self.ica_eog_channels_ = self.ica_eog_channels
            self._correct_ica()

        if self.lowpass_freq is not None or self.highpass_freq is not None:
            self._filter()

    def _resample(self):
        """Resample the raw data to the specified sampling frequency."""

        self.raw.resample(self.downsample_sfreq)

    def _add_heog(self):
        """Add a bipolar HEOG channel to the raw data."""

        if self.heog_channels == "auto":
            self.heog_channels_ = AUTO_HEOG_CHANNELS
        else:
            self.heog_channels_ = self.heog_channels

        self._add_eog(self.heog_channels_, name="HEOG")

    def _add_veog(self):
        """Add a bipolar VEOG channel to the raw data."""

        if self.veog_channels == "auto":
            self.veog_channels_ = AUTO_VEOG_CHANNELS
        else:
            self.veog_channels_ = self.veog_channels

        self._add_eog(self.veog_channels_, name="VEOG")

    def _add_eog(self, channels, name):
        """Add a bipolar EOG channel to the raw data."""

        channels = [ch for ch in channels if ch in self.raw.ch_names]

        assert len(channels) == 2, (
            "Invalid channel selection for computing "
            f'bipolar channel "{name}". Please '
            "provide exactly two channels that are "
            "present in the EEG data"
        )

        anode = channels[0]
        cathode = channels[1]

        self.raw = set_bipolar_reference(
            self.raw, anode, cathode, name, drop_refs=False, verbose=False
        )
        self.raw.set_channel_types({name: "eog"})

    def _apply_montage(self):
        """Apply a standard or custom montage to the raw data."""

        if not isinstance(self.montage, DigMontage):
            if Path(self.montage).exists():
                self.montage_ = read_custom_montage(self.montage)

            elif self.montage in get_builtin_montages():
                self.montage_ = make_standard_montage(self.montage)

            else:
                raise ValueError(
                    "`montage` must be a valid file path, the "
                    "name of a valid standard montage, or an MNE "
                    "`DigMontage` object"
                )

        else:
            self.montage_ = self.montage

        self.raw.set_montage(self.montage_, match_case=False, on_missing="warn")

    def _adjust_channel_types(self):
        """Adjust the channel types of the raw data."""

        self._adjust_channel_type(DEFAULT_EOG_CHANNELS, type="eog")
        self._adjust_channel_type(DEFAULT_MISC_CHANNELS, type="misc")

    def _adjust_channel_type(self, channels, type):
        """Adjust the channel type of the specified channels."""

        for ch_name in channels:
            if ch_name in self.raw.ch_names:
                self.raw.set_channel_types({ch_name: type}, on_unit_change="ignore")

    def _interpolate_bad_channels(self):
        """Interpolate bad channels in the raw data."""

        self.raw.info["bads"] += self.bad_channels
        self.raw.interpolate_bads()

    def _set_eeg_reference(self):
        """Set the EEG reference to the specified channels."""

        self.raw.set_eeg_reference(self.ref_channels)

    def _correct_besa(self):
        """Correct the raw data using the BESA/MSEC procedure."""

        # Subset BESA matrix to only channels that are in the data
        besa = self.besa.copy()
        eeg_channels = self.raw.copy().pick_types(eeg=True).ch_names
        eeg_channels_upper = pd.Series(eeg_channels).str.upper().values
        besa.index = besa.index.str.upper()
        besa.columns = besa.columns.str.upper()
        row_channels = [ch for ch in besa.index if ch in eeg_channels_upper]
        col_channels = [ch for ch in besa.columns if ch in eeg_channels_upper]
        besa = besa.reindex(index=row_channels, columns=col_channels)

        eeg_data, _ = self.raw[eeg_channels]
        eeg_data = besa.values.dot(eeg_data)
        self.raw[eeg_channels] = eeg_data

    def _correct_ica(self, random_seed=1234):
        """Correct the raw data using independent component analysis (ICA)."""

        if (
            self.ica_n_components is not None
            and self.ica_n_components >= 1.0
            and not isinstance(self.ica_n_components, int)
        ):
            warn(
                "Converting `ica_n_components` to integer: "
                f"{self.ica_n_components} -> {int(self.ica_n_components)}"
            )
            self.ica_n_components = int(self.ica_n_components)

        raw_ica = self.raw.copy()
        raw_ica.load_data().filter(l_freq=1, h_freq=None, verbose=False)
        ica = ICA(
            self.ica_n_components,
            random_state=random_seed,
            method=self.ica_method,
            max_iter="auto",
        )
        ica.fit(raw_ica)

        eog_indices, _ = ica.find_bads_eog(
            self.raw, ch_name=self.ica_eog_channels_, verbose=False
        )
        ica.exclude = eog_indices

        self.ica = ica
        self.raw = ica.apply(self.raw)

    def _filter(self):
        """Filter the raw data using a bandpass filter."""

        self.raw.filter(self.highpass_freq, self.lowpass_freq, n_jobs=1, picks="eeg")


AUTO_HEOG_CHANNELS = ["F9", "F10", "Afp9", "Afp10", "HEOG_left", "HEOG_right"]
AUTO_VEOG_CHANNELS = ["Fp1", "FP1", "Auge_u", "IO1", "VEOG_lower", "VEOG_upper"]

DEFAULT_EOG_CHANNELS = [
    "HEOG",
    "VEOG",
    "IO1",
    "IO2",
    "Afp9",
    "Afp10",
    "Auge_u",
    "VEOG_upper",
    "VEOG_lower",
    "HEOG_left",
    "HEOG_right",
]
DEFAULT_MISC_CHANNELS = ["A1", "A2", "M1", "M2", "audio", "sound", "pulse"]
