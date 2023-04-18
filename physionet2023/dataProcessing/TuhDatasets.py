import glob

import mne
import numpy as np
import torch

from physionet2023.dataProcessing.recordingDatasets import preprocess_signal


class TuhPatientDataset(torch.utils.data.Dataset):
    # PIDs with insufficient channels to reconstruct the physionet bipolar channels
    bad_pids = ["aaaaahie", "aaaaaskf", "aaaaaria", "aaaaamqq"]

    le_anodes = [
        "EEG FP1-LE",
        "EEG F7-LE",
        "EEG T3-LE",
        "EEG T5-LE",
        "EEG FP2-LE",
        "EEG F8-LE",
        "EEG T4-LE",
        "EEG T6-LE",
        "EEG FP1-LE",
        "EEG F3-LE",
        "EEG C3-LE",
        "EEG P3-LE",
        "EEG FP2-LE",
        "EEG F4-LE",
        "EEG C4-LE",
        "EEG P4-LE",
        "EEG FZ-LE",
        "EEG CZ-LE",
    ]

    le_cathodes = [
        "EEG F7-LE",
        "EEG T3-LE",
        "EEG T5-LE",
        "EEG O1-LE",
        "EEG F8-LE",
        "EEG T4-LE",
        "EEG T6-LE",
        "EEG O2-LE",
        "EEG F3-LE",
        "EEG C3-LE",
        "EEG P3-LE",
        "EEG O1-LE",
        "EEG F4-LE",
        "EEG C4-LE",
        "EEG P4-LE",
        "EEG O2-LE",
        "EEG CZ-LE",
        "EEG PZ-LE",
    ]

    ref_anodes = [
        "EEG FP1-REF",
        "EEG F7-REF",
        "EEG T3-REF",
        "EEG T5-REF",
        "EEG FP2-REF",
        "EEG F8-REF",
        "EEG T4-REF",
        "EEG T6-REF",
        "EEG FP1-REF",
        "EEG F3-REF",
        "EEG C3-REF",
        "EEG P3-REF",
        "EEG FP2-REF",
        "EEG F4-REF",
        "EEG C4-REF",
        "EEG P4-REF",
        "EEG FZ-REF",
        "EEG CZ-REF",
    ]

    ref_cathodes = [
        "EEG F7-REF",
        "EEG T3-REF",
        "EEG T5-REF",
        "EEG O1-REF",
        "EEG F8-REF",
        "EEG T4-REF",
        "EEG T6-REF",
        "EEG O2-REF",
        "EEG F3-REF",
        "EEG C3-REF",
        "EEG P3-REF",
        "EEG O1-REF",
        "EEG F4-REF",
        "EEG C4-REF",
        "EEG P4-REF",
        "EEG O2-REF",
        "EEG CZ-REF",
        "EEG PZ-REF",
    ]

    channel_names = [
        "Fp1-F7",
        "F7-T3",
        "T3-T5",
        "T5-O1",
        "Fp2-F8",
        "F8-T4",
        "T4-T6",
        "T6-O2",
        "Fp1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "Fp2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "Fz-Cz",
        "Cz-Pz",
    ]

    def __init__(self, tuh_dir="./tuh/edf", fft_coeff=10) -> None:
        super().__init__()
        self.root_dir = tuh_dir

        self.patient_paths = list()
        for subdir in glob.glob(f"{self.root_dir}/*"):
            for patient_dir in glob.glob(f"{subdir}/*"):
                if len(glob.glob(f"{patient_dir}/*/*/*.edf")) > 0:

                    if not self.__is_bad_pid(patient_dir):
                        self.patient_paths.append(patient_dir)

        if not len(self.patient_paths) > 0:
            raise ValueError(f"No data found in path: {tuh_dir}")

        self.fft_coeff = fft_coeff

    def __is_bad_pid(self, patient_dir: str):
        for p in self.bad_pids:
            if p in patient_dir:
                return True

        return False

    def __len__(self):
        return len(self.patient_paths)

    def _get_physionet_channels(self, index: int):
        # TODO: just take first edf for now but later could go for longest and / or highest-quality
        all_possible_edfs = glob.glob(f"{self.patient_paths[index]}/*/*/*.edf")

        assert len(all_possible_edfs) > 0

        edf_obj = mne.io.read_raw_edf(
            all_possible_edfs[0],
            verbose=False,
            preload=True,
            include=(
                self.le_anodes + self.le_cathodes + self.ref_anodes + self.ref_cathodes
            ),
        )

        # Re-reference to match the competition bipolar electrode setup:
        # https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html
        channels = edf_obj.ch_names

        if all(
            [ref_chan in channels for ref_chan in (self.le_anodes + self.le_cathodes)]
        ):
            raw_bipolar = mne.set_bipolar_reference(
                edf_obj,
                anode=self.le_anodes,
                cathode=self.le_cathodes,
                ch_name=self.channel_names,
                verbose=False,
            )
        elif all(
            [ref_chan in channels for ref_chan in (self.ref_anodes + self.ref_cathodes)]
        ):
            raw_bipolar = mne.set_bipolar_reference(
                edf_obj,
                anode=self.ref_anodes,
                cathode=self.ref_cathodes,
                ch_name=self.channel_names,
                verbose=False,
            )
        else:
            raise ValueError(
                f"Recording did not have epxected channels at path: {self.patient_paths[index]}"
            )

        eeg_waveform = raw_bipolar.get_data()
        assert eeg_waveform.shape[0] == len(self.channel_names)

        return eeg_waveform, edf_obj.info["sfreq"]

    def get_channels(self, index: int):
        all_possible_edfs = glob.glob(f"{self.patient_paths[index]}/*/*/*.edf")

        assert len(all_possible_edfs) > 0

        edf_obj = mne.io.read_raw_edf(
            all_possible_edfs[0], verbose=False, infer_types=True
        )

        channels = edf_obj.ch_names

        return channels

    def _get_edf_obj(self, index: int):
        # TODO: just take first edf for now but later could go for longest and / or highest-quality
        all_possible_edfs = glob.glob(f"{self.patient_paths[index]}/*/*/*.edf")

        assert len(all_possible_edfs) > 0

        edf_obj = mne.io.read_raw_edf(all_possible_edfs[0], verbose=False)
        return edf_obj

    def __getitem__(self, index: int):
        eeg_waveform, sfreq = self._get_physionet_channels(index)

        processed_signal = preprocess_signal(eeg_waveform, original_rate=int(sfreq))

        return torch.tensor(processed_signal)


class TuhPreprocessedDataset(torch.utils.data.Dataset):

    seq_len = 30000

    def __init__(self, path="cache/tuh_cache") -> None:
        super().__init__()

        self.fnames = [f for f in glob.glob(f"{path}/*.pt") if not f.startswith("_")]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index: int):
        data = torch.load(self.fnames[index])

        if data.shape[-1] > self.seq_len:
            overflow = data.shape[-1] - self.seq_len
            left_margin = int(overflow / 2)
            right_margin = data.shape[-1] - left_margin

            ret = data[:, left_margin:right_margin]

        elif data.shape[-1] < self.seq_len:
            pad = self.seq_len - data.shape[-1]
            left_pad = int(torch.floor(pad / 2))
            right_pad = int(torch.ceil(pad / 2))

            ret = torch.pad(data, (left_pad, right_pad))

        assert ret.shape[-1] == self.seq_len

        return ret.float()


if __name__ == "__main__":
    ds = TuhPreprocessedDataset()
    print(len(ds))
    a = ds[0]
