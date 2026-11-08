"""CWRU 12 kHz drive-end catalogue: which file is which physical bearing.

Everything the split logic needs to know about a recording lives here, so the
protocol can be reviewed without opening a single .mat file:

* `file_id`   — the number in the official file name (`97.mat`, `105.mat`, ...).
* `bearing_id`— the *physical* bearing: fault type + fault diameter (+ outer-race
                position). Every faulty bearing was recorded once per motor load,
                so a bearing_id groups 4 recordings. The healthy set is ONE
                bearing (`N`) recorded at 4 loads — there is no second healthy
                bearing anywhere in CWRU, which is why the healthy class can only
                be held out by load, never by bearing.
* `load_hp`   — motor load 0..3 HP (nominal shaft speed 1797/1772/1750/1730 rpm).
* `fs_hz`     — sampling rate of the file as published. Faulty recordings are
                12 kHz; the healthy baseline was recorded at 48 kHz and is
                decimated x4 by the loader (recorded in the run metadata).
* `quirks`    — known oddities of the published file, kept as data so they are
                logged, not silently absorbed.

Sources: CWRU Bearing Data Center file tables; Smith & Randall, MSSP 64-65 (2015)
(diagnosability of individual records); srigas/CWRU_Bearing_NumPy (duplicated /
foreign variables inside .mat files); Hendriks, Dumond & Knox, MSSP 169 (2022)
(recording-level leakage in 40 of 41 CWRU papers).
"""

from __future__ import annotations

from dataclasses import dataclass, field

LOAD_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
FAULT_FS_HZ = 12_000
NORMAL_FS_HZ = 48_000


@dataclass(frozen=True)
class Recording:
    file_id: int
    fault: str                 # "N" | "IR" | "B" | "OR"
    size_in: float | None      # fault diameter, inches (None for healthy)
    or_pos: int | None         # outer-race position, o'clock (6 = centred / load zone)
    load_hp: int
    fs_hz: int
    quirks: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rpm(self) -> int:
        return LOAD_RPM[self.load_hp]

    @property
    def is_anomaly(self) -> bool:
        return self.fault != "N"

    @property
    def bearing_id(self) -> str:
        """'N', 'IR007', 'B021', 'OR007@6' — one id per physical bearing."""
        if self.fault == "N":
            return "N"
        mils = int(round(self.size_in * 1000))
        base = f"{self.fault}{mils:03d}"
        return f"{base}@{self.or_pos}" if self.fault == "OR" else base

    @property
    def mat_name(self) -> str:
        return f"{self.file_id}.mat"

    @property
    def de_key(self) -> str:
        """Variable name of the drive-end accelerometer inside the .mat file."""
        return f"X{self.file_id:03d}_DE_time"


def _group(fault: str, size_in: float | None, or_pos: int | None, file_ids: tuple[int, int, int, int],
           fs_hz: int = FAULT_FS_HZ, quirks: tuple[str, ...] = ()) -> list[Recording]:
    return [
        Recording(file_id=fid, fault=fault, size_in=size_in, or_pos=or_pos, load_hp=load, fs_hz=fs_hz, quirks=quirks)
        for load, fid in enumerate(file_ids)
    ]


# Official CWRU file numbers, 12 kHz drive-end fault data + the 48 kHz baseline.
# Order inside each tuple = load 0, 1, 2, 3 HP.
RECORDINGS: list[Recording] = [
    # --- healthy baseline: ONE bearing, 48 kHz ------------------------------
    *_group("N", None, None, (97, 98, 99, 100), fs_hz=NORMAL_FS_HZ,
            quirks=("recorded_at_48kHz_decimated_x4",)),
    # --- 0.007" -------------------------------------------------------------
    *_group("IR", 0.007, None, (105, 106, 107, 108)),
    *_group("B", 0.007, None, (118, 119, 120, 121)),
    *_group("OR", 0.007, 6, (130, 131, 132, 133)),
    *_group("OR", 0.007, 3, (144, 145, 146, 147)),
    *_group("OR", 0.007, 12, (156, 158, 159, 160)),
    # --- 0.014" (validation bearings in protocol lolo_v1) --------------------
    *_group("IR", 0.014, None, (169, 170, 171, 172)),
    *_group("B", 0.014, None, (185, 186, 187, 188)),
    *_group("OR", 0.014, 6, (197, 198, 199, 200)),
    # --- 0.021" -------------------------------------------------------------
    *_group("IR", 0.021, None, (209, 210, 211, 212)),
    *_group("B", 0.021, None, (222, 223, 224, 225)),
    *_group("OR", 0.021, 6, (234, 235, 236, 237)),
    *_group("OR", 0.021, 3, (246, 247, 248, 249)),
    *_group("OR", 0.021, 12, (258, 259, 260, 261)),
    # --- 0.028" (12k DE only; no outer-race bearing at this size) -------------
    *_group("IR", 0.028, None, (3001, 3002, 3003, 3004),
            quirks=("de_variable_not_named_after_file_id",)),
    *_group("B", 0.028, None, (3005, 3006, 3007, 3008),
            quirks=("de_variable_not_named_after_file_id",)),
]

# Known per-file oddities of the published .mat files (logged by the loader).
FILE_QUIRKS: dict[int, tuple[str, ...]] = {
    97: ("baseline_0hp_is_half_length",),          # ~243 k samples at 48 kHz vs ~485 k for 1-3 HP
    99: ("contains_foreign_variable_X098",),       # X098_DE_time also present inside 99.mat
}

BY_ID: dict[int, Recording] = {r.file_id: r for r in RECORDINGS}
assert len(BY_ID) == len(RECORDINGS), "duplicate file_id in catalogue"

BEARING_IDS: list[str] = sorted({r.bearing_id for r in RECORDINGS})
FAULT_BEARING_IDS: list[str] = [b for b in BEARING_IDS if b != "N"]


def recordings_for(bearing_ids: set[str] | list[str] | None = None,
                   loads: set[int] | list[int] | None = None,
                   sizes_in: set[float] | list[float] | None = None) -> list[Recording]:
    """Filter the catalogue. `None` means no restriction on that axis."""
    out = []
    for r in RECORDINGS:
        if bearing_ids is not None and r.bearing_id not in set(bearing_ids):
            continue
        if loads is not None and r.load_hp not in set(loads):
            continue
        if sizes_in is not None and r.is_anomaly and r.size_in not in set(sizes_in):
            continue
        out.append(r)
    return out


def quirks_of(recording: Recording) -> tuple[str, ...]:
    return tuple(recording.quirks) + FILE_QUIRKS.get(recording.file_id, ())
