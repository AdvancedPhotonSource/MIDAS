"""Consolidated grain↔peak HDF5 (gap #11).

Mirrors ``ff_MIDAS.py:generate_consolidated_hdf5``. Runs after
``process_grains`` when ``config.generate_h5`` is true. Reads:

  - Grains.csv
  - SpotMatrix.csv
  - InputAllExtraInfoFittingAll.csv
  - Radius_StartNr_*.csv
  - MergeMap.csv
  - hkls.csv
  - paramstest.txt
  - Result_StartNr_*.csv (peak-fit summary)
  - IDRings.csv, IDsHash.csv, SpotsToIndex.csv, GrainIDsKey.csv
  - per-frame ``Temp/*_PS.csv`` or consolidated ``Temp/AllPeaks_PS.bin``

… and emits ``{stem}_consolidated.h5`` plus an ``analysis/provenance``
group inside each detector's zarr.

For multi-detector runs the zarr stem of the first detector is used
for the HDF5 filename; provenance is appended to every detector's zarr.
"""
from __future__ import annotations

import os
import re
import struct
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from ._base import StageContext
from .._logging import LOG, stage_timer
from ..results import StageResult


_GRAINS_HEADER_LINES = 9

_GRAINS_COLS = [
    "GrainID", "O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
    "X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
    "eFab11", "eFab12", "eFab13", "eFab21", "eFab22", "eFab23",
    "eFab31", "eFab32", "eFab33",
    "eKen11", "eKen12", "eKen13", "eKen21", "eKen22", "eKen23",
    "eKen31", "eKen32", "eKen33",
    "RMSErrorStrain", "Confidence", "Reserved1", "Reserved2",
    "PhaseNr", "Radius", "Eul0", "Eul1", "Eul2", "Reserved3", "Reserved4",
]
_SPOT_COLS = ["GrainID", "SpotID", "Omega", "DetY", "DetZ", "OmeRaw",
              "Eta", "RingNr", "YLab", "ZLab", "Theta", "StrainError"]
_MERGE_MAP_COLS = ["MergedSpotID", "FrameNr", "PeakID"]
_PS_COLS = [
    "SpotID", "IntegratedIntensity", "Omega", "YCen", "ZCen", "IMax",
    "Radius", "Eta", "SigmaR", "SigmaEta", "NrPixels",
    "TotalNrPixelsInPeakRegion", "nPeaks", "maxY", "maxZ", "diffY", "diffZ",
    "rawIMax", "returnCode", "retVal", "BG", "SigmaGR", "SigmaLR",
    "SigmaGEta", "SigmaLEta", "MU",
]
_EXTRA_INFO_COLS = [
    "YLab", "ZLab", "Omega", "GrainRadius", "SpotID", "RingNumber",
    "Eta", "Ttheta", "OmegaIni", "YOrig", "ZOrig",
    "YOrigDetCor", "ZOrigDetCor", "OmegaOrigDetCor", "IntegratedIntensity",
]
_RADIUS_COLS = [
    "SpotID", "IntegratedIntensity", "Omega", "YCen", "ZCen", "IMax",
    "MinOme", "MaxOme", "Radius", "Theta", "Eta", "DeltaOmega", "NImgs",
    "RingNr", "GrainVolume", "GrainRadius", "PowderIntensity",
    "SigmaR", "SigmaEta", "NrPx", "NrPxTot",
]
_HKLS_COLS = ["h", "k", "l", "D-spacing", "RingNr",
              "g1", "g2", "g3", "Theta", "2Theta", "Radius"]
_RESULT_COLS = [
    "SpotID", "IntegratedIntensity", "Omega", "YCen", "ZCen", "IMax",
    "MinOme", "MaxOme", "SigmaR", "SigmaEta", "NrPx", "NrPxTot", "Radius", "Eta",
]
_IDRINGS_COLS = ["RingNumber", "OriginalID", "NewID"]
_IDSHASH_COLS = ["RingNr", "StartID", "EndID", "D-spacing"]


def _stamp_h5(h5_root) -> None:
    """Stamp the H5 root with MIDAS version metadata. Soft import: if the
    bundled ``utils/version.py`` is reachable we use it, else write a
    minimal stamp so the file is still self-describing."""
    try:
        utils = Path(os.environ.get("MIDAS_INSTALL_DIR",
                                    Path(__file__).resolve().parents[5])) / "utils"
        if str(utils) not in sys.path:
            sys.path.insert(0, str(utils))
        from version import stamp_h5 as _stamp  # type: ignore
        _stamp(h5_root)
        return
    except Exception:
        pass
    h5_root.attrs["software"] = "MIDAS / midas_ff_pipeline"


def _read_genfromtxt(path: Path, skip_header: int = 1) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            arr = np.genfromtxt(path, skip_header=skip_header)
    except Exception as e:
        LOG.warning("consolidation: failed to read %s: %s", path, e)
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _parse_paramstest(path: Path) -> dict[str, list[str] | str]:
    out: dict[str, list[str] | str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip().rstrip(";").rstrip()
        if not line or line.startswith("%"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts
        if key in out:
            existing = out[key]
            if isinstance(existing, list):
                existing.append(val)
            else:
                out[key] = [existing, val]
        else:
            out[key] = val
    return out


def _load_radius_csv(layer_dir: Path) -> tuple[Optional[np.ndarray], Optional[Path]]:
    for f in layer_dir.iterdir():
        if f.name.startswith("Radius_StartNr_") and f.suffix == ".csv":
            return _read_genfromtxt(f), f
    return None, None


def _load_result_csv(layer_dir: Path) -> Optional[np.ndarray]:
    for f in layer_dir.iterdir():
        if f.name.startswith("Result_StartNr_") and f.suffix == ".csv":
            return _read_genfromtxt(f)
    return None


def _load_consolidated_peaks(temp_dir: Path) -> tuple[dict, list]:
    ps_cache: dict[tuple[int, int], np.ndarray] = {}
    all_ps_rows: list[np.ndarray] = []
    consolidated_ps = temp_dir / "AllPeaks_PS.bin"
    if consolidated_ps.exists():
        N_PEAK_COLS = 29
        with consolidated_ps.open("rb") as bf:
            n_frames = struct.unpack("i", bf.read(4))[0]
            n_peaks_arr = struct.unpack(f"{n_frames}i", bf.read(4 * n_frames))
            offsets = struct.unpack(f"{n_frames}q", bf.read(8 * n_frames))
            remaining = bf.read()
        header_size = 4 + 4 * n_frames + 8 * n_frames
        for frame_idx in range(n_frames):
            n_pk = n_peaks_arr[frame_idx]
            if n_pk == 0:
                continue
            frame_nr = frame_idx + 1
            byte_off = offsets[frame_idx] - header_size
            frame_data = np.frombuffer(
                remaining, dtype=np.float64,
                count=n_pk * N_PEAK_COLS, offset=byte_off,
            ).reshape(n_pk, N_PEAK_COLS)
            for row in frame_data:
                peak_id = int(row[0])
                ps_cache[(frame_nr, peak_id)] = row
                all_ps_rows.append(np.concatenate([[frame_nr], row]))
        LOG.info("  loaded %d peak entries from AllPeaks_PS.bin", len(ps_cache))
        return ps_cache, all_ps_rows

    if not temp_dir.exists():
        return ps_cache, all_ps_rows
    pat = re.compile(r"_(\d{6})_PS\.csv$")
    files = sorted(p for p in temp_dir.iterdir() if p.name.endswith("_PS.csv"))
    for ps_path in files:
        m = pat.search(ps_path.name)
        if not m:
            continue
        frame_nr = int(m.group(1))
        try:
            arr = _read_genfromtxt(ps_path)
        except Exception:
            arr = None
        if arr is None:
            continue
        for row in arr:
            peak_id = int(row[0])
            ps_cache[(frame_nr, peak_id)] = row
            all_ps_rows.append(np.concatenate([[frame_nr], row]))
    LOG.info("  loaded %d peak entries from %d _PS.csv files",
             len(ps_cache), len(files))
    return ps_cache, all_ps_rows


def _zarr_stem(zarr_path: str) -> str:
    return Path(zarr_path).stem if zarr_path else "output"


def _append_provenance_to_zarr(zarr_path: Path, merge_map_data,
                               spot_data: np.ndarray) -> None:
    try:
        import zarr  # local import — soft dep
        store = zarr.ZipStore(str(zarr_path), mode="a")
        root = zarr.open(store, mode="a")
        prov = root.require_group("analysis/provenance")

        if merge_map_data is not None:
            mm_grp = prov.require_group("merge_map")
            for ci, col in enumerate(_MERGE_MAP_COLS):
                if ci < merge_map_data.shape[1]:
                    ds = col.lower()
                    if ds in mm_grp:
                        del mm_grp[ds]
                    mm_grp.create_dataset(ds, data=merge_map_data[:, ci],
                                          chunks=True, overwrite=True)

        gs = prov.require_group("grain_spots")
        for k in ("grain_ids", "spot_ids"):
            if k in gs:
                del gs[k]
        gs.create_dataset("grain_ids", data=spot_data[:, 0].astype(int),
                          chunks=True, overwrite=True)
        gs.create_dataset("spot_ids", data=spot_data[:, 1].astype(int),
                          chunks=True, overwrite=True)
        store.close()
        LOG.info("  provenance appended to %s", zarr_path)
    except Exception as e:
        LOG.warning("  failed to append provenance to %s: %s", zarr_path, e)


def run(ctx: StageContext) -> StageResult:
    started = time.time()

    if not ctx.config.generate_h5:
        return StageResult(stage_name="consolidation",
                           started_at=started, finished_at=started,
                           duration_s=0.0, metrics={"skipped": True})

    layer_dir = ctx.layer_dir
    grains_csv = layer_dir / "Grains.csv"
    spotmatrix_csv = layer_dir / "SpotMatrix.csv"
    if not grains_csv.exists() or not spotmatrix_csv.exists():
        raise FileNotFoundError(
            f"consolidation requires Grains.csv + SpotMatrix.csv in {layer_dir}"
        )

    with stage_timer("consolidation"):
        zarr_paths = [d.zarr_path for d in ctx.detectors if d.zarr_path]
        primary_zarr = zarr_paths[0] if zarr_paths else ""
        stem = _zarr_stem(primary_zarr)
        h5_path = layer_dir / f"{stem}_consolidated.h5"

        # ---- read inputs ----
        with grains_csv.open() as fp:
            _ = [fp.readline().strip() for _ in range(_GRAINS_HEADER_LINES)]
        grains_data = np.genfromtxt(grains_csv, skip_header=_GRAINS_HEADER_LINES)
        if grains_data.ndim == 1:
            grains_data = grains_data.reshape(1, -1)

        spot_data = np.genfromtxt(spotmatrix_csv, skip_header=1)
        if spot_data.ndim == 1:
            spot_data = spot_data.reshape(1, -1)

        merge_map_file = layer_dir / "MergeMap.csv"
        merge_map_data = None
        if merge_map_file.exists():
            merge_map_data = np.genfromtxt(merge_map_file, skip_header=1,
                                           delimiter="\t", dtype=int)
            if merge_map_data.ndim == 1:
                merge_map_data = merge_map_data.reshape(1, -1)

        extra_info_data = _read_genfromtxt(layer_dir / "InputAllExtraInfoFittingAll.csv")
        radius_data_arr, _radius_path = _load_radius_csv(layer_dir)
        params_dict = _parse_paramstest(layer_dir / "paramstest.txt")
        hkls_data = _read_genfromtxt(layer_dir / "hkls.csv")
        result_data = _load_result_csv(layer_dir)
        idrings_data = _read_genfromtxt(layer_dir / "IDRings.csv")
        idshash_data = _read_genfromtxt(layer_dir / "IDsHash.csv", skip_header=0)
        spots2index_data = None
        s2i = layer_dir / "SpotsToIndex.csv"
        if s2i.exists():
            tmp = np.genfromtxt(s2i, dtype=int)
            spots2index_data = tmp.reshape(1) if tmp.ndim == 0 else tmp

        grainidskey_data = None
        gik = layer_dir / "GrainIDsKey.csv"
        if gik.exists():
            raw_rows: list[list[int]] = []
            with gik.open() as fp:
                for line in fp:
                    vals = [int(x) for x in line.strip().split() if x]
                    if vals:
                        raw_rows.append(vals)
            if raw_rows:
                max_len = max(len(r) for r in raw_rows)
                grainidskey_data = np.full((len(raw_rows), max_len), -1, dtype=int)
                for ri, row in enumerate(raw_rows):
                    grainidskey_data[ri, :len(row)] = row

        merge_idx: dict[int, list[tuple[int, int]]] = defaultdict(list)
        if merge_map_data is not None:
            for row_i in range(merge_map_data.shape[0]):
                merged_id = int(merge_map_data[row_i, 0])
                merge_idx[merged_id].append((int(merge_map_data[row_i, 1]),
                                             int(merge_map_data[row_i, 2])))

        ps_cache, all_ps_rows = _load_consolidated_peaks(layer_dir / "Temp")

        radius_idx: dict[int, int] = {}
        if radius_data_arr is not None:
            for row_i in range(radius_data_arr.shape[0]):
                radius_idx[int(radius_data_arr[row_i, 0])] = row_i

        # ---- write HDF5 ----
        LOG.info("consolidation: writing %s", h5_path)
        with h5py.File(h5_path, "w") as h5:
            _stamp_h5(h5)

            pg = h5.create_group("parameters")
            for key, val in params_dict.items():
                try:
                    if isinstance(val, list):
                        float_vals: list[float] = []
                        for v in val:
                            float_vals.extend([float(x) for x in v.split()])
                        pg.create_dataset(key, data=np.array(float_vals))
                    else:
                        parts = val.split()
                        if len(parts) == 1:
                            try:
                                pg.create_dataset(key, data=float(val))
                            except ValueError:
                                pg.create_dataset(key, data=val)
                        else:
                            try:
                                pg.create_dataset(key,
                                                  data=np.array([float(x) for x in parts]))
                            except ValueError:
                                pg.create_dataset(key, data=val)
                except Exception:
                    pg.attrs[key] = str(val)

            if extra_info_data is not None:
                sg = h5.create_group("all_spots")
                sg.create_dataset("data", data=extra_info_data)
                sg.attrs["column_names"] = _EXTRA_INFO_COLS

            if radius_data_arr is not None:
                rg = h5.create_group("radius_data")
                for ci, col_name in enumerate(_RADIUS_COLS):
                    if ci < radius_data_arr.shape[1]:
                        rg.create_dataset(col_name, data=radius_data_arr[:, ci])
                rg.attrs["column_names"] = _RADIUS_COLS

            if merge_map_data is not None:
                mg = h5.create_group("merge_map")
                for ci, col_name in enumerate(_MERGE_MAP_COLS):
                    if ci < merge_map_data.shape[1]:
                        mg.create_dataset(col_name, data=merge_map_data[:, ci])
                mg.attrs["column_names"] = _MERGE_MAP_COLS

            grain_spot_indices: dict[int, list[int]] = defaultdict(list)
            for row_i in range(spot_data.shape[0]):
                grain_spot_indices[int(spot_data[row_i, 0])].append(row_i)

            radius_prop_names = ["MinOme", "MaxOme", "Theta", "DeltaOmega",
                                 "NImgs", "GrainVolume", "GrainRadius", "PowderIntensity"]
            radius_prop_indices = [
                _RADIUS_COLS.index(n) if n in _RADIUS_COLS else -1
                for n in radius_prop_names
            ]

            gg = h5.create_group("grains")
            gg.create_dataset("summary", data=grains_data)
            gg.attrs["column_names"] = _GRAINS_COLS[:min(len(_GRAINS_COLS),
                                                        grains_data.shape[1])]

            for grain_row in grains_data:
                grain_id = int(grain_row[0])
                grp = gg.create_group(f"grain_{grain_id:04d}")
                grp.create_dataset("grain_id", data=grain_id)
                grp.create_dataset("orientation", data=grain_row[1:10].reshape(3, 3))
                grp.create_dataset("position", data=grain_row[10:13])
                if grains_data.shape[1] > 43:
                    grp.create_dataset("euler_angles", data=grain_row[43:46])
                if grains_data.shape[1] > 18:
                    grp.create_dataset("lattice_params_fit", data=grain_row[13:19])
                if grains_data.shape[1] > 27:
                    grp.create_dataset("strain_fable",
                                       data=grain_row[19:28].reshape(3, 3))
                if grains_data.shape[1] > 36:
                    grp.create_dataset("strain_kenesei",
                                       data=grain_row[28:37].reshape(3, 3))
                if grains_data.shape[1] > 37:
                    grp.create_dataset("rms_strain_error", data=grain_row[37])
                if grains_data.shape[1] > 38:
                    grp.create_dataset("confidence", data=grain_row[38])
                if grains_data.shape[1] > 41:
                    grp.create_dataset("phase_nr", data=int(grain_row[41]))
                if grains_data.shape[1] > 42:
                    grp.create_dataset("radius", data=grain_row[42])

                row_indices = grain_spot_indices.get(grain_id, [])
                spots_grp = grp.create_group("spots")
                spots_grp.create_dataset("n_spots", data=len(row_indices))
                if row_indices:
                    grain_spots = spot_data[row_indices]
                    for ci, col in enumerate(_SPOT_COLS[1:], start=1):
                        if ci < grain_spots.shape[1]:
                            spots_grp.create_dataset(col.lower(),
                                                     data=grain_spots[:, ci])
                    if radius_data_arr is not None:
                        spot_ids = grain_spots[:, 1].astype(int)
                        for pi, rn in enumerate(radius_prop_names):
                            ri_ci = radius_prop_indices[pi]
                            if ri_ci < 0:
                                continue
                            vals = np.full(len(spot_ids), np.nan)
                            for si, sid in enumerate(spot_ids):
                                if sid in radius_idx:
                                    rrow = radius_data_arr[radius_idx[sid]]
                                    if ri_ci < rrow.shape[0]:
                                        vals[si] = rrow[ri_ci]
                            spots_grp.create_dataset(f"radius_{rn.lower()}",
                                                     data=vals)

            sm = h5.create_group("spot_matrix")
            sm.create_dataset("data", data=spot_data)
            sm.attrs["column_names"] = _SPOT_COLS

            if merge_idx and ps_cache:
                cp_rows = []
                for merged_id, constituents in merge_idx.items():
                    for fn, pid in constituents:
                        if (fn, pid) in ps_cache:
                            cp_rows.append(np.concatenate([[merged_id, fn, pid],
                                                           ps_cache[(fn, pid)]]))
                        else:
                            cp_rows.append(np.array([merged_id, fn, pid]
                                                    + [np.nan] * len(_PS_COLS)))
                if cp_rows:
                    cp_data = np.array(cp_rows)
                    cpg = h5.create_group("constituent_peaks")
                    cpg.create_dataset("data", data=cp_data)
                    cp_col_names = ["MergedSpotID", "FrameNr", "PeakID"] + _PS_COLS
                    cpg.attrs["column_names"] = cp_col_names[
                        :min(len(cp_col_names), cp_data.shape[1])]

            if hkls_data is not None:
                hg = h5.create_group("hkls")
                hg.create_dataset("data", data=hkls_data)
                hg.attrs["column_names"] = _HKLS_COLS[
                    :min(len(_HKLS_COLS), hkls_data.shape[1])]

            peg = h5.create_group("peaks")
            if result_data is not None:
                ps_sum = peg.create_group("summary")
                ps_sum.create_dataset("data", data=result_data)
                ps_sum.attrs["column_names"] = _RESULT_COLS[
                    :min(len(_RESULT_COLS), result_data.shape[1])]
            if all_ps_rows:
                pf_data = np.array(all_ps_rows)
                pf = peg.create_group("per_frame")
                pf.create_dataset("data", data=pf_data)
                pf_cols = ["FrameNr"] + _PS_COLS
                pf.attrs["column_names"] = pf_cols[:min(len(pf_cols),
                                                       pf_data.shape[1])]

            if idrings_data is not None:
                ir = h5.create_group("id_rings")
                ir.create_dataset("data", data=idrings_data)
                ir.attrs["column_names"] = _IDRINGS_COLS[
                    :min(len(_IDRINGS_COLS), idrings_data.shape[1])]

            if idshash_data is not None:
                ih = h5.create_group("ids_hash")
                ih.create_dataset("data", data=idshash_data)
                ih.attrs["column_names"] = _IDSHASH_COLS[
                    :min(len(_IDSHASH_COLS), idshash_data.shape[1])]

            if spots2index_data is not None:
                si = h5.create_group("spots_to_index")
                si.create_dataset("data", data=spots2index_data)

            if grainidskey_data is not None:
                gk = h5.create_group("grain_ids_key")
                gk.create_dataset("data", data=grainidskey_data)
                gk.attrs["description"] = (
                    "Each row is a grain. Values are alternating "
                    "(SpotID, LocalIndex) pairs. -1 indicates padding."
                )

            rr = h5.create_group("raw_data_ref")
            rr.create_dataset("zarr_path",
                              data=str(Path(primary_zarr).resolve())
                              if primary_zarr else "")
            if len(zarr_paths) > 1:
                rr.create_dataset(
                    "zarr_paths",
                    data=np.array([str(Path(z).resolve()) for z in zarr_paths],
                                  dtype="S"),
                )

        # ---- append provenance to each detector's zarr ----
        for z in zarr_paths:
            zp = Path(z)
            if zp.exists():
                _append_provenance_to_zarr(zp, merge_map_data, spot_data)

    finished = time.time()
    return StageResult(
        stage_name="consolidation",
        started_at=started, finished_at=finished,
        duration_s=finished - started,
        outputs={str(h5_path): ""},
        metrics={"n_grains": int(grains_data.shape[0]),
                 "n_spots": int(spot_data.shape[0])},
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    if not ctx.config.generate_h5:
        return []
    zarr_paths = [d.zarr_path for d in ctx.detectors if d.zarr_path]
    primary = zarr_paths[0] if zarr_paths else ""
    stem = _zarr_stem(primary)
    return [ctx.layer_dir / f"{stem}_consolidated.h5"]
