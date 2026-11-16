"""Lectura de adquisiciones SS-OCT guardadas en formato TDMS (LabVIEW).

Cada adquisición son dos archivos (más sus .tdms_index, que nptdms usa solo):
    <nombre>.tdms          -> datos crudos: grupo 'Data', canal 'Array of U16'
    <nombre>_header.tdms   -> metadatos: 'Patient Info', 'Measurement Info', 'Indexes'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tempfile

import numpy as np
from nptdms import TdmsFile

_PATIENT_KEYS = ["protocol", "date", "time", "eye", "operator",
                 "patient_id", "patient_name", "description"]


@dataclass
class OCTHeader:
    pixels: int          # muestras por A-scan (espectro)
    ascans: int          # A-scans por B-scan
    bscans: int          # B-scans declarados en el header
    range_2d: float
    range_3d: float
    sample_rate: float   # Hz
    channels: int
    patient: dict = field(default_factory=dict, repr=False)  # repr=False: no imprimir datos personales


def _paths(path: str | Path) -> tuple[Path, Path]:
    """Acepta el .tdms de datos o el _header.tdms y devuelve (datos, header)."""
    p = Path(path)
    stem = p.stem.removesuffix("_header")
    return p.with_name(stem + ".tdms"), p.with_name(stem + "_header.tdms")


def load_header(path: str | Path) -> OCTHeader:
    _, header_path = _paths(path)
    h = TdmsFile.read(header_path)
    m = h["Measurement Info"].channels()[0][:]
    patient_vals = h["Patient Info"].channels()[0][:]
    return OCTHeader(
        pixels=int(m[0]), ascans=int(m[1]), bscans=int(m[2]),
        range_2d=float(m[3]), range_3d=float(m[4]),
        sample_rate=float(m[5]), channels=int(m[6]),
        patient=dict(zip(_PATIENT_KEYS, map(str, patient_vals))),
    )


def load_raw(path: str | Path, memmap: bool = True) -> tuple[np.ndarray, OCTHeader]:
    """Devuelve (volumen, header) con volumen.shape = (n_bscans, ascans, pixels), uint16.

    El número de B-scans se calcula a partir del tamaño real de los datos,
    porque puede no coincidir con el del header (p. ej. 101 frente a 100).
    Con memmap=True los datos no se cargan enteros en RAM.
    """
    data_path, _ = _paths(path)
    hdr = load_header(path)
    tdms = TdmsFile.read(data_path, memmap_dir=tempfile.gettempdir() if memmap else None)
    raw = tdms["Data"]["Array of U16"][:]
    per_bscan = hdr.ascans * hdr.pixels
    if raw.size % per_bscan:
        raise ValueError(f"{raw.size} muestras no es múltiplo de {hdr.ascans}x{hdr.pixels}")
    n = raw.size // per_bscan
    return raw.reshape(n, hdr.ascans, hdr.pixels), hdr