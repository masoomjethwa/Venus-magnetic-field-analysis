#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: codernumber1
# coding assistant: 375219
# pip install pandas astropy

from pathlib import Path
from dataclasses import dataclass
from typing import List
import pandas as pd
from astropy.io.votable import parse
from astropy.io.votable.tree import Field


@dataclass
class MagnetometerReading:
    timestamp: str
    b_sc_x: float
    b_sc_y: float
    b_sc_z: float
    b_vso_x: float
    b_vso_y: float
    b_vso_z: float
    pos_vso_x: float
    pos_vso_y: float
    pos_vso_z: float


def parse_votable(file_path: Path) -> List[MagnetometerReading]:
    """
    Parses a VOTable XML file and returns structured magnetometer readings.
    """
    votable = parse(file_path)
    table = votable.get_first_table()

    # Map column IDs (e.g., 'col1') to semantic names (e.g., 'Time')
    col_id_to_name = {
        field.ID: field.name for field in table.fields if isinstance(field, Field)
    }

    df = table.to_table().to_pandas()
    df.columns = [col_id_to_name.get(col, col) for col in df.columns]

    readings = [
        MagnetometerReading(
            timestamp=str(row["Time"]),
            b_sc_x=float(row["B_SC_X"]),
            b_sc_y=float(row["B_SC_Y"]),
            b_sc_z=float(row["B_SC_Z"]),
            b_vso_x=float(row["B_VSO_X"]),
            b_vso_y=float(row["B_VSO_Y"]),
            b_vso_z=float(row["B_VSO_Z"]),
            pos_vso_x=float(row["POS_VSO_X"]),
            pos_vso_y=float(row["POS_VSO_Y"]),
            pos_vso_z=float(row["POS_VSO_Z"]),
        )
        for _, row in df.iterrows()
    ]

    return readings


def export_to_csv(readings: List[MagnetometerReading], output_path: Path, dry_run: bool = False) -> None:
    """
    Exports magnetometer readings to a CSV file. Skips actual write if dry_run is True.
    """
    df = pd.DataFrame([r.__dict__ for r in readings])
    if dry_run:
        print(f"   💡 [Dry Run] Would save {len(df)} rows to {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"   ✅ Saved to: {output_path.name}")


def process_all_votables(input_dir: Path, dry_run: bool = False) -> None:
    """
    Process all .xml VOTable files in the directory and convert to CSV.
    """
    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        print("❌ No XML files found in:", input_dir)
        return

    print(f"🔍 Found {len(xml_files)} VOTable files in '{input_dir}'")
    if dry_run:
        print("🚧 Dry run mode is ENABLED. No files will be written.\n")

    for i, xml_file in enumerate(xml_files, start=1):
        try:
            output_file = xml_file.with_suffix(".csv")
            if output_file.exists() and not dry_run:
                print(f"[{i}/{len(xml_files)}] ✅ Skipping (already exists): {output_file.name}")
                continue

            print(f"[{i}/{len(xml_files)}] ⏳ Processing: {xml_file.name}")
            readings = parse_votable(xml_file)
            export_to_csv(readings, output_file, dry_run=dry_run)

        except Exception as e:
            print(f"   ❌ Error processing {xml_file.name}: {e}")


def main() -> None:
    input_dir = Path(r"C:\Users\VASCSC\Desktop\venus\dataVenusMag")  # 👈 Set your folder here
    dry_run = False  # ✅ Set to False to actually write files

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"❌ Directory not found: {input_dir}")

    process_all_votables(input_dir, dry_run=dry_run)


if __name__ == "__main__":
    main()
