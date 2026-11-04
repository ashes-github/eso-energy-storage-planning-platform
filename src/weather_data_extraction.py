import calendar
from pathlib import Path

import cdsapi
import zipfile

# OUTPUT = Path("data/era5_land")
OUTPUT = Path("data/era5_land_temperature")
OUTPUT.mkdir(parents=True, exist_ok=True)


def weather_extraction(output_dir, start_year, end_year, months, variables):
    """
    Extract weather data from the ERA5-Land dataset for the specified years, months, and variables.

    Parameters:
        output_dir (Path): The directory where the extracted data will be saved.
        start_year (int): The starting year for data extraction.
        end_year (int): The ending year for data extraction.
        months (list): A list of months (1-12) to extract data for.
        variables (list): A list of variable names to extract from the dataset.
    """
    client = cdsapi.Client()

    dataset = "reanalysis-era5-land"
    times = [f"{h:02d}:00" for h in range(24)]

    for year in range(2021, 2024):
        final_month = 3 if year == 2023 else 12

        for month in range(1, final_month + 1):

            final_file = OUTPUT / f"era5_land_uk_temperature_{year}_{month:02d}.zip"
            temp_file = OUTPUT / f"era5_land_uk_temperature_{year}_{month:02d}.zip.part"

            # Skip only verified completed files
            if final_file.exists():
                try:
                    with zipfile.ZipFile(final_file) as z:
                        if z.testzip() is None:
                            print(f"Already complete: {final_file.name}")
                            continue
                except zipfile.BadZipFile:
                    print(f"Corrupt file found; redownloading: {final_file.name}")
                    final_file.unlink()

            # Remove partial download from a previous interrupted run
            if temp_file.exists():
                temp_file.unlink()

            days = [
                f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)
            ]

            request = {
                "variable": ["2m_temperature"],
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": times,
                "data_format": "netcdf",
                "download_format": "zip",
                "area": [61, -9, 49, 3],
            }

            print(f"\nSubmitting {year}-{month:02d}...")

            client.retrieve(dataset, request).download(str(temp_file))

            # Verify before marking complete
            with zipfile.ZipFile(temp_file) as z:
                bad_file = z.testzip()

                if bad_file is not None:
                    raise RuntimeError(f"Corrupt archive member: {bad_file}")

            temp_file.rename(final_file)

            print(f"Completed: {final_file.name}")


def main():
    print("Weather data extraction completed.")
    variables = [
        "2m_dewpoint_temperature",
        "2m_temperature",
        "surface_solar_radiation_downwards",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "total_precipitation",
    ]
    weather_extraction(
        output_dir=OUTPUT,
        start_year=2021,
        end_year=2023,
        months=list(range(1, 13)),
        variables=variables,
    )


if __name__ == "__main__":
    main()
