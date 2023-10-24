import csv
from pathlib import Path

import pyproj

pyproj.set_use_global_context(active=True)  # Speed up instantiation of lots of PyProj objects
pyproj.network.set_network_enabled(True)  # Load datum shift grids from PROJ CDN (https://proj.org/usage/network.html)

output: list[dict] = []
for c in pyproj.database.get_codes('EPSG', pyproj.enums.PJType.PROJECTED_CRS, allow_deprecated=True):
    crs = pyproj.CRS(f'EPSG:{c}')
    if not any(k in crs.name.lower() for k in ['maryland']):
        continue
    output.append(
        {
            'code': f'EPSG:{c}',
            'crs_name': crs.name,
            'operation_name': crs.coordinate_operation.name,
            'method_name': crs.coordinate_operation.method_name,
            **{p.name: p.value for p in crs.coordinate_operation.params},
        }
    )

with Path('crs.tsv').open('w') as fp:
    writer = csv.DictWriter(fp, fieldnames=output[0].keys(), delimiter='\t')
    writer.writeheader()
    for row in output:
        for field_name in writer.fieldnames:
            if field_name not in row:
                row[field_name] = ''
        try:
            writer.writerow(row)
        except Exception as e:
            breakpoint()
            raise e
