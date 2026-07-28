"""Manual scratchpad for inspecting metadata in one sample PDF.

This file is not part of the normal automated regression suite. Running it directly
opens ``tests/Site Plan - Lot 104.pdf`` with pikepdf and prints every XMP metadata
entry, which is useful when checking what the filing workflow wrote into a real PDF.
The commented examples at the bottom preserve two earlier developer experiments.
"""

# Historical experiment: query one Maryland Open Data address and print every raw
# SDAT field. It remains disabled so a local metadata inspection never makes an
# unexpected network request.
#
# import requests
#
# OPENDATAMD_API_URL = "https://opendata.maryland.gov/resource/ed4q-f8tm.json"
# response = requests.get(
#     OPENDATAMD_API_URL,
#     params={
#         "$limit": 1,
#         "$where": "upper(mdp_street_address_mdp_field_address) = '100 JIBSAIL DR'",
#     },
# )
# prop_dict = response.json()
# for key in prop_dict[0].keys():
#     print(f"Field:{key},     Value:{prop_dict[0].get(key)}")

import pikepdf

pdf = pikepdf.Pdf.open("tests\\Site Plan - Lot 104.pdf")

with pdf.open_metadata() as meta:
    for key, value in meta.items():
        print(key, value)

# Historical path experiment: show how pathlib separates a PDF's stem and name.
#
# from pathlib import Path
# source_path = Path("").cwd()
# source_path = Path(f"{source_path}.pdf")
# print(Path(source_path.stem))
# print(source_path.name)
