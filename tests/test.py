import requests

OPENDATAMD_API_URL = "https://opendata.maryland.gov/resource/ed4q-f8tm.json"

response = requests.get(
    OPENDATAMD_API_URL,
    params={
        "$limit": 1,
        "$where": "upper(mdp_street_address_mdp_field_address) = '1016 UPPER PINDELL RD'"
    }
)

prop_dict = response.json()
for key in prop_dict[0].keys():  # noqa: SIM118
    print(f"Field:{key},     Value:{prop_dict[0].get(key)}")

"""import pikepdf

pdf = pikepdf.Pdf.open("tests\\Site Plan - Lot 102.pdf")

with pdf.open_metadata() as meta:
    for key, value in meta.items():
        print(key, value)
"""