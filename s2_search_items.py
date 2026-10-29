import json
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry

URL = "https://stac.dataspace.copernicus.eu/v1"

retry = Retry(
    total=6,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=None,
    respect_retry_after_header=True,
)
cat = pystac_client.Client.open(URL, stac_io=StacApiIO(max_retries=retry))
cat.add_conforms_to("ITEM_SEARCH")

params = {
    "max_items": 500,
    "collections": ["sentinel-2-l2a"],
    "datetime": "2017-08-01/2017-10-31",
    "bbox": [-114.4, 48.9, -113.7, 49.2],
    "sortby": [{"field": "properties.datetime", "direction": "asc"}],
}

items = list(cat.search(**params).items_as_dicts())
print(f"Found {len(items)} items")

with open("kenow_items.json", "w") as f:
    json.dump(items, f)