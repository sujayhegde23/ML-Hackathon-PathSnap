# Map screenshots

Static captures of the interactive Folium maps in `maps/`. Regenerate with:

```bash
python scripts/eda.py
firefox --headless --window-size=1300,900 \
  --screenshot results/figures/maps/eda_sparse_network.png \
  "file://$PWD/maps/eda_sparse_network.html"
```

The basemap comes from Esri and is rate limited, so a capture occasionally comes back with
grey patches where tiles did not load. The road network and GPS layers are drawn by Leaflet
and always appear. A capture with missing tiles should be retaken rather than used.
