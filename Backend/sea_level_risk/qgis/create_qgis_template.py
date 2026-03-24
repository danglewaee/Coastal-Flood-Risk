from pathlib import Path


def _combined_extent(layers):
    extent = None
    for layer in layers:
        if layer is None or not layer.isValid():
            continue
        layer_extent = layer.extent()
        if extent is None:
            extent = layer_extent
        else:
            extent.combineExtentWith(layer_extent)
    return extent


def _remove_existing_layout(project, layout_name: str):
    manager = project.layoutManager()
    existing = manager.layoutByName(layout_name)
    if existing is not None:
        manager.removeLayout(existing)


def _create_print_layout(project, layers, title_text: str):
    from qgis.PyQt.QtGui import QFont
    from qgis.core import (
        QgsLayoutItemLabel,
        QgsLayoutItemLegend,
        QgsLayoutItemMap,
        QgsLayoutItemPage,
        QgsLayoutItemScaleBar,
        QgsLayoutPoint,
        QgsLayoutSize,
        QgsPrintLayout,
        QgsTextFormat,
        QgsUnitTypes,
    )

    layout_name = "Anti Flood Layout"
    _remove_existing_layout(project, layout_name)

    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(layout_name)
    project.layoutManager().addLayout(layout)

    page = layout.pageCollection().pages()[0]
    page.setPageSize("A4", QgsLayoutItemPage.Orientation.Landscape)

    title = QgsLayoutItemLabel(layout)
    title.setText(title_text)
    title_format = QgsTextFormat()
    title_format.setFont(QFont("Arial", 18))
    title.setTextFormat(title_format)
    title.adjustSizeToText()
    title.attemptMove(QgsLayoutPoint(12, 8, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(title)

    map_item = QgsLayoutItemMap(layout)
    map_item.attemptMove(QgsLayoutPoint(10, 22, QgsUnitTypes.LayoutMillimeters))
    map_item.attemptResize(QgsLayoutSize(210, 140, QgsUnitTypes.LayoutMillimeters))
    map_item.setLayers(layers)
    extent = _combined_extent(layers)
    if extent is not None:
        map_item.setExtent(extent)
    layout.addLayoutItem(map_item)

    legend = QgsLayoutItemLegend(layout)
    legend.setTitle("Layers")
    legend.setLinkedMap(map_item)
    legend.setAutoUpdateModel(True)
    legend.attemptMove(QgsLayoutPoint(226, 24, QgsUnitTypes.LayoutMillimeters))
    legend.attemptResize(QgsLayoutSize(48, 70, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(legend)

    scale_bar = QgsLayoutItemScaleBar(layout)
    scale_bar.setStyle("Single Box")
    scale_bar.setLinkedMap(map_item)
    scale_bar.setNumberOfSegments(4)
    scale_bar.setUnits(QgsUnitTypes.DistanceKilometers)
    scale_bar.setUnitLabel("km")
    scale_bar.update()
    scale_bar.attemptMove(QgsLayoutPoint(12, 166, QgsUnitTypes.LayoutMillimeters))
    scale_bar.attemptResize(QgsLayoutSize(70, 8, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(scale_bar)

    notes = QgsLayoutItemLabel(layout)
    notes.setText("Source: NOAA water level + local DEM threshold (coast-connected)")
    note_format = QgsTextFormat()
    note_format.setFont(QFont("Arial", 8))
    notes.setTextFormat(note_format)
    notes.adjustSizeToText()
    notes.attemptMove(QgsLayoutPoint(120, 167, QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(notes)

    return layout


def create_project(package_dir: str, out_project: str | None = None, include_layout: bool = True):
    try:
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsProject,
            QgsRasterLayer,
            QgsVectorLayer,
        )
    except ImportError as exc:
        raise RuntimeError("Run this script inside QGIS Python Console (PyQGIS required).") from exc

    pkg = Path(package_dir).resolve()
    if not pkg.exists():
        raise FileNotFoundError(str(pkg))

    layers_dir = pkg / "layers"
    styles_dir = pkg / "styles"

    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))

    dem_candidates = list(layers_dir.glob("*.tif"))
    if not dem_candidates:
        raise FileNotFoundError("No DEM .tif found in package layers folder")

    dem_path = dem_candidates[0]
    dem_layer = QgsRasterLayer(str(dem_path), "dem")
    if not dem_layer.isValid():
        raise RuntimeError(f"Invalid DEM layer: {dem_path}")
    project.addMapLayer(dem_layer, True)

    layer_specs = [
        ("flood_plus_20cm", "flood_plus_20cm.geojson", "flood_plus_20cm.qml"),
        ("flood_plus_50cm", "flood_plus_50cm.geojson", "flood_plus_50cm.qml"),
        ("flood_plus_100cm", "flood_plus_100cm.geojson", "flood_plus_100cm.qml"),
        ("hotspots", "hotspots.geojson", "hotspots.qml"),
    ]

    created = [dem_layer]

    for name, file_name, style_name in layer_specs:
        path = layers_dir / file_name
        if not path.exists():
            continue

        layer = QgsVectorLayer(str(path), name, "ogr")
        if not layer.isValid():
            continue

        style = styles_dir / style_name
        if style.exists():
            layer.loadNamedStyle(str(style))
            layer.triggerRepaint()

        project.addMapLayer(layer, True)
        created.append(layer)

    root = project.layerTreeRoot()
    root.setCustomLayerOrderByIds([layer.id() for layer in created])
    root.setHasCustomLayerOrder(True)

    if include_layout:
        city_name = pkg.name.split("_")[0].replace("-", " ").title()
        _create_print_layout(project, created, f"Anti-Flood Priority Map - {city_name}")

    if out_project is None:
        out_project = str(pkg / "anti_flood_template.qgz")

    out_path = Path(out_project)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = project.write(str(out_path))
    if not ok:
        raise RuntimeError(f"Failed to write QGIS project: {out_path}")

    return str(out_path)


# Example usage in QGIS Python Console:
# from Backend.sea_level_risk.qgis.create_qgis_template import create_project
# create_project(r"<repo_root>\\Backend\\sea_level_risk\\outputs\\qgis_packages\\honolulu_20260314_014207")
