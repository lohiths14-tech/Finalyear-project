import pytest
from app.geospatial.core import (
    GPSCoordinate,
    FieldGeometry,
    NDVICloudProvider,
    GeospatialContextBuilder
)

def test_gps_coordinate():
    valid = GPSCoordinate(45.0, -90.0)
    assert valid.is_valid()
    
    invalid = GPSCoordinate(100.0, -90.0) # lat out of bounds
    assert not invalid.is_valid()
    
    p1 = GPSCoordinate(0.0, 0.0)
    p2 = GPSCoordinate(0.0, 1.0)
    dist = p1.distance_to(p2)
    # 1 degree at equator is approx 111km
    assert dist > 100000 

def test_field_geometry():
    corners = [
        GPSCoordinate(0.0, 0.0),
        GPSCoordinate(0.0, 0.01),
        GPSCoordinate(0.01, 0.01),
        GPSCoordinate(0.01, 0.0)
    ]
    field = FieldGeometry(boundary=corners)
    
    centroid = field.centroid
    assert centroid.latitude == pytest.approx(0.005)
    assert centroid.longitude == pytest.approx(0.005)
    
    # 0.01 degree at equator is ~1.11 km, area is ~1.23 sq km = 123 hectares
    assert field.area_hectares > 100
    assert field.area_acres > field.area_hectares

def test_ndvi_cloud_provider():
    provider = NDVICloudProvider()
    field = FieldGeometry.from_bbox(0.0, 0.0, 0.01, 0.01)
    
    with pytest.raises(Exception, match="Satellite NDVI data unavailable"):
        grid, metadata = provider.get_field_ndvi_grid(field, resolution_m=300)

def test_geospatial_context_builder():
    builder = GeospatialContextBuilder()
    context = builder.build(
        center_lat=10.0,
        center_lon=20.0,
        enable_ndvi=True
    )
    
    assert context.is_available
    assert context.field_center.latitude == 10.0
    assert not context.satellite_available
    assert context.ndvi_grid is None
