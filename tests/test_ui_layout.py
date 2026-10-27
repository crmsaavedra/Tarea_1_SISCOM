import importlib
import importlib.util

import numpy as np
import pytest


def layout_module():
    assert importlib.util.find_spec("siscom.ui_layout") is not None, (
        "the shared UI layout module is missing"
    )
    return importlib.import_module("siscom.ui_layout")


def test_portrait_preview_fills_a_9_by_16_canvas_without_rotation():
    image = np.full((160, 90, 3), (12, 80, 160), dtype=np.uint8)

    preview = layout_module().fit_portrait_preview(image)

    assert preview.shape == (640, 360, 3)
    assert preview.dtype == np.uint8
    np.testing.assert_array_equal(preview[320, 180], (12, 80, 160))


def test_landscape_preview_rotates_clockwise_into_portrait():
    image = np.empty((90, 160, 3), dtype=np.uint8)
    image[:, :80] = (10, 20, 30)
    image[:, 80:] = (40, 50, 60)

    preview = layout_module().fit_portrait_preview(image)

    assert preview.shape == (640, 360, 3)
    np.testing.assert_array_equal(preview[160, 180], (10, 20, 30))
    np.testing.assert_array_equal(preview[480, 180], (40, 50, 60))


def test_nonstandard_portrait_preview_is_letterboxed_without_stretching():
    image = np.full((120, 90, 3), (120, 40, 20), dtype=np.uint8)

    preview = layout_module().fit_portrait_preview(image)

    assert preview.shape == (640, 360, 3)
    np.testing.assert_array_equal(preview[0, 180], (23, 33, 43))
    np.testing.assert_array_equal(preview[320, 180], (120, 40, 20))


@pytest.mark.parametrize("source_size", [(160, 90), (90, 160), (120, 90)])
def test_roi_display_mapping_round_trips_original_normalized_coordinates(source_size):
    layout = layout_module()
    roi_type = __import__("siscom.models", fromlist=["ROI"]).ROI
    original = roi_type(0.2, 0.15, 0.35, 0.4)

    bounds = layout.roi_to_preview_bounds(original, *source_size, width=360, height=640)
    restored = layout.preview_bounds_to_roi(bounds, *source_size, width=360, height=640)

    assert restored.left == pytest.approx(original.left, abs=0.01)
    assert restored.top == pytest.approx(original.top, abs=0.01)
    assert restored.width == pytest.approx(original.width, abs=0.01)
    assert restored.height == pytest.approx(original.height, abs=0.01)


def test_web_column_helper_is_replaced_by_desktop_panel_layout():
    layout = layout_module()

    assert not hasattr(layout, "workbench_columns")
