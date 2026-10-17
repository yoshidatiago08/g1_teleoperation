"""The notebook window layout (needs cv2 only)."""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from dex_hand.notebook import compose


def jpeg(color, size=(480, 360)):
    image = np.full((size[1], size[0], 3), color, np.uint8)
    return cv2.imencode('.jpg', image)[1].tobytes()


def test_the_close_up_cameras_go_in_a_second_row_below_the_first_two_tiles():
    state = {'scale': 0.3, 'status': 'IMITATING', 'calibration': {'state': 'done'}, 'scene': {'cube_lifted_cm': 2.5}}
    one_row = compose(jpeg(50), jpeg(100), state)
    assert one_row.shape == (360, 960, 3)
    two_rows = compose(jpeg(50), jpeg(100), state, {'close': jpeg(200)})
    assert two_rows.shape == (720, 960, 3)
    assert two_rows[400:700, 20:460].mean() > 150                                  # the close-up fills the left tile of the second row
    assert two_rows[400:700, 520:940].mean() < 5                                   # and the rest of the row is empty
    three = compose(jpeg(50), jpeg(100), state, {'close': jpeg(200), 'top': jpeg(220)})
    assert three.shape == (720, 960, 3) and three[400:700, 520:940].mean() > 150
