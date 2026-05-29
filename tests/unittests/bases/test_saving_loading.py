import paddle
import pytest

from paddlemetrics.classification import MulticlassAccuracy


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("in_device", ["cpu", "cuda"])
@pytest.mark.parametrize("out_device", ["cpu", "cuda"])
def test_saving_loading(persistent, in_device, out_device):
    """Test that saving and loading works as expected."""
    if (in_device == "cuda" or out_device == "cuda") and not paddle.cuda.is_available():
        pytest.skip("Test requires cuda, but GPU not available.")
    metric1 = MulticlassAccuracy(num_classes=5).to(in_device)
    metric1.persistent(persistent)
    metric1.update(
        paddle.randint(low=0, high=5, shape=(100,)).to(in_device),
        paddle.randint(low=0, high=5, shape=(100,)).to(in_device),
    )
    paddle.save(obj=metric1.state_dict(), path="metric.pth")
    metric2 = MulticlassAccuracy(num_classes=5).to(out_device)
    metric2.load_state_dict(paddle.load(path=str("metric.pth")))
    metric_state1 = metric1.metric_state
    metric_state2 = metric2.metric_state
    for k, v in metric_state1.items():
        v2 = metric_state2[k]
        if in_device == out_device:
            if persistent:
                assert paddle.allclose(x=v, y=v2).item()
            else:
                assert not paddle.allclose(x=v, y=v2).item()
        elif persistent:
            assert paddle.allclose(x=v, y=v2.to(v.place)).item()
        else:
            assert not paddle.allclose(x=v, y=v2.to(v.place)).item()
