import paddle

from paddlemetrics.wrappers.feature_share import NetworkCache


def test_network_cache():
    """Test the NetworkCache class."""

    class TestNetwork(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: paddle.Tensor) -> paddle.Tensor:
            self.calls += 1
            return x

    network = TestNetwork()
    cached_net = NetworkCache(network, max_size=2)
    x1 = paddle.randn(1, 3, 64, 64)
    cached_net(x1)
    assert network.calls == 1
    cached_net(x1)
    assert network.calls == 1
    x2 = paddle.randn(1, 3, 64, 64)
    cached_net(x2)
    assert network.calls == 2
    x3 = paddle.randn(1, 3, 64, 64)
    cached_net(x3)
    assert network.calls == 3
