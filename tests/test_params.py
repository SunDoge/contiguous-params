import copy

import pytest
import torch
import torch.distributed as dist
from contiguous_params.params import ContiguousParams
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_contiguous_params(device: str):
    if device == 'cuda' and not torch.cuda.is_available():
        print("No GPU available, skipping GPU test.")
        return

    """Verify that the parameters are the same after a few updates."""
    x = torch.randn(1, 8).to(device)

    model_ref = nn.Sequential(*[nn.Linear(8, 8) for i in range(10)])
    model_ref = model_ref.to(device)
    optimizer = torch.optim.SGD(model_ref.parameters(), lr=1e-3)

    model_c: nn.Module = copy.deepcopy(model_ref)
    parameters_c = ContiguousParams(model_c.parameters())
    optimizer_c = torch.optim.SGD(parameters_c.contiguous(), lr=1e-3)

    for model, optimizer in zip([model_ref, model_c], [optimizer, optimizer_c]):
        for step in range(5):
            loss: Tensor = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Verify that the model/optimizer did not modify the data or grad handle.
    parameters_c.assert_buffer_is_valid()

    # Verify that both models applied the same parameter updates.
    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        assert torch.allclose(p1.data, p2.data, atol=1e-06)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_buffer_invalidation_detection(device):
    """Verify that we recognize an invalidated buffer."""
    if device == "cuda" and not torch.cuda.is_available():
        print("No GPU available, skipping GPU test.")
        return
    model = nn.Linear(8, 8)
    parameters = ContiguousParams(model.parameters())
    parameters.assert_buffer_is_valid()
    # Invalidate the buffer.
    model.weight.data = model.weight + 4
    assert not parameters.buffer_is_valid()
    with pytest.raises(ValueError):
        parameters.assert_buffer_is_valid()


def test_distributed_data_parallel():
    """Verify that this works in the distributed data paralllel setting."""
    # Create 20 samples with 10 features, label one out of 5 classes.
    # data_X = torch.as_tensor(np.random.randn(20, 10), dtype=torch.float32)
    # data_y = torch.as_tensor(np.random.choice(5, (20)), dtype=torch.int64)

    port = 12345
    dist.init_process_group(
        'GLOO',
        init_method='tcp://127.0.0.1:{}'.format(port),
        world_size=1,
        rank=0
    )

    data_X = torch.rand(20, 10)
    data_y = torch.randint(0, 5, (20,))

    dataset = TensorDataset(data_X, data_y)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def build_model():
        m = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        return DistributedDataParallel(m)

    model_ref = build_model()

    initial_configuration = copy.deepcopy(model_ref.state_dict())

    model_c = build_model()
    model_c.load_state_dict(initial_configuration)

    contiguous_params = ContiguousParams(model_c.parameters())

    optimizer_ref = torch.optim.SGD(model_ref.parameters(), lr=1e-3)
    optimizer_c = torch.optim.SGD(contiguous_params.contiguous(), lr=1e-3)

    for i, (model, optimizer) in enumerate([(model_ref, optimizer_ref), (model_c, optimizer_c)]):
        # Choose different ports to prevent
        # RuntimeError("Address already in use.").
        # os.environ['MASTER_PORT'] = str(port + i)
        # trainer = pytorch_lightning.Trainer(
        #     distributed_backend="ddp", max_epochs=1, gpus=[0])
        # trainer.fit(model)
        # Make sure the optimizer did update the weights.

        for batch in loader:
            x, y = batch
            pred = model(x)
            loss: Tensor = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for p1, p2 in zip(model.parameters(), initial_configuration.values()):
            assert not torch.allclose(p1.data, p2.data, atol=1e-06)

    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        assert torch.allclose(p1.data, p2.data, atol=1e-06)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_multi_contiguous_params(device: str):
    if device == 'cuda' and not torch.cuda.is_available():
        print("No GPU available, skipping GPU test.")
        return

    """Verify that the parameters are the same after a few updates."""
    x = torch.randn(1, 8).to(device)

    model_ref = nn.Sequential(*[nn.Linear(8, 8) for i in range(10)])
    model_ref = model_ref.to(device)
    optimizer = torch.optim.SGD(model_ref.parameters(), lr=1e-3)

    model_c: nn.Module = copy.deepcopy(model_ref)
    parameters_c = ContiguousParams([
        {'params': model_c[:5].parameters(), 'lr': 1e-3},
        {'params': model_c[5:].parameters(), 'lr': 1e-3}
    ])
    optimizer_c = torch.optim.SGD(parameters_c.contiguous())

    for model, optimizer in zip([model_ref, model_c], [optimizer, optimizer_c]):
        for step in range(5):
            loss: Tensor = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Verify that the model/optimizer did not modify the data or grad handle.
    parameters_c.assert_buffer_is_valid()

    # Verify that both models applied the same parameter updates.
    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        assert torch.allclose(p1.data, p2.data, atol=1e-06)
