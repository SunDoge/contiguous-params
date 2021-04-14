from typing import Iterable, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter


class SingleContiguousParams:

    def __init__(
        self,
        params: Iterable[Parameter],
    ) -> None:

        params = list(params)

        param_buffer = self.init_buffer(params)
        grad_buffer = param_buffer.clone()

        data_pointers, grad_pointers = self.make_contiguous_params(
            params, param_buffer, grad_buffer
        )

        self._params = params
        self._param_buffer = param_buffer
        self._grad_buffer = grad_buffer
        self._data_pointers = data_pointers
        self._grad_pointers = grad_pointers

    def contiguous(self) -> List[Tensor]:
        return [self._param_buffer]

    def parameters(self) -> Iterable[Parameter]:
        return iter(self.contiguous())

    def original(self) -> List[Parameter]:
        return self._params

    def buffer_is_valid(self) -> bool:
        params_and_pointers = zip(
            self._params,
            self._data_pointers,
            self._grad_pointers
        )

        return all(
            p.data.data_ptr() == data_ptr and
            p.grad.data.data_ptr() == grad_ptr
            for p, data_ptr, grad_ptr in params_and_pointers
        )

    def assert_buffer_is_valid(self):
        if not self.buffer_is_valid():
            raise ValueError(
                "The data or gradient buffer has been invalidated. Please make "
                "sure to use inplace operations only when updating parameters "
                "or gradients."
            )

    @staticmethod
    def init_buffer(params: List[Parameter]) -> Tensor:
        dtype = params[0].dtype
        device = params[0].device

        if not all(p.dtype == dtype for p in params):
            raise ValueError("All parameters must be of the same dtype.")
        if not all(p.device == device for p in params):
            raise ValueError("All parameters must be on the same device.")

        buffer_size = sum(p.numel() for p in params)

        buffer = torch.zeros(buffer_size, dtype=dtype, device=device)

        return buffer

    @staticmethod
    def make_contiguous_params(
        params: List[Parameter],
        param_buffer: Tensor,
        grad_buffer: Tensor,
    ) -> Tuple[List[int], List[int]]:
        data_pointers = []
        grad_pointers = []

        index = 0
        for p in params:
            size = p.numel()
            param_buffer[index: index + size] = p.data.flatten()
            p.data = param_buffer[index: index + size].view_as(p.data)
            p.grad = grad_buffer[index: index + size].view_as(p.data)

            data_pointers.append(p.data.data_ptr())
            grad_pointers.append(p.grad.data.data_ptr())

            index += size

        param_buffer.grad = grad_buffer

        return data_pointers, grad_pointers


class MultiContiguousParams(SingleContiguousParams):

    def __init__(
        self,
        params: List[dict],
    ) -> None:
        self._contiguous_params = [
            SingleContiguousParams(p.pop('params'))
            for p in params
        ]
        self._params = params

    def contiguous(self) -> List[dict]:
        return [{'params': cp.contiguous(), **p} for cp, p in zip(self._contiguous_params, self._params)]

    def original(self) -> List[dict]:
        return [{'params': cp.original(), **p} for cp, p in zip(self._contiguous_params, self._params)]

    def buffer_is_valid(self) -> bool:
        return all(cp.buffer_is_valid() for cp in self._contiguous_params)

    def parameters(self) -> Iterable[Parameter]:
        return iter(cp._param_buffer for cp in self._contiguous_params)


class ContiguousParams(MultiContiguousParams):

    def __init__(
        self,
        params: Union[Iterable[Parameter], List[dict]]
    ) -> None:
        params = list(params)
        if not isinstance(params[0], dict):
            params = [{'params': params}]

        super().__init__(params)
