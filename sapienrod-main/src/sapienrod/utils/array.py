from typing import Union
import warp as wp


def wp_slice(a: wp.array, begin: int, end: int) -> wp.array:
    """Utility function to slice a warp array along the first dimension"""

    assert a.is_contiguous
    assert 0 <= begin <= end <= a.shape[0], f"Slice \"[{begin}:{end}]\" is out of bound for array with shape[0]={a.shape[0]}."
    return wp.array(
        ptr=a.ptr + begin * a.strides[0],
        dtype=a.dtype,
        shape=(end - begin, *a.shape[1:]),
        strides=a.strides,
        device=a.device,
        copy=False,
        owner=False,
    )
