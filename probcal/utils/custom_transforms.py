from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision.utils import _log_api_usage_once
from torchvision import tv_tensors
class Transform(nn.Module):

    # Class attribute defining transformed types. Other types are passed-through without any transformation
    # We support both Types and callables that are able to do further checks on the type of the input.
    _transformed_types: Tuple[Union[Type, Callable[[Any], bool]], ...] = (torch.Tensor, PIL.Image.Image)

    def __init__(self) -> None:
        super().__init__()
        _log_api_usage_once(self)

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        pass

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def _call_kernel(self, functional: Callable, inpt: Any, *args: Any, **kwargs: Any) -> Any:
        kernel = _get_kernel(functional, type(inpt), allow_passthrough=True)
        return kernel(inpt, *args, **kwargs)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def forward(self, *inputs: Any) -> Any:
        flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])

        self._check_inputs(flat_inputs)

        needs_transform_list = self._needs_transform_list(flat_inputs)
        params = self._get_params(
            [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
        )

        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _needs_transform_list(self, flat_inputs: List[Any]) -> List[bool]:
        # Below is a heuristic on how to deal with pure tensor inputs:
        # 1. Pure tensors, i.e. tensors that are not a tv_tensor, are passed through if there is an explicit image
        #    (`tv_tensors.Image` or `PIL.Image.Image`) or video (`tv_tensors.Video`) in the sample.
        # 2. If there is no explicit image or video in the sample, only the first encountered pure tensor is
        #    transformed as image, while the rest is passed through. The order is defined by the returned `flat_inputs`
        #    of `tree_flatten`, which recurses depth-first through the input.
        #
        # This heuristic stems from two requirements:
        # 1. We need to keep BC for single input pure tensors and treat them as images.
        # 2. We don't want to treat all pure tensors as images, because some datasets like `CelebA` or `Widerface`
        #    return supplemental numerical data as tensors that cannot be transformed as images.
        #
        # The heuristic should work well for most people in practice. The only case where it doesn't is if someone
        # tries to transform multiple pure tensors at the same time, expecting them all to be treated as images.
        # However, this case wasn't supported by transforms v1 either, so there is no BC concern.

        needs_transform_list = []
        transform_pure_tensor = not has_any(flat_inputs, tv_tensors.Image, tv_tensors.Video, PIL.Image.Image)
        for inpt in flat_inputs:
            needs_transform = True

            if not check_type(inpt, self._transformed_types):
                needs_transform = False
            elif is_pure_tensor(inpt):
                if transform_pure_tensor:
                    transform_pure_tensor = False
                else:
                    needs_transform = False
            needs_transform_list.append(needs_transform)
        return needs_transform_list

    def extra_repr(self) -> str:
        extra = []
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue

            if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)):
                continue

            extra.append(f"{name}={value}")

        return ", ".join(extra)

    # This attribute should be set on all transforms that have a v1 equivalent. Doing so enables two things:
    # 1. In case the v1 transform has a static `get_params` method, it will also be available under the same name on
    #    the v2 transform. See `__init_subclass__` for details.
    # 2. The v2 transform will be JIT scriptable. See `_extract_params_for_v1_transform` and `__prepare_scriptable__`
    #    for details.
    _v1_transform_cls: Optional[Type[nn.Module]] = None

    def __init_subclass__(cls) -> None:
        # Since `get_params` is a `@staticmethod`, we have to bind it to the class itself rather than to an instance.
        # This method is called after subclassing has happened, i.e. `cls` is the subclass, e.g. `Resize`.
        if cls._v1_transform_cls is not None and hasattr(cls._v1_transform_cls, "get_params"):
            cls.get_params = staticmethod(cls._v1_transform_cls.get_params)  # type: ignore[attr-defined]

    def _extract_params_for_v1_transform(self) -> Dict[str, Any]:
        # This method is called by `__prepare_scriptable__` to instantiate the equivalent v1 transform from the current
        # v2 transform instance. It extracts all available public attributes that are specific to that transform and
        # not `nn.Module` in general.
        # Overwrite this method on the v2 transform class if the above is not sufficient. For example, this might happen
        # if the v2 transform introduced new parameters that are not support by the v1 transform.
        common_attrs = nn.Module().__dict__.keys()
        return {
            attr: value
            for attr, value in self.__dict__.items()
            if not attr.startswith("_") and attr not in common_attrs
        }

    def __prepare_scriptable__(self) -> nn.Module:
        # This method is called early on when `torch.jit.script`'ing an `nn.Module` instance. If it succeeds, the return
        # value is used for scripting over the original object that should have been scripted. Since the v1 transforms
        # are JIT scriptable, and we made sure that for single image inputs v1 and v2 are equivalent, we just return the
        # equivalent v1 transform here. This of course only makes transforms v2 JIT scriptable as long as transforms v1
        # is around.
        if self._v1_transform_cls is None:
            raise RuntimeError(
                f"Transform {type(self).__name__} cannot be JIT scripted. "
                "torchscript is only supported for backward compatibility with transforms "
                "which are already in torchvision.transforms. "
                "For torchscript support (on tensors only), you can use the functional API instead."
            )

        return self._v1_transform_cls(**self._extract_params_for_v1_transform())

class _BaseMixUpCutMix(Transform):
    def __init__(self, *, alpha: float = 1.0, num_classes: Optional[int] = None, labels_getter="default") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

        self.num_classes = num_classes

        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)

        if has_any(flat_inputs, PIL.Image.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask):
            raise ValueError(f"{type(self).__name__}() does not support PIL images, bounding boxes and masks.")

        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")
        if labels.ndim not in (1, 2):
            raise ValueError(
                f"labels should be index based with shape (batch_size,) "
                f"or probability based with shape (batch_size, num_classes), "
                f"but got a tensor of shape {labels.shape} instead."
            )
        if labels.ndim == 2 and self.num_classes is not None and labels.shape[-1] != self.num_classes:
            raise ValueError(
                f"When passing 2D labels, "
                f"the number of elements in last dimension must match num_classes: "
                f"{labels.shape[-1]} != {self.num_classes}. "
                f"You can Leave num_classes to None."
            )
        if labels.ndim == 1 and self.num_classes is None:
            raise ValueError("num_classes must be passed if the labels are index-based (1D)")

        params = {
            "labels": labels,
            "batch_size": labels.shape[0],
            **self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
            ),
        }

        # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
        # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
        needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
        expected_num_dims = 5 if isinstance(inpt, tv_tensors.Video) else 4
        if inpt.ndim != expected_num_dims:
            raise ValueError(
                f"Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead."
            )
        if inpt.shape[0] != batch_size:
            raise ValueError(
                f"The batch size of the image or video does not match the batch size of the labels: "
                f"{inpt.shape[0]} != {batch_size}."
            )

    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
        if label.ndim == 1:
            label = one_hot(label, num_classes=self.num_classes)  # type: ignore[arg-type]
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))




class MixUp(_BaseMixUpCutMix):
    """Apply MixUp to the provided batch of images and labels.

    Paper: `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int, optional): number of classes in the batch. Used for one-hot-encoding.
            Can be None only if the labels are already one-hot-encoded.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``MixUp()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))  # type: ignore[arg-type]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]

        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=lam)
        elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))

            if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
                output = tv_tensors.wrap(output, like=inpt)

            return output
        else:
            return inpt
