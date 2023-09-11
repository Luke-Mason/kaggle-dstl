import json
from pathlib import Path
import attr

@attr.s(slots=True)
class HyperParams:
    classes = attr.ib(default=list(range(10)))
    net = attr.ib(default='UNet')
    n_channels = attr.ib(default=12)  # max 20
    total_classes = 10
    thresholds = attr.ib(default=[0.5])
    pre_buffer = attr.ib(default=0.0)

    patch_size = attr.ib(default=64)
    patch_overlap_size = attr.ib(default=16)

    augment_rotations = attr.ib(default=10.0)  # degrees
    augment_flips = attr.ib(default=0)
    augment_channels = attr.ib(default=0.0)

    validation_square = attr.ib(default=400)

    dropout = attr.ib(default=0.0)
    bn = attr.ib(default=1)
    activation = attr.ib(default='relu')
    top_scale = attr.ib(default=2)
    log_loss = attr.ib(default=1.0)
    dice_loss = attr.ib(default=0.0)
    jaccard_loss = attr.ib(default=0.0)
    dist_loss = attr.ib(default=0.0)
    dist_dice_loss = attr.ib(default=0.0)
    dist_jaccard_loss = attr.ib(default=0.0)

    filters_base = attr.ib(default=32)

    n_epoch = attr.ib(default=100)
    oversample = attr.ib(default=0.0)
    lr = attr.ib(default=0.0001)
    lr_decay = attr.ib(default=0.0)
    weight_decay = attr.ib(default=0.0)
    batch_size = attr.ib(default=128)

    @property
    def n_classes(self):
        """
        :return: Number of classes enabled
        """
        return len(self.classes)

    @property
    def has_all_classes(self):
        """
        :return: True if all classes are enabled
        """
        return self.n_classes == self.total_classes

    @property
    def needs_dist(self):
        """
        :return: True if any of the distance losses are enabled
        """
        return (self.dist_loss != 0 or self.dist_dice_loss != 0 or
                self.dist_jaccard_loss != 0)

    @classmethod
    def from_dir(cls, root: Path):
        """
        :param root: Root directory
        :return: HyperParams object loaded from hps.json in root
        """
        params = json.loads(root.joinpath('hps.json').read_text())
        fields = {field.name for field in attr.fields(HyperParams)}
        return cls(**{k: v for k, v in params.items() if k in fields})

    def update(self, hps_string: str):
        """
        Update hyperparams from a string of comma-separated key=value pairs
        :param hps_string: The string to parse
        :return: None
        """
        if hps_string:
            values = dict(pair.split('=') for pair in hps_string.split(','))
            for field in attr.fields(HyperParams):
                v = values.pop(field.name, None)
                if v is not None:
                    default = field.default
                    assert not isinstance(default, bool)
                    if isinstance(default, (int, float, str)):
                        v = type(default)(v)
                    elif isinstance(default, list):
                        v = [type(default[0])(x) for x in v.split('-')]
                    setattr(self, field.name, v)
            if values:
                raise ValueError('Unknown hyperparams: {}'.format(values))
