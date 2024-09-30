import os
import importlib.util
import colorlog
from pathlib import Path
from absl import logging, app, flags
from socket import gethostname
import sys
from ml_collections import ConfigDict, config_dict, FieldReference, config_flags

reserved_flag_names = [
    "logtostderr",
    "alsologtostderr",
    "log_dir",
    "v",
    "verbosity",
    "logger_levels",
    "stderrthreshold",
    "showprefixforinfo",
    "run_with_pdb",
    "pdb_post_mortem",
    "pdb",
    "run_with_profiling",
    "profile_file",
    "use_cprofile_for_profiling",
    "only_check_args",
]


def eval_parser(argument):
    return (
        eval(
            argument.lower()
            .replace("k", "*1024")
            .replace("m", "*1024**2")
            .replace("g", "*1024**3")
            .replace("b", "*1024**3")
            .replace("t", "*1024**4")
        )
        if isinstance(argument, str)
        else argument
    )


def parse_args(parts):
    processed = [False] * len(parts)
    args = {}
    for i, part in enumerate(parts):
        if processed[i]:
            continue
        if "=" in part:
            key, value = part.split("=")
            args[key.lstrip("-")] = value
            processed[i] = True
        else:
            if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                args[part.lstrip("-")] = parts[i + 1]
                processed[i] = True
                processed[i + 1] = True
            else:
                args[part.lstrip("-").lstrip("no")] = "no" not in part
                processed[i] = True

    return args


def is_notebook() -> bool:
    # Is environment a Jupyter notebook? Verified on Colab, Jupyterlab, Kaggle, Paperspace
    # This is actually is_ipython()
    try:
        import IPython

        ipython_type = str(type(IPython.get_ipython()))
        return any(kw in ipython_type for kw in ["colab", "zmqshell", "ipykernel"])
    except:
        return False


is_ipython = is_notebook


def lazy_apply(parent: FieldReference, fn, field_type=None):
    child = parent._apply_op(fn)  # pylint: disable = protected-access

    if field_type is not None:
        child._field_type = field_type  # pylint: disable = protected-access

    a_child_value = child.get()
    if not isinstance(
        a_child_value,
        child._field_type,  # pylint: disable = protected-access
    ):
        raise TypeError(
            f"Expected operation result to be of type {field_type}. "
            f"Instead however it was {type(a_child_value)}. "
            "Adjust the `field_type` parameter accordingly."
        )

    return child


def import_from_file(file):
    file = str(file)
    specs = importlib.util.spec_from_file_location("_aaaa", file)
    M = importlib.util.module_from_spec(specs)
    specs.loader.exec_module(M)
    return M


class IPytorchFormatter(colorlog.ColoredFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = os.environ.get("RANK", "0")
        self.world_size = os.environ.get("WORLD_SIZE", "1")
        rank = f"RANK {self.rank}/{self.world_size}" if int(self.world_size) > 1 else ""
        self.prefix = f"{rank} {gethostname()}: "

    def format(self, record):
        return self.prefix + super().format(record)
        # record = ColoredRecord(record, self.escape_codes)


def init_logger():
    ipformatter = IPytorchFormatter(
        # "%(log_color)s%(levelname)s %(asctime)s %(thread)d:%(filename)s:%(lineno)d] %(message)s",
        "%(log_color)s%(levelname)s %(asctime)s %(name)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%I:%M:%S %p",
    )

    logging.get_absl_handler().setFormatter(ipformatter)
    logging.use_absl_handler()
    logging._warn_preinit_stderr = 0
    logging.set_verbosity(logging.INFO)


def init_flags(
    predefined=False,
    alias=False,
):
    if flags.FLAGS.is_parsed():
        return flags.FLAGS, sys.argv
    if is_ipython() and ("-f" in sys.argv):
        sys.argv = ["python", "some.py"]
    if predefined:
        if Path("config.py").exists():
            logging.info("Using local config")
            config_flags.DEFINE_config_dict(
                "c", import_from_file("config.py").get_config(), lock_config=False
            )
    if alias:
        defined_flag_names = [
            name for name in flags.FLAGS if name not in reserved_flag_names
        ]
        logging.debug(f"Defined flags: {defined_flag_names}")
        for flag_name in defined_flag_names:
            if "-" in flag_name:
                flags.DEFINE_alias(flag_name.replace("-", "_"), flag_name)

    cmd = None
    if len(sys.argv) > 1:
        maybe_cmd = sys.argv[1]
        if not (maybe_cmd.startswith("-") or Path(maybe_cmd).exists()):
            cmd = sys.argv.pop(1)

    logging.info(sys.argv)

    args = app._run_init(sys.argv, app.parse_flags_with_usage)
    return cmd, args


init_logger()

flags.IntegerParser.convert = lambda _, argument: int(eval_parser(argument))
flags.FloatParser.convert = lambda _, argument: float(eval_parser(argument))

flags.Flag.__getstate__ = lambda self: self.__dict__
flags.FlagValues.__getstate__ = lambda self: self.__dict__
flags.FlagValues.__setstate__ = lambda self, state: self.__dict__.update(state)

flags.init = init_flags

__all__ = ["logging", "flags", "lazy_apply", "ConfigDict"]
