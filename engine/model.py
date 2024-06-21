# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
import sys
from pathlib import Path
from typing import Union

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.hub.utils import HUB_WEB_ROOT
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, callbacks, checks, emojis, yaml_load


class Model(nn.Module):
    """
    model (str, Path): 要加载或创建的模型文件的路径。
    task (Any, optional): YOLO模型的任务类型。默认为None。
    属性：

    predictor (Any): 预测器对象。
    model (Any): 模型对象。
    trainer (Any): 训练器对象。
    task (str): 模型任务类型。
    ckpt (Any): 如果从*.pt文件加载了模型，它是检查点对象。
    cfg (str): 如果从*.yaml文件加载了模型，它是模型配置。
    ckpt_path (str): 检查点文件路径。
    overrides (dict): 训练器对象的覆盖参数。
    metrics (Any): 指标数据。
    方法：

    call(source=None, stream=False, **kwargs): 预测方法的别名。
    _new(cfg:str, verbose:bool=True) -> None: 初始化一个新模型，并从模型定义中推断任务类型。
    _load(weights:str, task:str=‘’) -> None: 初始化一个新模型，并从模型头推断任务类型。
    _check_is_pytorch_model() -> None: 如果模型不是PyTorch模型，则引发TypeError。
    reset() -> None: 重置模型模块。
    info(verbose:bool=False) -> None: 记录模型信息。
    fuse() -> None: 合并模型以加速推理。
    predict(source=None, stream=False, **kwargs) -> List[ultralytics.engine.results.Results]: 使用YOLO模型进行预测。
    返回值：

    list(ultralytics.engine.results.Results): 预测结果。

    """

    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        model = str(model).strip()  # strip spaces

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            from ultralytics.hub.session import HUBTrainingSession
            self.session = HUBTrainingSession(model)
            model = self.session.model_file

        # Check if Triton Server model
        # 检查 Triton Server 是否已经加载了模型。
        elif self.is_triton_model(model):
            self.model = model
            self.task = task
            return

        # Load or create new YOLO model
        # 加载或创建新的YOLO模型
        model = checks.check_model_file_from_stem(model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        if Path(model).suffix in ('.yaml', '.yml'):
            self._new(model, task)
        else:
            self._load(model, task)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection.
            调用具有给定参数的’predict’函数来执行目标检测。
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model):
        """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task_name>
            模型是否一个Triton Server URL字符串
        """
        from urllib.parse import urlsplit
        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {'http', 'grfc'}

    @staticmethod
    def is_hub_model(model):
        """Check if the provided model is a HUB model.
            检查提供的模型是否为HUB模型
        """
        return any((
            model.startswith(f'{HUB_WEB_ROOT}/models/'),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
            [len(x) for x in model.split('_')] == [42, 20],  # APIKEY_MODELID
            len(model) == 20 and not Path(model).exists() and all(x not in model for x in './\\')))  # MODELID

    def _new(self, cfg: str, task=None, model=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
            verbose (bool): display model info on load
        这段代码是用于初始化一个新的模型并从模型定义中推断任务类型的。它接受以下参数：

        cfg (str)：模型配置文件的路径。
        task (str | None)：模型的任务类型，可以是字符串或None。
        model (BaseModel)：自定义的模型对象。
        verbose (bool)：是否在加载模型时显示模型信息。
        根据提供的参数，代码会执行以下操作：

        读取模型配置文件（cfg）。
        如果提供了任务类型（task），则将其赋值给model.task；否则，根据模型配置文件中的配置自动推断任务类型。
        根据推断出的任务类型，设置相应的模型属性和参数。
        如果verbose为True，则打印模型信息。
        返回初始化后的模型对象。
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load('model'))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides['model'] = self.cfg
        self.overrides['task'] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    def _check_is_pytorch_model(self):
        #检查是否是pytorch模型
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == '.pt'
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'")

    def reset_weights(self):
        """Resets the model modules parameters to randomly initialized values, losing all training information."""
        #重置weights
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights='yolov8n.pt'):
        """Transfers parameters with matching names and shapes from 'weights' to model."""
        #将具有相同名称和形状的参数从'weights'传输到模型。
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """Fuse PyTorch Conv2d and BatchNorm2d layers."""
        self._check_is_pytorch_model()
        self.model.fuse()

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        """
        Perform prediction using the YOLO model.

        参数：

        source (str | int | PIL | np.ndarray)：要进行预测的图像源。接受YOLO模型接受的所有源类型。
        stream (bool)：是否流式传输预测。默认为False。
        predictor (BasePredictor)：自定义的预测器。
        **kwargs：传递给预测器的其他关键字参数。请查看文档中“配置”部分以获取所有可用选项。
        返回值：

        (List[ultralytics.engine.results.Results]): 预测结果列表。
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        #print(is_cli)

        custom = {'conf': 0.25, 'save': is_cli}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'predict'}  # highest priority args on the right
        prompts = args.pop('prompts', None)  # for SAM-type models

        if not self.predictor:
            self.predictor = (predictor or self._smart_load('predictor'))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if 'project' in args or 'name' in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, 'set_prompts'):  # for SAM-type models
            self.predictor.set_prompts(prompts)

        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)


    def track(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.

        Args:
            source (str, optional): The input source for object tracking. Can be a file path or a video stream.
            stream (bool, optional): Whether the input source is a video stream. Defaults to False.
            persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): The tracking results.
        """
        if not hasattr(self.predictor, 'trackers'):
            from ultralytics.trackers import register_tracker
            register_tracker(self, persist)
        kwargs['conf'] = kwargs.get('conf') or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)

    def val(self, validator=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            validator (BaseValidator): Customized validator.
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        custom = {'rect': True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'val'}  # highest priority args on the right

        validator = (validator or self._smart_load('validator'))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(self, **kwargs):
        """
        Benchmark a model on all export formats.

        Args:
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {'verbose': False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, 'mode': 'benchmark'}
        return benchmark(
            model=self,
            data=kwargs.get('data'),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args['imgsz'],
            half=args['half'],
            int8=args['int8'],
            device=args['device'],
            verbose=kwargs.get('verbose'))

    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the Exporter. To see all args check 'configuration' section in docs.
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {'imgsz': self.model.args['imgsz'], 'batch': 1, 'data': None, 'verbose': False}  # method defaults
        args = {**self.overrides, **custom, **kwargs, 'mode': 'export'}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, trainer=None, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            trainer (BaseTrainer, optional): Customized trainer.
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs['cfg'])) if kwargs.get('cfg') else self.overrides
        custom = {'data': TASK2DATA[self.task]}  # method defaults
        args = {**overrides, **custom, **kwargs, 'mode': 'train'}  # highest priority args on the right
        if args.get('resume'):
            args['resume'] = self.ckpt_path

        self.trainer = (trainer or self._smart_load('trainer'))(overrides=args, _callbacks=self.callbacks)
        if not args.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(self, use_ray=False, iterations=10, *args, **kwargs):
        """
        Runs hyperparameter tuning, optionally using Ray Tune. See ultralytics.utils.tuner.run_ray_tune for Args.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune
            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, 'mode': 'train'}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn):
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides['device'] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    def add_callback(self, event: str, func):
        """Add a callback."""
        self.callbacks[event].append(func)

    def clear_callback(self, event: str):
        """Clear all event callbacks."""
        self.callbacks[event] = []

    def reset_callbacks(self):
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")) from e

    @property
    def task_map(self):
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        raise NotImplementedError('Please provide task map for your model!')
