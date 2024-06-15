# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import time

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode
#from word import word_detect

import paddlehub as hub
ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)# mkldnn加速仅在CPU下有效

class BaseTensor(SimpleClass):
    """Base tensor class with additional methods for easy manipulation and device handling."""

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with data and original shape.

        Args:
            data (torch.Tensor | np.ndarray): Predictions, such as bboxes, masks and keypoints.
            orig_shape (tuple): Original shape of image.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray))
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of the tensor as a numpy array."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Return a copy of the tensor on GPU memory."""
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype."""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    Args:
        orig_img (numpy.ndarray): The original image as a numpy array.
        path (str): The path to the image file.
        names (dict): A dictionary of class names.
        boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
        masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
        probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
        keypoints (List[List[float]], optional): A list of detected keypoints for each object.

    Attributes:
        orig_img (numpy.ndarray): The original image as a numpy array.
        orig_shape (tuple): The original image shape in (height, width) format.
        boxes (Boxes, optional): A Boxes object containing the detection bounding boxes.
        masks (Masks, optional): A Masks object containing the detection masks.
        probs (Probs, optional): A Probs object containing probabilities of each class for classification task.
        keypoints (Keypoints, optional): A Keypoints object containing detected keypoints for each object.
        speed (dict): A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
        names (dict): A dictionary of class names.
        path (str): The path to the image file.
        _keys (tuple): A tuple of attribute names for non-empty attributes.
    """

    def __init__(self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None) -> None:
        """Initialize the Results class."""
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.speed = {'preprocess': None, 'inference': None, 'postprocess': None}  # milliseconds per image
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = 'boxes', 'masks', 'probs', 'keypoints'

    def __getitem__(self, idx):
        """Return a Results object for the specified index."""
        return self._apply('__getitem__', idx)

    def __len__(self):
        """Return the number of detections in the Results object."""
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None):
        """Update the boxes, masks, and probs attributes of the Results object."""
        if boxes is not None:
            ops.clip_boxes(boxes, self.orig_shape)  # clip boxes
            self.boxes = Boxes(boxes, self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs

    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes. This
        function is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args: Variable length argument list to pass to the function.
            **kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            Results: A new Results object with attributes modified by the applied function.
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """Return a copy of the Results object with all tensors on CPU memory."""
        return self._apply('cpu')

    def numpy(self):
        """Return a copy of the Results object with all tensors as numpy arrays."""
        return self._apply('numpy')

    def cuda(self):
        """Return a copy of the Results object with all tensors on GPU memory."""
        return self._apply('cuda')

    def to(self, *args, **kwargs):
        """Return a copy of the Results object with tensors on the specified device and dtype."""
        return self._apply('to', *args, **kwargs)

    def new(self):
        """Return a new Results object with the same image, path, and names."""
        return Results(orig_img=self.orig_img, path=self.path, names=self.names)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font=r'E:\Anaconda\envs\y8\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\simhei.ttf',
        pil=True,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.

        Example:
            ```python
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model('bus.jpg')  # results list
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.show()  # show image
                im.save('results.jpg')  # save image
            ```
        """
        #print(img)
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()


        names = self.names
        fault='无故障'
        final_text = {}  # 最终调用数据
        pred_boxes, show_boxes = self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names)

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                    2, 0, 1).flip(0).contiguous() / 255
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes and show_boxes:
            #print(names)
            text_num=[]
            text_fault=''
            warn_fault=[]
            type_number = ''
            set_label = False
            comen_faut={   '呼吸机无法启动':{'原因':'1.没有连接 AC 电源，且电池电量不足;2.AC 输入插座保险丝熔断，且电池电量不足;3.显示电缆（母板处或主机外部连接器处）脱落或连接不可靠;4. AC-DC 板硬件电路故障导致没有18.8V 电源输出，且电池电量不足;5.DC-DC 板硬件电路故障导致没有5V、3.3V、7V、12V 等直流电源输出','解决办法':'1.检查并确保 AC 电源正确连接;2.更换保险丝，更换保险丝后如果开机时故障仍然存在，则说明机器内部存在短路现象;3.检查并确保线缆可靠连接，并确保显示电缆的紧固螺钉为拧紧状态;4.检查并确保线缆可靠连接;5.更换 AC-DC 板;6.更换 DC-DC 板'},
                           '屏幕没有显示(黑屏)':{'原因':'1.逆变器连接线（包括逆变器输入线和输出线）脱落或连接不可靠;2.逆变器损坏;3.主控板硬件故障，导致背光使能信号输出无效电平;4.主控板软件故障，导致背光使能信号输出无效电平;5.LCD 损坏','解决办法':'1.检查并确保线缆可靠连接;2.更换逆变器;3.更换主控板;4.升级主控板软件;5.更换 LCD'},
                           '白屏':{'原因':'1.显示电缆（LCD 处）脱落或连接不可靠;2.主控板硬件故障，导致 LCD 供电电源 3.3V 无输出或输出异常;3.LCD损坏','解决办法':'1.检查并确保线缆可靠连接;2.更换主控板;3.更换 LCD'},
                           '花屏':{'原因':'显示电缆（LCD 处）脱落或连接不可靠，导致部分颜色信号丢失','解决办法':'检查并确保线缆可靠连接'},
                           '风扇故障':{'原因':'1.风扇堵塞;2.风扇连接线脱落或风扇故障;3.DC-DC 板给风扇供电的 12V 损坏','解决办法':'1.查看风扇堵塞情况，并清除障碍物;2.重现插拔风扇连接线或更换风扇;3.更换 DC-DC 板'},
                           '涡轮自检失败':{'原因':'1.涡轮供电连接线断开;2.涡轮故障','解决办法':'1.确认涡轮供电电线连接正确;2.更换涡轮'},
                           '02流量传感器自检失败':{'原因':'1.高压氧供气不足。2.低压氧通气。3.氧气比例阀与02流量传感器偏差较大。4.氧气比例阀故障。5.02流量传感器故障','解决办法':'1.检查是否连接高压氧气源，且气源是否充足。2.在【主菜单】→【维护】→【用户维护】→输入用户维护密码→【设置】→【气源】→【氧气气源类型】查看选择的是否是低压氧，如果是，请切换为高压氧。3.重新执行流量校准。4.在阀门诊断界面诊断氧气比例阀是否异常（参见 吸气阀或氧气比例阀状态异常诊断），如果异常更换氧气比例阀，重新执行系统自检。5.检查O2流量传感器是否正确连接，如果故障仍然存在，请更换 O2 流量传感器，重新执行系统自检。6.更换监控板。'},
                           '吸气流量传感器自检失败':{'原因':'1.吸气流量传感器与吸气阀的开阀流速偏差较大。2.吸气阀故障。3.吸气流量传感器故障','解决办法':'1.重新执行流量校准，参见流量校准。2.在阀门诊断界面诊断吸气阀是否异常（吸气阀或氧气比例阀状态异常诊断），如果异常更换吸气阀，重新执行系统自检。3.检查吸气流量传感器是否正确连接，如果故障仍然存在，更换吸气流量传感器，重新执行系统自检。4.更换监控板。'},
                           '呼气流量传感器自检失败':{'原因':'1.Ypiece没堵住或管路未连接;2.呼气流量传感器与吸气流量传感器测量流速偏差较大;3.呼气流量传感器故障','解决办法':'1.检查 Ypiece 是否堵住或管路是否连接。2.重新执行流量校准。3.检查呼气流量传感器是否正确连接，如果故障仍然存在，更换呼气流量传感器，重新执行流量校准和系统自检。4.更换监控板。'},
                           '压力传感器自检失败':{'原因':'1.Ypiece 没堵住或管路未连接。2.吸气压力传感器、呼气压力传感器测量压力和呼气阀开阀压力偏差较大。3.吸气压力、呼气压力传感器采样管未连接或漏气。4.呼气阀未安装或呼气阀供电异常。5.呼气阀异常','解决办法':'1.检查 Ypiece 是否堵住或管路是否连接。2.重新执行压力校准。3.确认吸气压力、呼气压力传感器采样管是否完好（无断裂）且正确连接。4.确认呼气阀供电电线连接正确，呼气阀安装正确，重新执行系统自检。5.在阀门诊断界面诊断呼气阀是否异常（参见呼气阀状态异常诊断），如果异常，则更换呼气阀，再执行系统自检。6.更换监控板。'},
                           '呼气阀自检失败':{'原因':'1.Ypiece没堵住或管路未连接。2.吸气压力传感器、呼气压力传感器测量压力和呼气阀开阀压力偏差较大。3.吸气压力、呼气压力传感器采样管未连接或漏气。4.呼气阀未安装或呼气阀供电异常。5.呼气阀异常','解决办法':'1.检查 Ypiece 是否堵住或管路是否连接。2.重新执行压力校准。3.确认吸气压力、呼气压力传感器采样管是否完好（无断裂）且正确连接。4.确认呼气阀供电电线连接正确，呼气阀安装正确，重新执行系统自检。5.在阀门诊断界面诊断呼气阀是否异常（参见呼气阀状态异常诊断），如果异常，则更换呼气阀，再执行系统自检。6.更换监控板。'},
                           '安全阀自检失败':{'原因':'1.Ypiece没堵住或管路未连接。2.安全阀供电异常。3.安全阀异常。','解决办法':'1.检查 Ypiece 是否堵住或管路是否连接。2.确认安全阀供电电线连接正常。3.在阀门诊断界面诊断安全阀是否异常（参见 安全阀状态异常诊断），如果异常，更换安全阀，重新执行系统自检。4.更换监控板。'},
                           '泄漏量自检失败':{'原因':'1.Ypiece没堵住或管路未连接。2.吸气流量传感器测试或压力传感器测试失败。','解决办法':'1.检查 Ypiece 是否堵住或管路是否连接。2.吸气流量传感器测试和压力传感器测试成功后，再进行泄漏测试。'},
                           '顺应性自检失败':{'原因':'1.Ypiece没堵住或管路未连接。2.吸气流量传感器测试或压力传感器测试失败。','解决办法':'1.检查 Ypiece 是否堵住或管路是否连接。2.吸气流量传感器测试和压力传感器测试成功后，再进行泄漏测试。'},
                           '管路阻力自检失败':{'原因':'1.Ypiece没堵住或管路未连接。 2.呼气流量传感器测试或压力传感器测试失败。','解决办法':'1.检查 Ypiece 是否堵住或管路是否连接。2.呼气流量传感器测试和压力传感器测试成功后，再进行管路阻力测试。'},
                           '02传感器自检失败':{'原因':'1.界面氧监测功能关闭。2.氧传感器未连接或氧传感器失效。3.涡轮测试或O2流量传感器测试或吸气流量传感器测试失败。','解决办法':'1.开启氧监测功能。2.确保氧传感器连接正确，界面无任何与氧传感器相关的报警，重新进行氧传感器测试。3.涡轮测试、O2 流量传感器测试、吸气流量传感器测试都成功后，再进行氧传感器测试。4.进行 O2 传感器校准。5.更换监控板。'},
                           'Wifi模块工作异常':{'原因':'1.Wifi天线脱落或断裂;2.Wifi模块损坏;3.监控主板出现故障','解决办法':'1.重新插拔 wifi 天线或更换天线;2.更换 wifi 模块;3.更换监控模块板卡组件'},
                           '主流C02或旁流模块异常':{'原因':'1.CO2模块与监控主板的连接线出现脱落或断开;2.CO2模块自身故障;3.监控模块主板串口通讯出现异常','解决办法':'1.重新插拔或更换相关的线材;2.更换CO2模块;3.更换监控模块主板组件'},
                           '日期时间故障':{'原因':'请重新设置日期和时间：系统中没有纽扣电池，或电池中没有电量。','解决办法':'1.更换纽扣电池，并重新设置日期和时间。2.故障仍存在，需更换主控板。'},
                           '呼末正压过高': {'原因': '呼末正压监测值高于PEEP设置值+5cmH2O。','解决方法': '1.参见压力传感器测量准确性的检测章节，检查压力传感器，如果不准，请重新校准；2.检查参数设置。'},
                           '呼末正压过低': {'原因': '呼末正压监测值高于PEEP设置值+5cmH2O。','解决方法': '1. 参见压力传感器测量准确性的检测章节，检查压力传感器，如果不准，请重新校准；2.检查呼气阀安装是否正确； 3.检查参数设置。'},
                           '管道堵塞': {'原因': '病人端管路出现阻塞', '解决办法': '1.检查病人端管路是否有阻塞，如果有，请疏通； 2.参见流量传感器测量准确性的检测章节，检查流量传感器，如果不准，请重新校准； 3. 参见 压力传感器测量准确性的检测章节，检查压力传感器，如果不准，请重新校准。'},
                           '吸气支路管道堵塞': {'原因': '氧疗时，病人端管路出现弯折或阻塞;', '解决办法': '1.检查病人端管路是否有阻塞或弯折，如果有，请疏通；2.参见压力传感器测量准确性的检测章节，检查吸气压力传感器。如果不准，请重新校准。'},
                           '气道压力过高': {'原因': '病人气道压力持续在一个较高水平', '解决办法': '1.检查参数设置；2. 参见压力传感器测量准确性的检测章节，检查压力传感器，如果不准，请重新校准'},
                           '管道泄漏': {'原因': '病人端管路出现了泄漏', '解决办法': '1.检查病人端管路是否有泄漏，如果泄漏请更换管路；2. 参见流量传感器测量准确性的检测章节，检查流量传感器，如果不准，请重新校准； 3. 参见压力传感器测量准确性的检测章节，检查压力传感器，如果不准，请重新校准。'},
                           '管道断开': {'原因': '病人端管路脱落', '解决办法': '1.检查病人端管路是否脱落或松动，如果有，请重新连接； 2. 参见 流量传感器测量准确性的检测章节，检查流量传感器，如果不准，请重新校准；3. 参见压力传感器测量准确性的检测章节，检查压力传感器，如果不准，请重新校准。'},
                           '压力限制': {'原因': '压力达到压力报警高限-5cmH2O', '解决办法': '1.检查参数设置，包括压力报警高限设置；2.检查是否有压力传感器故障报警（对应报警字符串为“机器故障 09”和“机器故障 21”），如果故障请更换； 3.如果故障仍然存在，请更换监控板。'},
                           '容量限制': {'原因': '压力模式下，送气超过设定潮气量上限，提前转为呼气', '解决办法': '1.检查参数设置，包括呼出潮气量报警高限设置； 2.检查是否有“请检查呼气流量传感器”报警，如果有该报警，请先排除该报警；3.如果故障仍然存在，请更换监控板。'},
                           '吸气压力未达到': {'原因': '气道峰压没有达到设定值', '解决办法': '1.检查管道是否泄漏，如果有，请重新连接； 2. 参见压力传感器测量准确性的检测章节，检查压力传感器，如果不准，请重新校准； 3.检查参数设置。 '},
                           '呼出潮气量过低': {'原因': '潮气量没有达到设定值','解决方法': '1.检查管道是否泄漏，如果有，请重新连接；2. 参见流量传感器测量准确性的检测章节，检查流量传感器，如果不准，请重新校准；3.检查参数设置是否合理。 '},
                           '潮气量未达到': {'原因': '潮气量没有达到设定值','解决方法': '1.检查管道是否泄漏，如果有，请重新连接；2. 参见流量传感器测量准确性的检测章节，检查流量传感器，如果不准，请重新校准；3.检查参数设置是否合理。 '},
                           '叹息周期压力限制': {'原因': '叹息功能启动后，叹息周期的压力达到压力报警高限-5cmH2O', '解决办法': '1.检查参数设置，包括压力报警高限设置；2.检查是否有压力传感器故障报警（对应报警字符串为“机器故障 09”和“机器故障 21”），如果故障请更换； 3.如果故障仍然存在，请更换监控板。'},
                           '氧气不足': {'原因':'氧气供应不充足', '解决办法': '1.检查是否连接高压氧气源，且气源是否充足； 2.在阀门诊断界面诊断氧气比例阀是否异常（参见吸气阀或氧气比例阀状态异常诊断），如果异常更换氧气比例阀3.更换监控板。'},
                           '吸气时间过长': {'原因':'PSV 模式自主呼吸一直不满足呼气灵敏度而使得吸气过程不能结束。', '解决办法': '1.检查参数设置；2.检查并更换压力和流量传感器。'},
                           '呼气流量传感器故障': {'原因':'呼气流量传感器故障','解决办法':'1.校零，参见压力和流量校零； 2.校准呼气流量传感器，参见流量校准； 3.更换呼气流量传感器。 '},
                           '吸入气体温度过高': {'原因':'吸入气体温度超过限制','解决办法':'1.检查机器工作环境温度是否超过厂家声称最大工作温度 40℃； 2.检查风扇入口、出风口是否被堵，如果被堵，清理异物和灰尘；检查风扇运转情况，如果异常（如异响、转速不正常等），则更换风扇； 3.在 A/D 通道中，检查吸入混合气体和吸入氧气的温度测量值是否超出 A/D 通道提供的范围，更换对应温度超限的流量传感器（混合气体流量传感器或氧气流量传感器）'},
                           '氧传感器未连接': {'原因':'没有连接氧传感器', '解决办法':'1.检查氧传感器电缆是否脱落，如果脱落，请重新连接。2.如果故障仍然存在，更换氧传感器。'},
                           '请更换氧传感器': {'原因':'氧传感器用尽', '解决办法':'更换氧传感器'},
                           '请校准氧传感器': {'原因':'氧传感器未进行校准','解决办法':'重新进行21%和100%氧传感器校准，参见O2%校准'},
                           '请进行压力校准': {'原因': '压力传感器没有进行校准','解决办法': '1.进行压力传感器校准，参见压力校准（厂家）。2.更换监控板。'},
                           '请进行流量校准': {'原因': '流量传感器和吸气阀没有进行校准', '解决办法': '1.进行流量传感器和吸气阀校准，参见流量校准（厂家）章节。2.更换监控板。'},
                           '流量传感器类型错误': {'原因': '空气流量传感器或氧气流量传感器类型错', '解决办法': '1.检查混合气体流量传感器是否是空气流量传感器，如果不是，请更换。 2.检查氧气支路流量传感器是否是氧气流量传感器，如果不是，请更换。 '},
                           '涡轮温度过高': {'原因': '涡轮温度超过一定阈值', '解决办法': '1.检查机器工作环境温度是否超过厂家声称最大工作温度40℃。2.检查风扇入口、出风口是否被堵，如果被堵，清理异物和灰尘；检查风扇运转情况，如果异常（如异响、转速不正常等），则更换风扇。'},
                           '电池温度过高': {'原因':'电池放电过程中温度偏高', '解决办法': '1.确认使用环境温度是否过高，比如超过 35℃，如果超过，则建议客户在更低的环 境温度中使用，并保证机器附近没有发热源； 2.检查风扇入口、出风口是否被堵，如果被堵，清理异物和灰尘；检查风扇运转情况， 如果异常（如异响、转速不正常等），则更换风扇；3.若以上2条均排除，则检查电池是否正常4.故障仍然存在，请更换 DC-DC 板'},
                           '技术错误01': {'原因': '按键板通讯停止', '解决办法': '1.检查按键板和主控板接口是否接触不良，如果是，请重新拔插或者更换通讯线。2.故障仍存在，检查按键板软件的正确性。3.故障仍存在，更换按键板板卡。4.故障仍存在，更换主控板板卡'},
                           '技术错误02': {'原因': '按键板自检错误', '解决办法': '1.重启机器。2.更换按键板软件。3.如果故障不能解决，更换按键板'},
                           '技术错误03': {'原因':'涡轮温度传感器故障', '解决办法': '在A/D通道查看涡轮内部温度和涡轮外部温度，如果内部温度（或外部温度）超限，更换该温度传感器。'},
                           '技术错误05': {'原因': '环境大气压力传感器故障', '解决办法': '1.检查是否同时存在“技术错误 06”报警，如果存在，在 A/D 通道查看“Pfilter 压力传感器测量压力值”和“大气压力传感器测量大气压”哪个数值更接近当前环境大气压，更换偏差较大的传感器。2.更换监控板'},
                           '技术错误06': {'原因': 'HEPA 过滤器压力传感器故障', '解决办法': '1.检查是否同时存在“技术错误 05”报警，如果存在，在 A/D 通道查看“Pfilter 压力传感器测量压力值”和“大气压力传感器测量大气压”哪个数值更接近当前环境大气压，更换偏差较大的传感器；2.更换监控板'},
                           '技术错误07': {'原因': '三通阀故障', '解决办法': '检查三通阀，并更换'},
                           '技术错误08': {'原因': '雾化阀故障', '解决办法': '检查雾化阀，并更换'},
                           '技术错误09': {'原因': '吸入温度传感器故障', '解决办法': '1.在 A/D 通道中，检查吸入混合气体和吸入氧气的温度测量值是过大或过小，更换存在问题的温度传感器；2. 更换吸入混合气体温度传感器和吸入氧气温度传感器'},
                           '机器故障01': {'原因': '机器内部电压异常', '解决办法': '1.测量相应测试点电压；2.故障仍然存在更换电源板'},
                           '机器故障02': {'原因': '内存异常', '解决办法': '1.重启呼吸机；2.故障仍然存在更换监控板'},
                           '机器故障03': {'原因': '电源板自检错误', '解决办法': '1.重启呼吸机；2.更换电源板软件；3.如果故障不能解决，更换电源板'},
                           '机器故障04': {'原因': '监控模块初始化错误', '解决办法': '检查监控板硬件，如果故障请更换'},
                           '机器故障05': {'原因': '监控模块通讯停止', '解决办法': '1.检查监控板和主控板接口是否接触不良，如果是，请重新拔插或者更换通讯线；2.检查监控模块是否损坏，如果损坏，请更换监控模块；3.检查主控板是否损坏，如果损坏，请更换主控板；4.检查软件版本是否兼容，若不兼容，请重新升级正确的软件版本。'},
                           '机器故障06': {'原因': '监控模块自检错误', '解决办法': '1.重启呼吸机；2.检查并更换监控模块'},
                           '机器故障07': {'原因': '吸气模块通讯停止', '解决办法': '1.重启呼吸机；2.升级吸气模块软件和监控模块软件'},
                           '机器故障08': {'原因': '呼气模块通讯停止', '解决办法': '1.重启呼吸机；2.升级呼气模块软件和监控模块软件'},
                           '机器故障09': {'原因': '压力传感器通讯停止', '解决办法': '1.更换吸气压力传感器；2.更换呼气压力传感器'},
                           '机器故障10': {'原因': '安全阀故障', '解决办法': '检查安全阀，并更换'},
                           '机器故障12': {'原因': '吸气支路故障', '解决办法': '1.检查吸气阀是否工作正常，如果工作不正常，请更换吸气阀；2.检查吸气流量传感器；3.更换吸气阀或者吸气流量传感器并校准'},
                           '机器故障13': {'原因': 'O2支路故障', '解决办法': '1.检查氧气比例阀是否工作正常，如果工作不正常，请更换氧气比例阀；2.检查氧气流量传感器；3.更换氧气比例阀或者氧气流量传感器并校准'},
                           '机器故障14': {'原因':'涡轮失效', '解决办法': '更换涡轮'},
                           '机器故障15': {'原因': '涡轮温度过高', '解决办法': '1.检查机器工作环境温度是否超过工作温度；2.检查风扇入口、出风口是否被堵，如果被堵，清理异物和灰尘；检查风扇运转情况，如果异常（如异响、转速不正常等），则更换风扇'},
                           '机器故障16': {'原因': '吸气阀脱落', '解决办法': '1.检查吸气阀连接情况；2.更换吸气阀。'},
                           '机器故障17': {'原因': '吸气模块自检错误', '解决办法': '1.重启呼吸机；2.更换监控板。'},
                           '机器故障18': {'原因': '吸气模块自检错误', '解决办法': '1.重启呼吸机；2.更换监控板。'},
                           '机器故障19': {'原因': '电源板通讯停止', '解决办法': '1.检查电源板和主控板接口是否接触不良，如果是，请重新拔插或者更换通讯线；2.检查电源板是否损坏，如果损坏，请更换电源板；3.检查主控板是否损坏，如果损坏，请更换主控板；4.检查软件版本是否兼容，若不兼容，请重新升级正确的软件版本'},
                           '机器故障21': {'原因': '压力传感器零点错误', '解决办法': '1.更换吸气压力传感器。2.更换呼气压力传感器'},
                           '机器故障22': {'原因': '保护模块通讯停止', '解决办法': '1.检查主控板是否损坏，如果损坏，请更换主控板；2.检查软件版本是否兼容，若不兼容，请重新升级正确的软件版本'},
                           '电源板技术报警': {'原因':'1.电池1发生故障类报警;2.电池2发生故障类报警', '解决办法': '1.更换电池1;2.更换电池2'},
                           '电池1故障01':{'原因':'电池1温度异常，不能充电;','解决办法':'更换电池1'},
                           '电池1故障02': {'原因': '电池1充电故障', '解决办法': '更换电池1'},
                           '电池1故障03': {'原因': '电池1老化', '解决办法': '更换电池1'},
                           '电池1故障04': {'原因': '电池1通讯异常', '解决办法': '更换电池1'},
                           '电池1故障05': {'原因': '电池1故障', '解决办法': '更换电池1'},
                           '电池2故障01':{'原因':  '电池2温度异常，不能充电;','解决办法':'更换电池2'},
                           '电池2故障02': {'原因': '电池2充电故障', '解决办法': '更换电池2'},
                           '电池2故障03': {'原因': '电池2老化', '解决办法': '更换电池2'},
                           '电池2故障04': {'原因': '电池2通讯异常', '解决办法': '更换电池2'},
                           '电池2故障05': {'原因': '电池2故障', '解决办法': '更换电池2'},
                           'C02模块故障': {'原因': '1:CO2校零失败;2:CO2初始化错误;3:CO2自检错误;4:CO2硬件错误;5:CO2通讯停止;6:CO2校零错误', '解决办法':'1.1检查CO2传感器管路连接是否正常，重新校零；1.2如果仍然出现问题，请更换 CO2 模块。2.1重新插拔 CO2 模块；2.2更换 CO2 模块。3.1重新插拔 CO2 模块；3.2更换CO2模块。4.1检查传感器是否正确连接；4.2如果连接正确，仍然出现该报警，请更换CO2模块。5.1检查CO2与主控板连接线是否有问题，如果是请重新插拔或更换通讯线；5.2检查CO2与主控板是否有损坏，如果损坏请更换。6.1按照主流CO2校零的提示，检查校零方法是否正确，如果有误，请按照正确的方法重新校零；6.2检查主流 CO2 是否有损坏，如果损坏，请更换。'},
                           'C02通讯错误': {'原因': '1.CO2 模块通讯线接触不好；2.CO2 模块有问题。', '解决方法': '1.检查、重新插拔 CO2 模块通讯线；2.更换 CO2 模块。'},
                           '分钟通气量过低':{'原因': '潮气值没有达到设定值','解决方法': '检查管道是否是否泄露'},
                           '室惠':{'原因': '潮气值没有达到设定值','解决方法': '检查管道是否是否泄露'},
                           '电池未发现':{'原因': '没有检测到电池','解决方法': '检查电池是否安装'},
                           'None':{'原因':'None','解决方法':'None'}}#科曼数据
            mindray_fault = {}#迈凯瑞数据comen_faut mindray_fault
            aeonmed_fault ={'气道压力高': {'原因（低级）': '气道压力在一个呼吸周期内高于设定值。', '解决办法（低级）': '连续3个呼吸周期或者持续15秒气道压力低于设定限值',
                                         '原因（红色高级）': '连续3个呼吸周期气道压力低于设定限值。', '解决办法（红色高级）': '连续3个呼吸周期或者持续15秒气道压力高于设定限值'},
                            '泄漏': {'原因': '连续3个呼吸周期或10秒监测的分钟漏气量高于监测的分钟通气量', '解决办法': '连续3个呼吸周期或10秒监测的分钟漏气量在正常范围'},
                            '氧气不足': {'原因（低级）': '氧气气源压力低于160KPa，且氧气浓度设为21%', '解决办法（低级）': '氧气气源压力设为160KPa',
                                        '原因（红色高级）': '氧气气源压力低于160KPa%', '解决办法（红色高级）': '氧气气源压力高于160KPa',},
                            '交流电故障': {'原因': '发生交流电故障且没有电池供电，电路板发出至少120秒报警；有电池供电时出现‘交流电故障’报警。', '解决办法': '重新连接交流电'},
                            '雾化': {'原因': '开始雾化操作。', '解决办法': '雾化操作完成或中断'},
                            '肺复张': {'原因': '开始肺复张操作。', '解决办法': '肺复张操作完成或中断'},
                            '内部电池需要校验': {'原因': '内部电池需要校验。', '解决办法': '校验内部电池，重启呼吸机'},
                            '备用电池需要校验': {'原因': '备用电池需要校验。', '解决办法': '校验备用电池，重启呼吸机'},
                            '呼气保持中断': {'原因': '呼气保持倒计时结束后，还未松开呼气保持按钮。', '解决办法': '松开呼气保持按钮'},
                            '吸气保持中断': {'原因': '吸气保持倒计时结束后，还未松开吸气保持按钮。', '解决办法': '松开吸气保持按钮'},
                            '电池电量低': {'原因': '电池供电情况下，预计机器运行时间少于30分钟', '解决办法': '外接交流电'},
                            '自主呼吸频率高': {'原因': '连续4个呼吸周期或者20秒自主呼吸频率超过设置限值', '解决办法': '连续4个呼吸周期自主呼吸频率低于设置限值'},
                            '氧浓度传感器电压过低': {'原因': '在使用前测试期间检测', '解决办法': '重复使用前测试，测量传感器信号是否正常'},
                            '氧浓度高': {'原因': '连续30秒氧浓度监测值高于设置值6%以上。', '解决办法': '连续30秒氧浓度监测值低于设定值6%以上'},
                            '氧浓度低': {'原因': '连续30秒氧浓度监测值低于设置值6%以上或者18%以下', '解决办法': '连续30秒氧浓度监测值高于设定值6%以上或者18%以上'},
                            '分钟通气量高': {'原因': '呼吸1分钟后分钟通气量高于设定值上限', '解决办法': '分钟通气量低于设定限值15秒'},
                            '分钟通气量低': {'原因': '分钟通气量1分钟内低于设定下限。', '解决办法': '分钟通气量高于设定限值的通气量15秒'},
                            '吸气时间过长': {'原因': '无创-连续三次吸气时间超过设置限值；有创-连续3个呼吸周期成人吸气时间超过5秒，小儿超过2秒。', '解决办法': '连续3次吸气时间小于设定限值（无创）或固定限值（有创）'},
                            '雾化操作中断': {'原因': '雾化操作因为模式改变而中断', '解决办法': '雾化操作继续或者通过报警静音键消除'},
                            '呼气潮气量低': {'原因（低级）': '在1个呼吸周期内呼气潮气量低于设定值。', '解决办法（低级）': '在1个呼吸周期内呼气潮气量高于设定值',
                                           '原因（红色高级）': '连续3个周期或30秒呼气潮气量始终低于设定值。','解决办法（红色高级）':'连续3个呼吸周期或者持续30秒呼气潮气量高于设定限值',},
                            '电池电量耗尽': {'原因': '电池供电情况下，预计机器运行时间少于10分钟。','解决办法': '连接交流电'},
                            '内部错误': {'原因': '内部错误。', '解决办法': '重启设备；如果报警再次发生停止使用呼吸机，记录代码并联系售后服务人员'},
                            'BDU通信故障': {'原因': '呼吸机BDU出现通信故障', '解决办法':'重启设备，如果多次发生，请联系售后服务人员'},
                            'PS通信故障': {'原因': '呼吸机PS出现通信故障', '解决办法':'重启设备，如果多次发生，请联系售后服务人员'},
                            '呼气阀加热器故障': {'原因': '系统检查呼气阀加热器故障', '解决办法':'联系售后服务人员'},
                            '气道压力持续高': {'原因': '气道压力持续15秒超过PEEP15cmH2O，呼气阀打开释放压力', '解决办法':'连续5秒超出PEEP值小于15cmH2O'},
                            '窒息': {'原因': '超过窒息报警限值的时间内，没有触发呼吸周期', '解决办法': '参照11.1.3退出备份通气'},
                            'PEEP高': {'原因': '连续3个呼吸周期PEEP值高于设定限值','解决办法': '连续3个呼吸周期或者持续15秒PEEP值低于设定限值'},
                            '管路脱落': {'原因': '呼吸管路断开', '解决办法':'重新连接呼吸管路'},
                            '管路阻塞': {'原因': '呼吸管路被阻塞', '解决办法': '解除阻塞'},
                            '泄漏超出范围': {'原因': '仅在NIV模式下-连续3个呼吸周期泄露超过最大的补偿值','解决办法':'连续3个呼吸周期泄露低于最大补偿量'},
                            '气道压力低': {'原因': '连续3个呼吸周期气道压力低于设定限值', '解决办法': '连续3个呼吸周期或者持续15秒气道压力高于设定限值'},
                            'PEEP低': {'原因': '连续3个呼吸周期PEEP值低于设定限值', '解决办法': '连续3个呼吸周期或者持续15秒PEEP值高于设定限值'},
                            '通信故障': {'原因': '通信故障', '解决办法': '屏幕与主机的连接线接紧'},
                            }
            test_texts = []
            final_fault = {}#最终调用数据
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                text_num.append(c)
                if c == 5:
                    type_number = 'comen'
                if c == 7:
                    type_number = 'mindray'
                if c == 11:
                    type_number = 'aeonmed'
                    #print('谊安')
                if c ==6:#开机自检报错
                    #print(d.xyxy[0][0].item())
                    set_label =True
                    x1 = int(d.xyxy[0][0].item())
                    y1 = int(d.xyxy[0][1].item())
                    x2 = int(d.xyxy[0][2].item())
                    y2 = int(d.xyxy[0][3].item())
                    #print(self.orig_img)
                    box_image = self.orig_img[y1:y2, x1:x2]
                    start = time.time()
                    label = ocr.recognize_text(images=[box_image])
                    spend = time.time()-start
                    #print(label)打印文字识别初次结果
                    #print(spend,'text time')文字识别时间
                    #delete_char1 = '!'
                    #delete_char2 = '！'
                    # 在你的示例数据中寻找文本为 '通过' 的字典的索引号
                    """indices  = self.find_dict_index_by_text(label[0]['data'],'失败')
                    print("字典在数组中的索引号:", indices)"""
                    data = label[0]['data']
                    sorted_data = sorted(data,key=lambda x: (x['text_box_position'][0][1], x['text_box_position'][0][0]))
                    print(sorted_data)#排序之后的文字识别列表
                    #result = text.replace('!', '').replace('！', '')
                    if len(sorted_data)>15:
                        if len(sorted_data[0]['text'])==2:
                            for i, entry in enumerate(sorted_data):
                                if entry['text'] == '失败' or entry['text'] == '失购' and i + 1 < len(label[0]['data']):
                                    if i == 0:
                                        test_texts.append('涡轮自检失败')
                                    elif i == 2:
                                        test_texts.append('02流量传感器自检失败')
                                    elif i == 4:
                                        test_texts.append('吸气流量传感器自检失败')
                                    elif i == 6:
                                        test_texts.append('呼气流量传感器自检失败')
                                    elif i == 8:
                                        test_texts.append('压力传感器自检失败')
                                    elif i == 10:
                                        test_texts.append('呼气阀自检失败')
                                    elif i == 12:
                                        test_texts.append('安全阀自检失败')
                                    elif i == 14:
                                        test_texts.append('泄漏量自检失败')
                                    elif i == 16:
                                        test_texts.append('顺应性自检失败')
                                    elif i == 18:
                                        test_texts.append('泄漏量自检失败')
                                    elif i == 20:
                                        test_texts.append('泄漏量自检失败')
                        else:
                            for i, entry in enumerate(sorted_data):
                                if entry['text'] == '失败' or entry['text'] == '失购'and i + 1 < len(label[0]['data']):
                                    if i == 1:
                                        test_texts.append('涡轮自检失败')
                                    elif i == 3:
                                        test_texts.append('02流量传感器自检失败')
                                    elif i == 5:
                                        test_texts.append('吸气流量传感器自检失败')
                                    elif i == 7:
                                        test_texts.append('呼气流量传感器自检失败')
                                    elif i == 9:
                                        test_texts.append('压力传感器自检失败')
                                    elif i == 11:
                                        test_texts.append('呼气阀自检失败')
                                    elif i == 13:
                                        test_texts.append('安全阀自检失败')
                                    elif i == 15:
                                        test_texts.append('泄漏量自检失败')
                                    elif i == 17:
                                        test_texts.append('顺应性自检失败')
                                    elif i == 19:
                                        test_texts.append('泄漏量自检失败')
                                    elif i == 21:
                                        test_texts.append('泄漏量自检失败')
                    else:
                        for item in sorted_data:
                            if '失败' in item['text']:
                                test_texts.append(item['text'])
                    print(test_texts)#打印自检错误信息
                    name = ('' if id is None else f'id:{id} ') + names[c]
                    label = (f'{name} {conf:.2f}' if conf else name) if labels else None
                    annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

                if c ==4:#运行中报警
                    #print(d.xyxy[0][0].item())
                    x1 = int(d.xyxy[0][0].item())
                    y1 = int(d.xyxy[0][1].item())
                    x2 = int(d.xyxy[0][2].item())
                    y2 = int(d.xyxy[0][3].item())
                    #print(self.orig_img)
                    box_image = self.orig_img[y1:y2, x1:x2]
                    start = time.time()
                    label = ocr.recognize_text(images=[box_image])
                    spend = time.time()-start
                    #print(spend,'text time')文字识别花费时间
                    #delete_char1 = '!'
                    #delete_char2 = '！'
                    item = label[0]['data']
                    for text in item:
                        text = text['text']
                        if len(text) > 1:
                            result1 = text.replace('!', '').replace('！', '').replace('（5）','').replace('(1）','')
                            warn_fault.append(result1)
                            label1 = (f'{result1} {conf:.2f}' if conf else result1) if labels else None
                            annotator.box_label(d.xyxy.squeeze(), label1, color=colors(c, True))
                    first_chinese_index = -1
                    # 找到第一个汉字的索引
                    if len(warn_fault)==1:
                        for i, char in enumerate(warn_fault[0]):
                            if '\u4e00' <= char <= '\u9fff':
                                first_chinese_index = i
                                break
                        # 提取第一个汉字及其后面的所有信息
                        if first_chinese_index != -1:
                            chinese_and_after = warn_fault[0][first_chinese_index:]
                            warn_fault.append(chinese_and_after)
                        else:
                            print("未找到汉字")
                    print(warn_fault)#打印错误信息

                else:
                    name = ('' if id is None else f'id:{id} ') + names[c]
                    label = (f'{name} {conf:.2f}' if conf else name) if labels else None
                    annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if type_number == 'comen':#根据型号选择错误应对办法
                final_fault = comen_faut
            elif type_number == 'aeonmed':
                final_fault = aeonmed_fault
            elif type_number == 'mindray':
                final_fault = comen_faut
            #print(final_fault)打印调用型号错误数据库
            if set_label:#标志位判断 是否自检报错和运行中报错
                for test in test_texts:#自检错误
                    if test in final_fault:
                        final_text[test] = final_fault[test]
                        # Handle the case where warn_fault is not in final_fault
                    else:
                        final_text = {f'失败{i + 1}': text for i, text in enumerate(test_texts)}
            else:
                for test in warn_fault:#报警错误
                    if test in final_fault:
                        final_text[test] = final_fault[test]
                    if test == '室息':
                        final_text['窒息'] = final_fault['窒息']
            print(final_text)#最终报错信息
            #print(text_num)检测到各种目标框的序号（序号同data.yaml一样）

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ',\n'.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if self.keypoints is not None:
            for k in reversed(self.keypoints.data):
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        return annotator.result(),final_text#图像

    def find_dict_index_by_text(self, result_data, text):
        indices = []
        for i, dictionary in enumerate(result_data):
            if 'text' in dictionary and dictionary['text'] == text:
                indices.append(i)
        return indices

    def verbose(self):
        """Return log string for each task."""
        log_string = ''
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f'{log_string}(no detections), '
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save predictions into txt file.

        Args:
            txt_file (str): txt file path.
            save_conf (bool): save confidence score or not.
        """
        boxes = self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f'{probs.data[j]:.2f} {self.names[j]}') for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *d.xywhn.view(-1))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(), )
                line += (conf, ) * save_conf + (() if id is None else (id, ))
                texts.append(('%g ' * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, 'a') as f:
                f.writelines(text + '\n' for text in texts)

    def save_crop(self, save_dir, file_name=Path('im.jpg')):
        """
        Save cropped predictions to `save_dir/cls/file_name.jpg`.

        Args:
            save_dir (str | pathlib.Path): Save path.
            file_name (str | pathlib.Path): File name.
        """
        if self.probs is not None:
            LOGGER.warning('WARNING ⚠️ Classify task do not support `save_crop`.')
            return
        for d in self.boxes:
            save_one_box(d.xyxy,
                         self.orig_img.copy(),
                         file=Path(save_dir) / self.names[int(d.cls)] / f'{Path(file_name).stem}.jpg',
                         BGR=True)

    def tojson(self, normalize=False):
        """Convert the object to JSON format."""
        if self.probs is not None:
            LOGGER.warning('Warning: Classify task do not support `tojson` yet.')
            return

        import json

        # Create list of detection dictionaries
        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
            conf = row[-2]
            class_id = int(row[-1])
            name = self.names[class_id]
            result = {'name': name, 'class': class_id, 'confidence': conf, 'box': box}
            if self.boxes.is_track:
                result['track_id'] = int(row[-3])  # track ID
            if self.masks:
                x, y = self.masks.xy[i][:, 0], self.masks.xy[i][:, 1]  # numpy array
                result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
            results.append(result)

        # Convert detections to JSON
        return json.dumps(results, indent=2)


class Boxes(BaseTensor):
    """
    A class for storing and manipulating detection boxes.

    Args:
        boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6) or (num_boxes, 7). The last two columns contain confidence and class values.
            If present, the third last column contains track IDs.
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        xyxy (torch.Tensor | numpy.ndarray): The boxes in xyxy format.
        conf (torch.Tensor | numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor | numpy.ndarray): The class values of the boxes.
        id (torch.Tensor | numpy.ndarray): The track IDs of the boxes (if available).
        xywh (torch.Tensor | numpy.ndarray): The boxes in xywh format.
        xyxyn (torch.Tensor | numpy.ndarray): The boxes in xyxy format normalized by original image size.
        xywhn (torch.Tensor | numpy.ndarray): The boxes in xywh format normalized by original image size.
        data (torch.Tensor): The raw bboxes tensor (alias for `boxes`).

    Methods:
        cpu(): Move the object to CPU memory.
        numpy(): Convert the object to a numpy array.
        cuda(): Move the object to CUDA memory.
        to(*args, **kwargs): Move the object to the specified device.
    """

    def __init__(self, boxes, orig_shape) -> None:
        """Initialize the Boxes class."""
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (6, 7), f'expected `n` in [6, 7], but got {n}'  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """Return the confidence values of the boxes."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Return the class values of the boxes."""
        return self.data[:, -1]

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Return the boxes in xywh format."""
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """Return the boxes in xyxy format normalized by original image size."""
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """Return the boxes in xywh format normalized by original image size."""
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Masks(BaseTensor):
    """
    A class for storing and manipulating detection masks.

    Attributes:
        xy (list): A list of segments in pixel coordinates.
        xyn (list): A list of normalized segments.

    Methods:
        cpu(): Returns the masks tensor on CPU memory.
        numpy(): Returns the masks tensor as a numpy array.
        cuda(): Returns the masks tensor on GPU memory.
        to(device, dtype): Returns the masks tensor with the specified device and dtype.
    """

    def __init__(self, masks, orig_shape) -> None:
        """Initialize the Masks class with the given masks tensor and original image shape."""
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Return normalized segments."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)]

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Return segments in pixel coordinates."""
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in ops.masks2segments(self.data)]


class Keypoints(BaseTensor):
    """
    A class for storing and manipulating detection keypoints.

    Attributes:
        xy (torch.Tensor): A collection of keypoints containing x, y coordinates for each detection.
        xyn (torch.Tensor): A normalized version of xy with coordinates in the range [0, 1].
        conf (torch.Tensor): Confidence values associated with keypoints if available, otherwise None.

    Methods:
        cpu(): Returns a copy of the keypoints tensor on CPU memory.
        numpy(): Returns a copy of the keypoints tensor as a numpy array.
        cuda(): Returns a copy of the keypoints tensor on GPU memory.
        to(device, dtype): Returns a copy of the keypoints tensor with the specified device and dtype.
    """

    @smart_inference_mode()  # avoid keypoints < conf in-place error
    def __init__(self, keypoints, orig_shape) -> None:
        """Initializes the Keypoints object with detection keypoints and original image size."""
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # points with conf < 0.5 (not visible)
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @property
    @lru_cache(maxsize=1)
    def xy(self):
        """Returns x, y coordinates of keypoints."""
        return self.data[..., :2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self):
        """Returns normalized x, y coordinates of keypoints."""
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @property
    @lru_cache(maxsize=1)
    def conf(self):
        """Returns confidence values of keypoints if available, else None."""
        return self.data[..., 2] if self.has_visible else None


class Probs(BaseTensor):
    """
    A class for storing and manipulating classification predictions.

    Attributes:
        top1 (int): Index of the top 1 class.
        top5 (list[int]): Indices of the top 5 classes.
        top1conf (torch.Tensor): Confidence of the top 1 class.
        top5conf (torch.Tensor): Confidences of the top 5 classes.

    Methods:
        cpu(): Returns a copy of the probs tensor on CPU memory.
        numpy(): Returns a copy of the probs tensor as a numpy array.
        cuda(): Returns a copy of the probs tensor on GPU memory.
        to(): Returns a copy of the probs tensor with the specified device and dtype.
    """

    def __init__(self, probs, orig_shape=None) -> None:
        """Initialize the Probs class with classification probabilities and optional original shape of the image."""
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """Return the index of top 1."""
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """Return the indices of top 5."""
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """Return the confidence of top 1."""
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """Return the confidences of top 5."""
        return self.data[self.top5]
