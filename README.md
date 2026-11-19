无人机-电力设备无人机巡检数据集，5600张，横担腐蚀、塔头损坏、绑扎松脱等缺陷标注，支持YOLO格式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b00cb2ea85f64667bdfa3bcea7debc48.png)

适合做电力巡检、缺陷检测、AI算法训练用，数据量大，标签齐全，直接可用


# 电力设备无人机巡检缺陷检测数据集
|项目|详情|
| ---- | ---- |
|数据集名称|电力设备无人机巡检缺陷检测数据集|
|标注格式|YOLO TXT标注|
|样本总量|5600张|
|类别数量|3类|
|适用场景|架空线路无人机巡检、电力设施缺陷自动识别、电网运维、输电通道智能监测算法研发|

### 类别明细
|类别ID|类别名称|中文释义|
| ---- | ---- | ---- |
|0|crossarm_corrosion|横担腐蚀|
|1|tower_head_damage|塔头损坏|
|2|binding_loose|绑扎松脱|

### 应用价值![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d8b22fda9854d809b7c84b2993a25cc.png)

无人机航拍识别电线杆、横担、绝缘子等部件的腐蚀、破损、绑扎松动故障，替代人工登杆排查，用于电网自动化巡检、电力安全预警YOLO检测模型训练。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9309c366e4a64140854932c5ba08bd4e.png)

```python
# YOLOv11n训练脚本
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov11n.pt")
    model.train(
        data="power_inspect.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0
    )
```
