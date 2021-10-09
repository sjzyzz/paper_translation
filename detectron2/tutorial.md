# 使用自定义的数据集

这篇文档解释了数据集API(DatasetCatalog, MetadataCatalog)是如何工作的，以及如何使用它们来添加自定义的数据库

在detectron2中集成支持的数据里列在了[这里](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html)。如果你想使用自定义的数据集同时复用detectron2的数据加载器，你需要做的是：
- 注册你的数据集(也就是，告诉detectron2如何得到你的数据集)
- 以及可选的，为你的数据集注册元数据
接下来，我们详细解释以上两个概念

## 注册数据集

为了让detectron2知道如何得到名为“my_dataset”的数据集，用户需要实现一个返回数据集中的物体的函数并告诉detectron2这个函数：

```python
def my_dataset_function():
    ...
    return list[dict] in the following format
from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", my_dataset_function)
# later, to access the data:
data: List[Dict] = DatasetCatalog.get("my_dataset")
```

这里，这个代码段将名为“my_dataset”的数据集和返回数据的函数关联了起来。如果这个函数被多次使用，那么它必须按照相同的顺序返回相同的数据。知道进程结束，这个注册一直有效。

这个函数可以做任意的事情并应该返回`list[dict]`形式的数据，每一个dict应该是如下的一种格式：

- 下面将要描述的detectron2的标准数据集字典。这将会使它可以和detectron2的集成支持特性一同工作，所以当可行时，这是推荐的方式
- 任何自定义的格式。你也可以返回任意自定格式的字典，例如为新任务添加额外的键。这是你将需要在下游任务中合适地处理它们。下面将会展示更多细节。

## 标准数据集字典

对于标准任务（实例检测、实例/语义/全景分割和关键点检测），我们以类似于COCO标注的规范将原本的数据集载入`list[dict]`。这是我们对于数据集的标准表示。

每一个字典包含一张图片的信息。字典可能有如下的域，以及要求的域随着数据加载器和任务的需要而变化（更多见下表）。

| **任务** | **域** |
|---------|--------|
|共同的|file_name, height, width, image_id|
|实例检测/分割|annotations|
|语义分割|sem_seg_file_name|
|全景分割|pan_seg_file_name, segments_info|

- `file_name`：图片文件的完全路径
- `height, width`：整数。图片的形状
- `image_id`：字符串或者整数。可以识别图片的唯一id。许多评价者为了识别图片可能会要求，但是数据集也可以处于不同的目的使用它
- `annotations`(list[dict])：实例检测/分割或者关键点检测任务要求这个域。每一个字典对应于图片中一个实例的标注，可能包含如下键：
  - `bbox`(list[float])：表示实例边界框的包含四个数字的列表
  - `bbox_mode`(int)：边界框的格式。它一定是[structrues.BoxMode](https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BoxMode)中的一员。目前支持：`BoxMode.XYXY_ABS`和`BoxMode.XYWH_ABS`
  - `category_id`(int)：一个[0, num_categories-1]中的整数，表示类别标签。如果需要，num_categories被保留用来表示“背景”类别。
  - `segmentation`(list[list[float]]或者dict)：实例的分割掩码
    - 如果是`list[list[float]]`形式，它表示一个多边形列表，其中每一个多边形表示物体的连接的组分。每一个`list[float]`是一个格式为`[x1, y1, ..., xn, yn]`（n≥3）的简单多边形。其中Xs和Ys是单位像素下的绝对坐标。
    - 如果是`dict`形式，它表示
  - `keypoints`
- `sem_seg_file_name`
- `pan_seg_file_name`
- `segments_info`
  - `id`
  - `category_id`
  - `iscrowd` 
  - 
## 为了新任务自定义数据集字典

在你的数据集函数返回的`list[dict]`中，这个字典也可以包含**任意自定义的数据**。这对于需要使用没有被标准数据字典包含的额外信息的新任务有帮助。在这种情况下，你需要保证下游代码可以正确处理你的数据。这经常要求为数据加载器写一个新的`mapper`（见[使用自定义数据加载器](https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html)）。

当设计一个自定义格式时，注意所有的字典都储存在内存中（有时被序列化并有多份拷贝）。为了节约内存，每个字典应该包含关于每个样本少但是充分的信息，例如文件名和标注。加载全部样本通常在数据加载器中发生。

对于在整个数据集中共享的属性，使用`Metadata`（见后文）。为了避免额外的内存，不要将这种信息存在每一个样本中。

## 数据集的元数据

每个数据集斗鱼某些元数据向关联，可以通过`MetadataCatalog.get(dataset_name).some_metadata`访问。元数据是包含整个数据集共享的信息的键值对映射，通常被用来解释数据集中有什么，例如，类别的名字、类别的颜色和根文件等等。这些信息对于数据增强、评估、可视化和日志等很有用。元数据的结构取决于对应下游代码中的需要。

如果你通过`DatasetCatalog.register`注册了一个新数据集，为了附能一些需要元数据的特性，你可能也想通过`MetadataCatalog.get(dataset_name).some_key = some_value`来添加它对应的元数据。你可以这样做（使用元数据键“thing_classes”作为例子）：

```python
from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
```

这里是使用detectron2内建特性的元数据列表。如果你在没有这些元数据的情况下添加了你自己的数据集，某些特性可能不可用：
- `thing_classes`
- `thing_colors`
- `stuff_classes`
- `ignore_label`
- `keypoint_names`
- `keypoint_flip_map`
- `keypoint_connection_rules`
一些额外的为了评估某些数据集（例如COCO）的特定元数据
- `thing_dataset_id_to_contiguous_id`
- `stuff_dataset_id_to_contiguous_id`
- `json_file`
- `panoptic_root`和`panoptic_json`
- `evaluator_type`
- 
## 注册一个COCO格式的数据集

如果你的实例层级（检测、分割和关键点）数据集已经是COCO格式下的json文件，这个数据集以及它关联的元数据可以如下轻易注册：
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```
如果你的数据集是COCO的格式但是需要进一步处理，或者有额外的自定义标注，[load_coco_json](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.datasets.load_coco_json)函数可能会有用。

## 为了新的数据集更新配置

一旦你注册了数据集，你可以在`cfg.DATASETS.{TRAIN, TEST}`中使用数据集的名字。这里还有一些其他的你为了在新的数据集上训练或者评测可能想要改变的配置：
- `MODEL.ROI_HEAD.NUM_CLASSES`和`MODEL.RETINANET.NUM_CLASSES`
- `MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS`
- `MODEL>SEM_SEG_HEAD.NUM_CLASSES`
- `TEST.DETECTIOND_PER_IMAGE`
- `DATASETS.PROPOSAL_FILES_{TRAIN, TEST}`
新的模型经常有它们自己类似的需要更改配置。

# Dataloader

数据加载器是为模型提供数据的模块。一个数据加载器通常（但并不是一直）使用来自数据集的未加工信息，并且将它们处理为模型需要的格式

## 现有的数据加载器是如何工作的

detectron2包含一个内建的数据加载流水线。理解它是如何工作是好的，万一你需要自己写一个呢。

detectron2提供了两个函数[build_detection_{train,test}_loader](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.build_detection_train_loader)来从给定的配置创建数据加载器。这里介绍`build_detection_{train,test}_loader`是如何工作的：
- 它接收一个已经注册数据集的名字（例如，"coco_2017_train"）并加载一个以轻量化格式表示数据集项的`list[dict]`。这些数据集项并没有准备好被模型使用（例如，图片还没有被载入内存、随机的数据增强还没有被应用等等）。数据集格式和数据集注册的细节可以在[这里](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)被找到。
- 列表中的每一个dict都会被函数（“mapper”）映射：
  - 用户可以通过指定`build_detection_{train,test}_loader`中的“mapper”参数来自定义这个映射函数。默认的mapper是[DatasetMapper](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.DatasetMapper)
  - mapper的输出格式可以使任意的，只要它被这个数据加载器的消费者（通常是模型）接受。默认mapper的输出，在batching后，遵循[Use Model](https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format)中指出的默认的模型输入格式。
  - mapper的作用是将数据集项的轻量化表示转化为准备好模型使用的格式（包括，例如读入图片、进行随机的数据增强和转化为torch张量）。如果你想要对数据进行自定义的转化，你经常想要写一个自定义的mapper
- mapper的输出是batched（简单的放入一个列表中）
- 这个batched数据就是数据加载器的输出。一般来说，也是`model.forward()`的输入

## 写一个自定义的数据加载器

使用不同mapper的`build_detection_{train,test}_loader`适用大部分自定义数据加载的情况。例如，如果你为了训练想将所有的图片裁剪为固定的尺寸，使用：

```python
    import detectron2.data.transforms as T
    from detectron2.data import DatasetMapper
    dataloader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.Reszie((800, 800))]))
```

如果默认[DatasetMapper](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.DatasetMapper)的参数没有提供你需要的，你可以写一个自己的mapper函数并使用它，例如：

```python
    from detectron2.data import detection_utils as utils
    def mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        auginput = T.AugInput(image)
        transfrom = T.Resize((800, 800))(auginput)
        image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")    
        ]
        return {
            "image":image,
            "instances": utils.annotations_to_instances(annos, image.shape[1:])
        }
    dataloader = build_detection_train_loader(cfg, mapper=mapper)
```

如果你不仅想改变mapper（例如，为了实现不同的采样或者batching逻辑），那么`build_detection_{train,test}_loader`不再适用，你需要写一个不同的数据加载器。数据加载器就是一个生产模型接受格式的Python的迭代器。你可以通过任何你喜欢的工具实现它。

无论实现什么，我们推荐你去查看[API documentation of detectron2.data](https://detectron2.readthedocs.io/en/latest/modules/data.html)来学习更多关于这些函数的API。

## 使用一个自定义的数据加载器

如果你使用[DefaultTrainer](https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.defaults.DefaultTrainer)，你可以重写它的`build_{train,test}_loader`来使用你自己的数据加载器。例子可见[deeplab dataloader](https://github.com/facebookresearch/detectron2/blob/master/projects/DeepLab/train_net.py)

如果你要写自己的训练循环，你可以轻易的插入你的数据加载器

# 数据增强

增强是训练的重要部分。detectron2的数据增强系统意在达到如下目标：
- 允许同时增强多个数据类型（例如，和图片一起的边界框和掩码）
- 允许应用一系列静态声明的增强
- 允许为增强添加自定义的新数据类型（旋转边界框，视频片段等等）
- 处理和操作由增强应用的操作
前两个特性包含了大部分通产使用的情况，并且在例如[albumentations](https://medium.com/pytorch/multi-target-in-albumentations-16a777e9006e)的其他库中也可用。支持其他特性为detectron2的增强API添加了一些优势，这也是我们将会在这个教程中解释的

这个教程聚焦于当写新的数据加载器时如何使用增强，以及如何写新的增强。如果你使用detctron2中的默认数据加载器，正如[Dataloader tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html)中解释的，它已经支持接收一个用户提供的自定义增强列表

## 基本用法

特征1和特征2的基本用法如下：

```python
from detectron2.data import transforms as T
# 定义一个增强序列
augs = T.AugmentationList([
    T.RandomBrightness(0.9, 1.1),
    T.RandomFlip(prob=0.5),
    T.RandomCrop("absolute", (640, 640))
]) # 类型：T.Augmentation
# 定义增强输入
input = T.AugInput(image, boxes=boxes, sem_seg=sem_seg)
# 应用增强
transform = augs(input)
image_transfomed = input.image # 新图片
sem_seg_transformed = input.sem_seg # 新语义分割

# 对于其他需要被一起增强的额外数据，使用transform，例如：
image2_transformed = transform.apply_image(mage2)
polygons_transformed = transform.apply_polygons(polygons)
```

这里涉及三个基本概念：
- T.Augmentation定义了需改输入的**政策**
  - 它的`__call__(AugInput) -> Transform`通过in-place的方式增强数据，并且返回应用的操作
- T.Transform实现了真正的变形数据的操作
  - 它有例如`apply_image`和`apply_coords`的方法来定义如何为每一种数据类型变形
- T.AugInput存储了`T.Augmentation`需要的输入以及它们应该如何变形。一些高阶用法需要这个概念。如上所示，由于没在`T.AugInput`中的额外数据可以通过使用返回的`transform`增强，所以直接使用这个类对于所有常用情形应该是足够的。

## 写新的增强

大部分二维增强只需要知道数据图像。这样的增强可以如下轻易实现：

```python
class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        r = np.random.rand(2)
        return T.ColorTransform(lambda x: x * r[0] + r[1] * 10)

class MyCustonResize(T.Augmentation):
    def get_transform(self.image):
        old_h, old_w = image.shape[:2]
        new_h, new_w = int(old_h * np.random.rand()), int(old_w * 1.5)
        return T.ResizeTransform(old_h, old_w, new_h, new_w)

augs = MyCustomResize()
transform = augs(input)
```

除了图片，只要他们是函数签名的一部分，给定`AugInput`的任意属性都可以被使用，例如：

```python
class MyCustomCrop(T.Augmentation):
    def get_transform(self, image, sem_seg):
        # decide where to crop using both image and sem_seg
        return T.CropTransform(...)

augs = MyCustomCrop()
assert hasattr(input, "image") and hasattr(input, "sem_seg")
transform = augs(input)
```

新的变形操作也可以通过继承[T.Transform](https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html#detectron2.data.transforms.Transform)来添加

## 高级用法

我们提供了一些我们的系统使能的高级用法的例子。虽然对于标准使用的情形来说修改它们是不必要的，但是这些选项可能对于新研究来说很有趣。

### 自定义变形策略

detectron2的`Augmentation`并不是仅仅返回增强的数据，而是返回`T.Transform`形式的**操作**。这允许用户对数据使用自定义的变形策略。我们使用关键点数据作为例子。

关键点是(x, y)坐标，但是由于它们携带的语义信息，所以增强它们并不容易。只有用户知道这样的含义，因此用户想根据返回的`transform`来手动增强它们。例如，当一个图片是水平翻转的，我们希望将关键点标注“左眼”和“右眼”对换。这可以通过如下代码实现（这被detectron2的默认数据加载器包含在内）:

```python
# augs和input在之前的例子中定义
transfom = augs(input)
keypoints_xy = transform.apply_coords(keypoints_xy)

# 得到一个所有的应用的transform的列表
transforms = T.TransformList([transform]).transforms
# 检查是否翻转了奇数次
do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms) % 2 == 1
if do_hflip:
    keypoints_xy = keypoints_xy[flip_indices_mapping]
```

# 使用模型

## 从Yacs配置构建模型

可以通过例如`build_model`、`build_backbone`和`build_roi_heads`函数从一个yacs配置对象构建模型:

```python
from detectron2.modeling import build_model
model = build_model(cfg) # 返回一个torch.nn.Module
```

`build_model`仅仅构建模型结构，其中的参数是随机数。如下将展示如何加载已有的checkpoint以及如何使用`model`对象。

### 加载/存储checkpoint

```python
from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(file_path_or_url) # 加载文件，通常通过cfg.MODEL.WRIGHTS

checkpointer = DetectionCheckpointer(model, save_dir="output")
checkpointer.save("model_999") # 存至output/model_999.pth
```

detectron2的checkpointer识别pytorch的`.pth`格式模型，以及我们的模型动物园中的`.pkl`文件。关于它的用法的更多细节见[API doc](https://detectron2.readthedocs.io/en/latest/modules/checkpoint.html#detectron2.checkpoint.DetectionCheckpointer)

模型文件可以通过`torch.{load, save}`（对于`.pth`文件）或`pickle.{dump, load}`（对于`.pkl`）随意操控

### 使用模型

可以通过`outputs = model(inputs)`来调用模型，其中`inputs`是`list[dict]`。

# Evaluation
评估是一个使用一定数量的输入/输出对并聚合它们的过程。你可以一直直接使用[use the model](https://detectron2.readthedocs.io/en/latest/tutorials/models.html)并手动分析它的输出/输出来进行评估。或者，使用detectron2中已经实现的[DatasetEvaluator](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)来进行评估

detectron2包含一些使用特定数据集API（例如COOC和LVIS）来计算指标的`DatasetEvaluator`。你也可以实现你自己的`DatasetEvaluator`来进行另外的一些使用输入/输出对的工作。例如，为验证集中检测到的实例进行计数：

```python
class Counter(DatasetEvaluator):
    def reset(self):
        self.count = 0
    def process(self, inputs, outputs):
        for output in outputs:
            self.count += len(output["instances"])
    def evaluate(self):
        return {"count": self.count}
```

## 使用评测器

手动使用评测器的方法进行评估：

```python
def get_all_inputs_outputs():
    for data in data_loader:
        yield data, model(data)

evaluator.reset()
for inputs, outputs in get_all_inputs_outputs():
    evaluator.process(inputs, outputs)
eval_results = evaluator.evaluate()
```

评测器也可以与[inference_on_dataset](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.inference_on_dataset)一同使用。例如：

```python
eval_results = inference_on_dataset(
    model,
    data_loader,
    DatasetEvaluators([COCOEvaluator(...), Counter()]))
```

这将会在来自`data_loader`的所有输入上执行`model`，并调用evaluator来执行它们。

相比于手动使用模型运行评测，这个函数的好处是评测器可以通过使用[DatasetEvaluators](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.DatasetEvaluators)合并起来，所有的评测可以通过遍历数据集的以此前向传播全部完成。这个函数也为给定的模型和参数提供了准确的高速基准。

## 对于自定义数据的的评测器

为了使用每个数据集特定的官方API，detectron2中的许多评测器被用来特定的数据集。除此之外，有两个评测器可以评测任意遵循detectron2[标准数据集格式](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)的任意普通数据集，所以它们可以被用来评测自定义数据集：

- [COCOEvaluator](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.COCOEvaluator)可以在任意自定义数据集上为边界框检测、实例分割和关键点检测评估AP（平均精度）
- [SemSegEvaluator](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.SemSegEvaluator)可以在任意自定义数据集上评估语义分割指标

# Yacs Configs

detectron2提供了基于键值的配置系统，它可以被用来获得标准、公共的行为

这个系统使用YAML和[yacs](https://github.com/rbgirshick/yacs)。Yaml是一个非常局限的语言，所以我们并不期待通过配置实现detectron2中的所有特性。如果你需要一些在配置空间中不可用的东西，请使用detectron2的API来写代码。

协同介绍一个更加有能力的[LazyConfig system](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html)，我们不再向基于Yacs/Yaml的配置系统添加功能或者新键。

## 基本用法

一些`CfgNode`对象的基本用法展示如下。可以在[documentation](https://detectron2.readthedocs.io/en/latest/modules/config.html#detectron2.config.CfgNode)中查看更多用法。

```python
from detectron2.config import get_config
cfg = get_cfg() # 得到detectron2的默认配置
cfg.xxx = yyy # 为你自己的自定义组件添加新的配置
cfg.merge_from_file("my_cfg.yaml") # 从一个文件中载入值

cfg.merge_from_list(["MODEL.WEIGHTS", "weights.pth"]) # 也可以从字符串列表载入值
print(cfg.dump()) # 打印整齐的配置
with open("output.yaml", "w") as f:
    f.write(cfg.dump())
```

除了基本的Yaml用法，配置文件可以定义一个`_BASE_: base.yaml`的域，这将会先载入一个基础配置文件。如果基础配置和子配置有冲突，基础配置中的值将会被子配置中的覆盖。我们为标注模型架构提供了一系列基础配置。

许多detectron2中的自建工具接受命令行配置覆盖：命令行中提供的键值对将会覆盖配置文件中的已有值。例如，[demo.py](https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py)可以遮掩使用：

```bash
./demo.py --config-file config.yaml [--other-options] --opts MODEL.WEIGHTS /path/to/weights INPUT.MIN_SIZE_TEST 1000
```

可以查看[Config Reference](https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references)来查看detectron2中可用配置的列表以及它们的含义。

## 项目中的配置

一个在detectron2库之外的项目可能会定义它自己的配置，为了项目的实用，这将需要被添加进去，例如：
```python
from detectron2.projects.point_rend import add_pointrend_config
cfg = get_cfg() # 得到detectron2的默认配置
add_pointrend_config(cfg) # 添加pointrend的默认配置
# ... ...
```

## Configs的最佳实践

- 将你写得配置视为“代码”：避免复制或者重复它们；实用`_BASE_`来共享配置间的公共部分
- 保持你写的配置简单：不要包含不影响实验设置的键