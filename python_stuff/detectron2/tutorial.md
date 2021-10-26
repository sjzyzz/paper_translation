





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