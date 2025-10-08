这次dbug真的很痛苦，记录一下我做了那些改动
服务器平台A100，Linux
官方给的pull用不了，尝试自己创环境遇到了严重的版本冲突问题，主要是pandas，numpy，sklearn，Minkowski
最后没有办法，只有找了系统里有什么就用什么，找到了：apptainer 
用 module load apps/apptainer/1.3.4 
load了apptainer以后勉强pull了环境。然后在一些apptainer内的小的升级之后开始了尝试

我的目的是先试试这个模型效果怎么样，所以就切了一小块想用他训练好的模型试试实例分割。所以没有label也没有训练过程种产生的一些参数，这些参数都需要手动赋值


Hydra的问题，很容易遇到一个bug
apptainer exec --nv \ > --bind /mnt/data/project0065/Zhimeng_Data/segmentanytree:/mnt/data/project0065/Zhimeng_Data/segmentanytree \ > /mnt/data/project0065/Zhimeng_Data/segmentanytree/sif/segment-any-tree_latest.sif \ > python3 /mnt/data/project0065/Zhimeng_Data/segmentanytree/SegmentAnyTree/eval.py \ > model_name=PointGroup-PAPER \ > ++training.checkpoint_dir=/mnt/data/project0065/Zhimeng_Data/segmentanytree/SegmentAnyTree/model_file \ > ++tracker_options.ply_output=myrun_output_fast.ply \ > ++data.fold=[/mnt/data/project0065/Zhimeng_Data/segmentanytree/output/utm2local/NS5462_test_out.ply] \ > ++training.cuda=0 \ > ++training.num_workers=4 \ > ++pretty_print=true \ > ++logging.progress_refresh_rate=10 \ > ++logging.wandb_dryrun=true /usr/local/lib/python3.8/dist-packages/wandb/__init__.py:998: SyntaxWarning: "is not" with a literal. Did you mean "!="? if name is None and resume is not "must": /usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.7) or chardet (5.2.0)/charset_normalizer (None) doesn't match a supported version! warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported " /usr/local/lib/python3.8/dist-packages/MinkowskiEngine/__init__.py:36: UserWarning: The environment variable OMP_NUM_THREADS not set. MinkowskiEngine will automatically set OMP_NUM_THREADS=16. If you want to set OMP_NUM_THREADS manually, please export it on the command line before running a python script. e.g. export OMP_NUM_THREADS=12; python your_program.py. It is recommended to set it below 24. warnings.warn( Error executing job with overrides: ['model_name=PointGroup-PAPER', '++training.checkpoint_dir=/mnt/data/project0065/Zhimeng_Data/segmentanytree/SegmentAnyTree/model_file', '++tracker_options.ply_output=myrun_output_fast.ply', '++data.fold=[/mnt/data/project0065/Zhimeng_Data/segmentanytree/output/utm2local/NS5462_test_out.ply]', '++training.cuda=0', '++training.num_workers=4', '++pretty_print=true', '++logging.progress_refresh_rate=10', '++logging.wandb_dryrun=true'] Traceback (most recent call last): File "/mnt/data/project0065/Zhimeng_Data/segmentanytree/SegmentAnyTree/eval.py", line 31, in run main(cfg) File "/mnt/data/project0065/Zhimeng_Data/segmentanytree/SegmentAnyTree/eval.py", line 12, in main cfg.hydra.searchpath[0].replace("file://", ""), # Hydra searchpath 第一条 omegaconf.errors.ConfigAttributeError: Missing key hydra full_key: hydra object_type=dict

这个我通过给hydra上绝对路径解决了
#####################/SegmentAnyTree/eval.py
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_points3d.trainer import Trainer

# 🔧 使用绝对路径来指定配置文件目录
CONF_PATH = "/mnt/data/project0065/Zhimeng_Data/segmentanytree/SegmentAnyTree/conf"

@hydra.main(config_path=CONF_PATH, config_name="eval", version_base=None)
def main(cfg: DictConfig):
    print("\n[INFO] ✅ Hydra configuration loaded successfully!")
    OmegaConf.set_struct(cfg, False)

    print("[INFO] Model name:", cfg.get("model_name", "<missing>"))

    # 直接加载模型配置（避免 hydra.searchpath 问题）
    model_cfg_path = os.path.join(CONF_PATH, "models", f"{cfg.model_name}.yaml")
    if os.path.exists(model_cfg_path):
        print(f"[INFO] Loading model config from {model_cfg_path}")
        model_cfg = OmegaConf.load(model_cfg_path)
        cfg.models = {cfg.model_name: model_cfg}
    else:
        raise FileNotFoundError(f"❌ Model config not found: {model_cfg_path}")

    # 初始化 Trainer 并开始测试
    trainer = Trainer(cfg)
    trainer.eval(stage_name="test")


if __name__ == "__main__":
    main()

###################/segmentanytree/SegmentAnyTree/conf/eval.yaml
defaults:
  - models: PointGroup-PAPER
  - _self_

model_name: PointGroup-PAPER

data:
  fold:
    - /mnt/data/project0065/Zhimeng_Data/segmentanytree/output/utm2local/NS5462_test_out.ply

tracker_options:
  full_res: true
  make_submission: true
  ply_output: myrun_output_fast.ply

training:
  enable_cudnn: true
  batch_size: 1
  num_workers: 0
  cuda: 0
  shuffle: false
  checkpoint_dir: /mnt/data/project0065/Zhimeng_Data/segmentanytree/SegmentAnyTree/model_file
  weight_name: PointGroup-PAPER

pretty_print: true
debugging:
  profiling: false



