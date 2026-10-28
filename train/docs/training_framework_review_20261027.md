# 流式 SFT 框架复用与加速方案

本次保留原生逐秒媒体、问题后沉默监督、连续全部response和原子task_start，不将任务转换成整段视频问答。

| 检查对象 | 源码观察 | 本次决定 |
|---|---|---|
| OpenBMB finetune | 官方README列出Trainer+DeepSpeed、图像/对话数据和较早模型；未提供本仓库4.5双工监督的直接模板 | 可参考基础训练设施，不直接替换数据契约 |
| MS-Swift | 明确注册MiniCPM-o-4_5及专用模板；模板组织image/audio bounds、媒体块和常规对话 | 支持模型不等于现成支持本项目逐秒silent/speak标签，迁移仍需自定义模板/前向 |
| LLaMA-Factory | 注册MiniCPM-o，插件处理图像/视频/音频占位符与bounds | 已检查插件未发现本项目流式监督入口；直接迁移不能消除原生逐秒编码成本 |
| DeepSpeed | 已安装且现有工程已使用 | 复用其ZeRO-2原生数据并行，让各rank输入不同QA；保留自定义目标层 |

来源：

- [OpenBMB官方微调说明](https://github.com/OpenBMB/MiniCPM-V/blob/main/finetune/readme.md)
- [MS-Swift模型注册](https://github.com/modelscope/ms-swift/blob/main/swift/model/models/minicpm.py)
- [MS-Swift模板实现](https://github.com/modelscope/ms-swift/blob/main/swift/template/templates/minicpm.py)
- [LLaMA-Factory媒体插件](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/mm_plugin.py)
- [DeepSpeed ZeRO文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)

## 已实现的边界

1. 独立QA SFT限定ZeRO-2、累积1，旧ZeRO-3/GRPO默认行为保留。
2. 单次完整可微前向；长度和完整轨迹检查仍存在，无效QA单独剔除，不丢掉同批其他有效QA。
3. 局部长度分桶；GPU保存张量预算+CPU卸载；预算与剩余显存显式记录在配置。
4. 六卡有效batch=6，学习率仍1e-6、全部数据2轮；更新次数改变，不能把吞吐比较当成相同优化轨迹。
5. 两卡微型模型通过不同长度QA、有效/空槽两类比较；与单进程平均梯度后的权重最大差异约1.6e-8。真实模型profile与运行数据另记录，未测量前不承诺倍数。
6. 新版支持STOP文件在步边界保存；旧运行没有这个处理器，也没有周期checkpoint，不能凭修改文件从旧进程提取优化器状态。

## 四卡真实模型首次更新验证

当前运行0–3训练、6–7 benchmark，不使用4、5做本项目计算。独立QA有效batch=4，全部训练数据2轮共1998次优化更新；旧run42步的日志/W&B保留，新run因无旧checkpoint从基座开始。

[首批完整证据](dp4_first_update_20261027.json)：四个rank分别训练QA2321、106、392、2000，媒体轨迹约244–290秒，合计7个response。全局平均loss=4.949970，完整更新约208–213秒（约53秒/QA），峰值分配显存26.4–28.8 GiB/rank，激活GPU保留4 GiB/rank，其余CPU卸载。W&B新run为m74nub8n，并已核验服务端loss。

这只是一个偏长首批的吞吐，不是与旧模式同样本、同batch、同硬件负载的严格速度比较，暂不据此承诺固定加速倍数或全量ETA。
