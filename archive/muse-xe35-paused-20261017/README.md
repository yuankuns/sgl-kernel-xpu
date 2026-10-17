# Muse Xe35 临时代码存档（2026-10-17）

用户已停止全部工作。本分支只保存代码改动及恢复所需的基线/SHA信息，不保存测试结果、日志、tensor/raw、core、native binaries、安装包副本或构建产物。原机约710GB临时验证目录已按用户要求删除；这些数据未上传，后续复现需重建环境并重跑，不能从此分支恢复原始结果。本存档不代表验收通过，不开或更新正式PR。

远端仓库：`yuankuns/sgl-kernel-xpu`。
远端分支：`main`。
远端存档目录：`archive/muse-xe35-paused-20261017`，不替换main现有kernel实现。
存档基线：`833c95208dccd2af5628c121222f91186a5b2de7`。

## 代码布局

- `repositories/upstream-pr/`：11个PR工作树的未提交代码补丁，包含kernel、headers、Python、benchmark和tests。
- `repositories/personal-source/`：原优化工作树的未提交代码。个人根checkout原先已有的改动单独保存，不混称为本轮移植。
- `temporary-code/`：临时验证、对照、诊断、构建、调度脚本和候选代码补丁，保持artifact目录层级；这是代码而不是运行结果。
- `CODE_MANIFEST.json`：补丁基线、SHA、纳入路径与应用检查，临时代码SHA以及读取错误。
- 原优化分支HEAD仅记入metadata，不额外发布原仓库的完整分支历史；恢复时从原工作树或有权访问的原仓库取得对应基线。

## 以后恢复

1. 获取 `yuankuns/sgl-kernel-xpu` 的main并读取此存档；根据metadata从原工作树或有权访问的原仓库取得所需基线commit。
2. 读取目标模块 `metadata.json`，在其HEAD新建干净个人开发工作树，先 `git apply --check <working-tree.patch>` 再应用补丁；不要直接覆盖当前脏工作树。
3. 脚本保留原机器 `/home/syk`、容器 `/root` 绝对路径，并依赖未归档的数据、正常包或状态文件；恢复时需另行配置，不能仅凭代码存档宣称完整实验可复现。
4. 全部工作保持暂停，旧授权不复用，后续GPU工作需用户重新授权。C++/SYCL/host/link用icpx，C用icx，使用正常包入口。
5. Muse生产仅B8、ISL4096、OSL16、TP1/EP1、原精度。0–7可精度/性能，8–11仅精度。不得根据早期卡性能改kernel或放宽精度门槛。

未解决事项：W4大形状性能回退、FM/W4 graph崩溃、QK/vocab mask graph输出错误、permutation graph实例化失败、Residual环境脚本错误及部分未完成验证。存档可能包含未验证候选代码，不作为已验收版本合并。

官方main、原工作树及原index不改变。`archive_code.py`仅归档代码，不执行测试或GPU任务。
