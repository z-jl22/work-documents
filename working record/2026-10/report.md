### 一. aDSL实验报告
https://github.com/z-jl22/aDSL/blob/master/ADSL_CAPABILITY_REPORT.md
后续实验计划：
1. 目前对生成质量的分析局限在定性分析，后续可以调研一些可用的测试数据集或benchmark，定量分析aDSL生成质量
2. 继续调研agent+3D creation相关的方向
### 二. RawNorm工作总结
1. 整理dataset数据构建方法，包括L1/2/3数据如何分级、如何从点云得到view-GT、outlier的判定等；
2. 尝试实验后处理，探究使用错误点周围正确点进行投票的方式进行改错的可行性；
3. 跑baseline，包括shs-net、pff-net等方法在室内室外数据集上的训练和推理等；
4. 做消融实验，包括将V8部分输入通道置0进行推理、训练输出通道只有常数场没有一阶场的版本、将V8的注意力机制换成稀疏卷积、去掉训练数据的数据增广对比推理结果等。