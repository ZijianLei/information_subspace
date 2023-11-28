## Expereiment
1 单纯对应 larger fisher information,一些intrinsic dimension中的参数没有信息的汇入，所以导致没有直接选取first n 个效果好(partly need more experiment)
2 PAC-BAYESIAN INFORMATION BOTTLENECK AND SUBSPACE TRADE OFF
3 a100 加载问题： 因为cpp_extension 卡住，删除对应cache 文件
4 conda的srv04
5 select_direction_rate==0 时 v100和a100结果差异大
6 fisher 是针对parameter的，同一个block 给到 lid的fisher是累积计算出来的

2.16:
1 global 和 layer-wise 选取的mask 总数不一致
2 load best model at the end 没有选取最优模型

2.17 
1 原先基于fisher选取的方向和后续retrain的方向实际不一致=> 每层的所有block share 相同的一个mask(mask 是based on layer idx 而不是based on name of a weight matrix)=> code fisher requires evaluation
2 select bast model at end not work => remain to solve

2.22
1 模型重新初始化之后需要重新初始化优化器
2 todo: 如何有效的只储存相关的id，目前整个id空间都会优化，虽然对结果不影响？（不会对forward变化，因为有mask，但是是否影响grad）

3.10
1 同层之间用不同的random matrix，不同层共享同一套random matrix

dict_keys(['.attention.self.query.weight', '.attention.self.query.bias', '.attention.self.key.weight', '.attention.self.key.bias', '.attention.self.value.weight', '.attention.self.value.bias', '.attention.output.dense.weight', '.attention.output.dense.bias', '.attention.output.LayerNorm.weight', '.attention.output.LayerNorm.bias', '.intermediate.dense.weight', '.intermediate.dense.bias', '.output.dense.weight', '.output.dense.bias', '.output.LayerNorm.weight', '.output.LayerNorm.bias']) =>会减慢运行效率
2 不用hook rewrite wrapped model

3.15
1 dropout可以提升效果
2 目前dropout实现有误，改变dropout measurement 效果不变
3 是否因为parameter 太多，趋近于random dropout => 如何解决高维情况概率消失的问题 =>通过temperature 来解决（似乎隐性的解决了explore的问题）
random seed 3333 72.92 77.62

运行时间和内存占用

需要优化的量：
FT 5.8 it/s 3.26G
SA 14 it/s  1.83G  (Gradient check point?)

5/17
which information measure and why

5/22
在训练的结果dict中加入epoch信息

6/11
同样important 的参数怎么去选择性的mask
因为torch.multinomial可以直接将不想被sample的参数设置为0    

7/17
原始paper 实际是优化到一个点，再通过bootstrape计算fisher 来得到当前的infomation值，并没有存下中间的grad value

对比：
1 在一定step mask 最为important 的那些parameter 来进行训练 \
$72.68_{\pm3.54 }$& $90.75_{\pm0.62 }
2 始终根据importance 来选择进行训练 
$76.17_{\pm2.62 }$&
74.37
temperature 5e-2
 $72.68_{\pm3.98 }$& $90.95_{\pm0.43 }$

w/o energy function
rte 
[0.779783393501805, 0.7472924187725631, 0.779783393501805]
lid 1 1e-2 32768 0.7689530685920577 0.015316392372271797
mrpc
[0.8946078431372549, 0.8872549019607843, 0.8946078431372549]
lid 1 1e-2 32768 0.892156862745098 0.0034662097116988024
stsb
[0.9009599882956774, 0.8998319539095931, 0.8980606147771886]
lid 1 1e-2 32768 0.899617518994153 0.001193336624664467

W energy function & w/o SGLD beta=1
rte
[0.7833935018050542, 0.7472924187725631, 0.740072202166065]
lid 1 1e-2 32768 0.756919374247894 0.01895068080387924
mrpc
[0.8848039215686274, 0.8897058823529411, 0.8946078431372549]
lid 1 1e-2 32768 0.8897058823529411 0.004002434220233982
stsb
[0.9012023570720321, 0.9028826590196948, 0.899752626074479]
lid 1 1e-2 32768 0.9012792140554019 0.001278985745673376

W energy function & w/o SGLD beta=0.5
rte
[0.7870036101083032, 0.7545126353790613, 0.740072202166065]
lid 1 1e-2 32768 0.7605294825511432 0.019626361528640304
mrpc
[0.8946078431372549, 0.9044117647058824, 0.8995098039215687]
lid 1 1e-2 32768 0.8995098039215687 0.004002434220233937
stsb
[0.9031391352725338, 0.9038919417550687, 0.8993674016009425]
lid 1 1e-2 32768 0.902132826209515 0.001979454360728448

W energy function & w/o SGLD beta=0.1


# RESULT RECORD
## adapter 1.43%
## parallel adapter 1.43%
rte
best_accuracy: 0.7436823104693141, std: 0.020633487604671798, learning rate 5e-4
mrpc
best_accuracy: 0.8905228758169934, std: 0.009023987759139934, learning rate 5e-4
stsb
best_combined_score: 0.9020785987526089, std: 0.0010052390571856356, learning rate 5e-4
cola
best_matthews_correlation: 0.5997683090090985, std: 0.01406210021579686, learning rate 5e-4
sst2
best_accuracy: 0.9407492354740062, std: 0.0010812030293372483, learning rate 5e-4
qnli
best_accuracy: 0.9189090243455977, std: 0.0018120986521346486, learning rate 5e-4
qqp
best_accuracy: 0.8990353697749196, std: 0.0005977267362648702, learning rate 5e-4

## Lora 0.23%
rte
best_accuracy: 0.7460890493381468, std: 0.014540368199271445, learning rate 5e-4
mrpc
best_accuracy: 0.8856209150326797, std: 0.0030569096297172614, learning rate 5e-4
stsb
best_combined_score: 0.9062845634776143, std: 0.0007136892652521789, learning rate 5e-4
cola
best_matthews_correlation: 0.6008978911259666, std: 0.00933926380016897, learning rate 5e-4
sst2
best_accuracy: 0.9342507645259938, std: 0.0019491664807311788, learning rate 5e-4
qnli
best_accuracy: 0.9129294038684485, std: 0.002167590416539337, learning rate 5e-4
qqp
best_accuracy: 0.8836342649847473, std: 0.001262220643980551, learning rate 5e-4

# Bitfit

## SAID

## finetune 100%

our 5.52 it/s 2.68G
said 
finetune 3.27159716796875 g

beta
rte 1111
0 74.73
1000 59.59
100 74.73
1 
1e-2 73.29