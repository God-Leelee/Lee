# 演示预期输出

配置v1.8-research-preview-003。三个样例由固定程序在本机生成，无第三方原图、无AI真假真值。核验要求为真正完成计算、原件SHA一致、必需分支不失败、掩码尺寸与PTS正确、报告可读。风险高低不用于评估精度。

| 输入 | 参考风险 | 参考判定 | 完成情况 |
|---|---|---|---|
| sample01.png | 0.987071 | suspected_fake | 完整 |
| sample02.png | 0.970035 | suspected_fake | 完整 |
| sample03.mp4 | 0.806345 | suspected_fake | 完整 |

新请求case_id、创建时间和耗时会变化，不能逐字比较整份JSON。比较输入SHA、版本、分支状态、原图坐标、实际PTS和判定。其他GPU可有少量浮点差异，参考机器上的同版本复算应与上述分数接近。具体逐份记录见examples/expected_demo/。
