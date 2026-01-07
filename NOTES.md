

## AllReduce Calls per step:


#### GPT2
```bash
dpa: new-task allreduce.1120 [7087872 x f32] 0x7fca98000000 > 0x7fca98000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1121 [7087872 x f32] 0x7fca96000000 > 0x7fca96000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1122 [7087872 x f32] 0x7fca94000000 > 0x7fca94000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1123 [7087872 x f32] 0x7fca92000000 > 0x7fca92000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1124 [7087872 x f32] 0x7fca90000000 > 0x7fca90000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1125 [7087872 x f32] 0x7fca8e000000 > 0x7fca8e000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1126 [7087872 x f32] 0x7fca8c000000 > 0x7fca8c000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1127 [7087872 x f32] 0x7fca8a000000 > 0x7fca8a000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1128 [7087872 x f32] 0x7fca88000000 > 0x7fca88000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1129 [7087872 x f32] 0x7fca86000000 > 0x7fca86000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1130 [7087872 x f32] 0x7fca84000000 > 0x7fca84000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1131 [7087872 x f32] 0x7fca82000000 > 0x7fca82000000 4-pipe average quant.1 no-straggle 
dpa: new-task allreduce.1132 [39385344 x f32] 0x7fca72000000 > 0x7fca72000000 4-pipe average quant.1 no-straggle 
```


### Switch timeout profile

|  Timeout (ms)  | tsc | 100 batch        | 100               | 1                |
|----------------|-----|------------------|-------------------|------------------|
|  0.05 (0.065)  | 1   | 58800834 / 23766 | 58709251 / 115349 | 585589 / 2657    |
|  0.1  (0.13)   | 2   | 58808118 / 16482 | 58789741 / 34859  | 586780 / 1466    |
|  0.2  (0.26)   | 4   | 58814431 / 10169 | 58822561 / 2039 * | 586648 / 1598    |
|  0.3  (0.32)   | 5   | 58813042 / 11558 | 58823413 / 1187   | 587100 / 1146    |
|  0.5  (0.52)   | 8   | 58815840 / 8760  | 58823452 / 1148   | 587670 / 576 *** |    <---
|  1.0  (1.04)   | 16  | 58818347 / 6253  | 58824600 / 0      | 588246 / 0       |
|  1.5  (1.50)   | 23  | | | 588246 / 0       |


1.0 / 0.250  (4-5) -->  62Gb
1.5 / 0.250  (6-7) -->  

* sometimes, not all pipelines agree on timeout
** sometimes timeouts are (relatively) much higher (~ 2x)
*** sometimes timeouts can be both 2x higher or lower 