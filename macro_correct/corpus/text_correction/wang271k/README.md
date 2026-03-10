# csc_eval_public
## 一、测评数据说明
### 1.1 数据清洗
```
余-馀: 替换为馀-余
other - 馀: 替换为余
覆-复: 替换为复-覆
other-覆: # 答疆/回覆/反覆
          # 覆审
他-她:不纠
她-他:不纠
人名不纠: 识别人名并丢弃
的得地: 建议丢弃(标注得不准)

# # 的 - 地
# # 的 - 得
# # 它 - 他
# # 哪 - 那

# # 改-大小改:  余-馀 覆-复 借-藉 功-工 琅-瑯 震-振 百-白 也-叶 经-禁(经不起-禁不起)
# # 部分不变(人名)： 小-晓 一-逸 佳-家 得-地(马哈得) 红-虹 民-明
# # 匹配上但是不改的： 惟-唯 象-像 查-察 立-利 止-只 建-健 他-它 地-的 定-订 带-戴 力-利 成-城 点-店
# # 匹配上但是不改的： 作-做 得-的 场-厂 身-生 有-由 种-重 理-里


# # 空白没匹配上： 今-在 年-今 前-目 当-在 目-在 者-是
# # 外国人名等：其-齐 课-科 博-波
```

### 1.2 代码
```
a00_csc_clean_public_wang271k.py
```

### 1.3 数据来源
 - [shibing624/CSC](https://huggingface.co/datasets/shibing624/CSC)
```注意: 
test.json 和 dev.json 为 SIGHAN数据集， 包括SIGHAN13 14 15，来自 官方csc.html ，文件大小：339kb，4千条。
train.json 为 Wang271k数据集，包括 Wang271k ，来自 Automatic-Corpus-Generation dimmywang提供 ，文件大小：93MB，27万条。
```

## 二、论文
### 初始来源
 - [https://github.com/wdimmy/Automatic-Corpus-Generation](https://github.com/wdimmy/Automatic-Corpus-Generation)

### 论文
 - [A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Checking](https://aclanthology.org/D18-1273)
