# Agent 纯文本试验抽样检查

快照时间：2026-10-24T03:38:38.169759+00:00。

人工阅读原始文本的探索性检查，不是盲评，不计算内容 0–2 分，不修改解析或预测。字段齐全不等于 instruction 语义完整、时间命中或工具执行成功。

| 模型 | 已完成 QA | 双字段合规 QA | 无回复 QA |
|---|---:|---:|---:|
| minicpmo | 100 | 12 | 13 |
| gander | 100 | 1 | 72 |
| joyai | 55 | 53 | 0 |

JoyAI 仍在运行，以上是快照，三者分母不同，不据此做最终排名。

## Sample 8 / action_perception

If someone sitting on a bed gets up and heads for a door, save a note with the time and what happened so I can find this scene later.

参考窗口：00:12–00:15

MiniCPM：格式与回复时间均命中参考窗口，但回复声称 Noted、时间写成 00:15，仍需区分记录意图与已执行。JoyAI：1 秒便宣称观察到目标动作，而参考窗口为 12–15 秒。Gander：58 秒才回复，且没有 instruction。

### minicpmo

14.000s / valid
```text
Noted the subject leaving the bed area at 00:15 for the door<|task_start|>note_write title:"Subject exits room" content:"At 00:15, person leftbed and approached door"
```

### gander

58.000s / absent
```text
Detected: "Someone sitting on a bed gets up and heads for a door" has started.
```

### joyai

1.000s / valid
```text
I've noted that someone stood up from the bed and left toward the door around 00:00; I'll save this observation as a note. <|task_start|>note_write title="Bedtime Observation" content="At approximately 00:00, a person sat on the bed, got up, and walked toward the door."
```

## Sample 39 / emotion_recognition

Whenever a Como player celebrates a goal with joyous excitement, send a message to my soccer fan channel on Discord sharing who scored and the updated match score.

参考窗口：01:24–01:28, 01:54–01:58

JoyAI 自行补出了 @SoccerFanClub，当前用户请求和本轮有效 prompt 没有提供该频道标识；属于未经支持的参数。

### minicpmo

66.000s / absent
```text
The user wants notifications sent to their Discord channel when Como scores.
```

91.000s / absent
```text
Como scored; sending update to soccer fan channel now.
```

### gander

未生成回复。

### joyai

66.120s / valid
```text
I see a Como player celebrating enthusiastically after scoring; I will notify your soccer fan channel on Discord about the goal scorer and the new match score. <|task_start|>Send a message to @SoccerFanClub on Discord stating that a Como player scored, updating the score to FIO 1-1 COMO.
```

## Sample 133 / action_perception

Whenever someone starts eating a cookie while sitting at the counter, look up Brooki cookies online to check where the bakery is located and their signature menu items.

参考窗口：00:22–00:26

MiniCPM 实际生成 <| task_start|>，多了一个空格；有委托意图，但不符合本轮精确分隔符约定，保留原判定，不自动修复。

### minicpmo

11.000s / absent
```text
 The person is eating a cookie at the counter; looking up Brooki cookies now. <| task_start|>web_search: Brooki cookies location and signature items.
```

### gander

未生成回复。

### joyai

11.000s / valid
```text
I see someone started eating a cookie at the counter; I will look up Brooki cookies details now.<|task_start|>web_search: "Brooki cookies bakery location and signature menu items"
```

## Sample 141 / object_perception

Whenever a new bathroom fixture appears on screen, look up pricing and specifications for that type of faucet and write the details to a note titled 'Bathroom Remodel'.

参考窗口：00:32–00:36

用户要求查询价格/规格并写入 Bathroom Remodel 笔记；JoyAI instruction 只有 product_search，遗漏写笔记。

### minicpmo

42.000s / absent
```text
 Faucet appears; I'll look up its specs and pricing for your note.
```

### gander

未生成回复。

### joyai

23.022s / valid
```text
A bathroom sink faucet just appeared; I will check its specs and price.<|task_start|>product_search(query="bathroom sink faucet specifications and price")
```

## Sample 458 / action_counting

Keep track of how many commercial brand names the rapper mentions starting from here. Once he reaches two brands, look up online when that second brand was founded and what it sells.

参考窗口：00:44–00:47

MiniCPM 双字段形式存在，但 instruction 仅为 web_search，没有品牌名称、查询问题或任务范围，不能视为有用的完整委托。

### minicpmo

12.000s / valid
```text
 Tracking brand mentions; will count once two are identified.<|task_start|>web_search
```

136.800s / absent
```text
 Two brands mentioned: Microsoft and Old Navy. web_search for founding and product info.
```

### gander

未生成回复。

### joyai

11.000s / valid
```text
The rapper just mentioned his first brand; I'll wait for the second before looking it up.<|task_start|>Search web for the founding year and product category of the second brand name the rapper mentions.
```

## Sample 2064 / action_perception

Once the space shuttle finishes its roll maneuver, look up online which astronaut crew members are being delivered to the International Space Station on this mission.

参考窗口：00:20–00:25

Gander 唯一完整双字段例：内容基本表达查询宇航员名单的意图，但回复为 28 秒，晚于 20–25 秒窗口，且 instruction 使用 this mission，未给出独立任务标识。

### minicpmo

11.000s / absent
```text
 The shuttle's launch is complete; I'll identify the crew heading to the ISS.<| task_start|>web_search: space shuttle discovery mission crew international space station 20 00s.
```

106.000s / absent
```text
 The roll maneuver has finished; I'll look up which astronauts are on board for the ISS mission.
```

137.000s / absent
```text
 The solid rocket boosters have separated; I'll now search for the crew roster.
```

137.724s / absent
```text
 web_search: space shuttle discovery mission astronaut crew list international space station.
```

### gander

28.000s / valid
```text
The shuttle's roll maneuver has just finished. I'll check online for the crew delivering astronauts to the International Space Station. <|task_start|>search for crew members delivered to ISS by this mission shuttle launch
```

### joyai

快照时尚未完成。
