# labelme---matched
IoU-based JSON Annotation Matching and Grouping
# 先分别加载两个 JSON 文件中的数据，然后按照标签（face、facemask、head、headmask）对形状进行分组。对于每个标签组，遍历原始形状和新形状，计算它们之间的 IoU，并根据设定的阈值（t=0.5）来判断是否匹配。如果匹配，则为匹配的形状添加 group_id 属性，并从原始形状和新形状列表中删除已匹配的形状；如果无匹配，则为误判原始形状添加 group_id 为 58（谐音），为漏判新形状添加 group_id 为 68（谐音）。
