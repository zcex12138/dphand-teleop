import numpy as np

# 加载原始数据
data = np.load('./data/test_recording.npz', allow_pickle=True)

print(data['data'].size)  # 打印原始数据的键名以供检查
exit()

# 重命名键（假设旧键是 'keypoints'，新键是 'new_key'）
old_key = 'keypoints'
new_key = 'data'  # 替换为你需要的新键名

if old_key in data:
    data[new_key] = data.pop(old_key)  # 修改键名并保留数据
else:
    print(f"键 '{old_key}' 不存在，请检查名称！")

# 保存修改后的数据（使用压缩格式节省空间）
np.savez_compressed('./data/hand_keypoints_renamed.npz', **data)