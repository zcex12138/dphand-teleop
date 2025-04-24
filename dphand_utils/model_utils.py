import mujoco

def get_child_bodies_ids(model, parent_body_id):
    """获取子body"""
    parent_body_id = model.body(parent_body_id).id
    return [body_id for body_id in range(model.nbody) 
            if model.body_parentid[body_id] == parent_body_id]

def get_all_child_bodies_ids(model, parent_body_id):
    """递归获取所有子 body"""
    children = [parent_body_id]
    for child_name in get_child_bodies_ids(model, parent_body_id):
        children += get_all_child_bodies_ids(model, child_name)
    return children

def get_child_bodies_names(model, body_name):
    """获取子 body 名称列表"""
    parent_body_id = model.body(body_name).id
    child_body_ids = get_child_bodies_ids(model, parent_body_id)
    return [model.body(body_id).name for body_id in child_body_ids]

def get_all_child_bodies_names(model, body_name):
    """递归获取所有子 body 名称"""
    parent_body_id = model.body(body_name).id
    child_body_ids = get_all_child_bodies_ids(model, parent_body_id)
    return [model.body(body_id).name for body_id in child_body_ids]

def get_all_joint_ids(model, body_name):
    """获取所有关节 ID 列表"""
    child_body_ids = get_all_child_bodies_ids(model, body_name)
    return [jnt_id for jnt_id in range(model.njnt) if model.jnt_bodyid[jnt_id] in child_body_ids]

def get_actuator_joint_ids(model):
    """返回执行器对应的关节 ID 列表，按执行器顺序排列"""
    joint_ids = []
    for i in range(model.nu):
        # 检查传输类型是否为关节
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            # 获取关节 ID（actuator_trnid[i][0] 是第一个关联的关节）
            joint_id = model.actuator_trnid[i][0]
            joint_ids.append(joint_id)
        else:
            # 非关节类型（如肌腱或体自由度）
            joint_ids.append()  # 用 -1 表示无效
    return joint_ids

def get_joint_id_from_actuator(model, actuator_name):
    # 获取执行器 ID
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
    )
    if actuator_id == -1:
        raise ValueError(f"执行器 '{actuator_name}' 不存在")

    # 验证传输类型是否为关节
    trn_type = model.actuator_trntype[actuator_id]
    if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
        raise TypeError(f"执行器 '{actuator_name}' 未驱动关节")

    # 获取关节 ID
    joint_id = model.actuator_trnid[actuator_id][0]
    return joint_id

def is_strictly_ascending(arr):
    for i in range(1, len(arr)):
        if arr[i] <= arr[i-1]:
            return False
    return True