import json


def update_json_with_descriptions(txt_file_path, json_file_path, output_json_path):
    # 读取 txt 文件并解析描述
    descriptions = {}
    with open(txt_file_path, 'r') as txt_file:
        for line in txt_file:
            parts = line.split(", [")
            if len(parts) == 2:
                image_name = parts[0].strip()
                description = parts[1].strip().rstrip("]\n").replace("'", "")
                descriptions[image_name] = description

    # 读取 json 文件
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # 更新 json 数据
    for item in json_data:
        drone_img_name = item.get("drone_img_name")
        if drone_img_name in descriptions:
            item["drone_img_desc"] = descriptions[drone_img_name]

    # 写入更新后的 json 文件
    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON 文件已更新并保存到 {output_json_path}")


if __name__ == '__main__':
    txt_file_path = "/home/xmuairmud/jyx/Qwen2-VL/GTA-UAV-Lidar-description.txt"
    json_file_path = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar/" + "cross-area-drone2sate-train-12.json"
    output_json_path = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar/" + "cross-area-drone2sate-train-textqw2b-12.json"

    # 执行更新
    update_json_with_descriptions(txt_file_path, json_file_path, output_json_path)