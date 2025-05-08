import json


def update_json_with_descriptions(drone_txt_file_path, sate_txt_file_path, json_file_path, output_json_path):
    # 读取 txt 文件并解析描述
    drone_descriptions = {}
    with open(drone_txt_file_path, 'r') as txt_file:
        for line in txt_file:
            parts = line.split(".png, ")
            # print(parts)
            if len(parts) == 2:
                # print(parts)
                image_name = parts[0].strip()
                description = parts[1].strip().rstrip("]\n").replace("'", "")
                drone_descriptions[image_name] = description
    
    sate_descriptions = {}
    with open(sate_txt_file_path, 'r') as txt_file:
        for line in txt_file:
            parts = line.split(".png, ")
            # print(parts)
            if len(parts) == 2:
                # print(parts)
                image_name = parts[0].strip()
                description = parts[1].strip().rstrip("]\n").replace("'", "")
                sate_descriptions[image_name] = description

    print(sate_descriptions.keys())

    # print(descriptions.keys())

    # 读取 json 文件
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # 更新 json 数据
    for item in json_data:
        drone_img_name = item.get("drone_img_name").replace('.png', '')
        # print(drone_img_name)
        # item["drone_img_desc"] = ""
        if drone_img_name in drone_descriptions.keys():
            # print(drone_img_name)
            item["drone_img_desc"] = drone_descriptions[drone_img_name]
        else:
            item["drone_img_desc"] = ""
    
        pos_sate_img_list = item.get("pair_pos_sate_img_list")
        item["pair_pos_sate_img_desc_list"] = []
        for sate_img in pos_sate_img_list:
            sate_img_name = sate_img.replace(".png", "")
            if sate_img_name in sate_descriptions.keys():
                item["pair_pos_sate_img_desc_list"].append(sate_descriptions[sate_img_name])
            else:
                item["pair_pos_sate_img_desc_list"].append("")
        
        pos_semi_pos_sate_img_list = item.get("pair_pos_semipos_sate_img_list")
        item["pair_pos_semipos_sate_img_desc_list"] = []
        for sate_img in pos_semi_pos_sate_img_list:
            sate_img_name = sate_img.replace(".png", "")
            if sate_img_name in sate_descriptions.keys():
                item["pair_pos_semipos_sate_img_desc_list"].append(sate_descriptions[sate_img_name])
            else:
                item["pair_pos_semipos_sate_img_desc_list"].append("")

    # 写入更新后的 json 文件
    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON 文件已更新并保存到 {output_json_path}")


if __name__ == '__main__':
    drone_txt_file_path = "/home/xmuairmud/jyx/daily_scripts/GTA-UAV-MM-drone-description-gpt4o.txt"
    sate_txt_file_path = "/home/xmuairmud/jyx/daily_scripts/GTA-UAV-MM-satellite-description-gpt4o.txt"
    json_file_path = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-MM/" + "same-area-drone2sate-train-123.json"
    output_json_path = "/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-MM/" + "same-area-drone2sate-train-textallgpt-123.json"

    # 执行更新
    update_json_with_descriptions(drone_txt_file_path, sate_txt_file_path, json_file_path, output_json_path)