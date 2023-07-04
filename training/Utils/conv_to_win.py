def win_ops():
    data = ""

    with open("D:\\Repos\\LungTumorSegmentation\\Resources\\train_data_win.yaml", "r") as f:
        data = f.read()

    data_split = data.split("label")
    new_data = ""
    ds = []
    for i in data_split[3:]:
        ds.append(i.split(",")[0].split("\\")[2].split("'")[0])
    
    for n, i in enumerate(data_split[2:]):
        new_data += i
        if (n < 63):
            new_data += f"boxes' : '{ds[n]}',\n      'label"

    with open("D:\\Repos\\LungTumorSegmentation\\Resources\\train_data_win_with_boxes.yaml", "w+") as f:
        f.write(new_data)

def ub_ops():
    data = ""

    with open("D:\\Repos\\LungTumorSegmentation\\Resources\\train_data_win_with_boxes.yaml", "r") as f:
        data = f.read()

    data = data.replace("\\" + "\\", "/")

    with open("D:\\Repos\\LungTumorSegmentation\\Resources\\train_data_with_boxes.yaml", "w+") as f:
        f.write(data)
    

if __name__ == "__main__":
    ub_ops()
        
