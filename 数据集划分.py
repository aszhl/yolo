# by CSDN 迪菲赫尔曼
import os
import random
import shutil
import glob
import xml.etree.ElementTree as ET

#xml地址
xml_file=r""

#需要转换的目标种类
l=['warning light','power light','AC light','running light','warning area','comen','test area','SV 300','blurred screen','black screen','white screen','aeonmed']

def convert(box,dw,dh):
    x=(box[0]+box[2])/2.0
    y=(box[1]+box[3])/2.0
    w=box[2]-box[0]
    h=box[3]-box[1]

    x=x/dw
    y=y/dh
    w=w/dw
    h=h/dh

    return x,y,w,h

def f(name_id,xml_file):
    xml_o=open(xml_file+'\%s.xml'%name_id)
    txt_o=open(r'F:\yolov8\ultralytics-main\ultralytics-main\data\data\label\{}.txt'.format(name_id),'w')

    pares=ET.parse(xml_o)
    root=pares.getroot()
    objects=root.findall('object')
    size=root.find('size')
    dw=int(size.find('width').text)
    dh=int(size.find('height').text)

    for obj in objects :
        c=l.index(obj.find('name').text)
        bnd=obj.find('bndbox')

        b=(float(bnd.find('xmin').text),float(bnd.find('ymin').text),
           float(bnd.find('xmax').text),float(bnd.find('ymax').text))

        x,y,w,h=convert(b,dw,dh)

        write_t="{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(c,x,y,w,h)
        txt_o.write(write_t)

    xml_o.close()
    txt_o.close()

def copy_files(src_dir, dst_dir, filenames, extension):
    os.makedirs(dst_dir, exist_ok=True)
    missing_files = 0
    for filename in filenames:
        src_path = os.path.join(src_dir, filename + extension)
        dst_path = os.path.join(dst_dir, filename + extension)

        # Check if the file exists before copying
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: File not found for {filename}")
            missing_files += 1

    return missing_files

def del_file(path):
    for i in os.listdir(path):
        file_data = path + "\\" + i
        os.remove(file_data)

def split_and_copy_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, valid_ratio=0.2, test_ratio=0):
    # 获取所有图像文件的文件名（不包括文件扩展名）
    image_filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]

    # 随机打乱文件名列表
    random.shuffle(image_filenames)

    # 计算训练集、验证集和测试集的数量
    total_count = len(image_filenames)
    train_count = int(total_count * train_ratio)
    valid_count = int(total_count * valid_ratio)
    test_count = total_count - train_count - valid_count

    # 定义输出文件夹路径
    train_image_dir = os.path.join(output_dir, output_dir+r'\train\images')
    train_label_dir = os.path.join(output_dir, output_dir+r'\train\labels')
    valid_image_dir = os.path.join(output_dir, output_dir+r'\valid\images')
    valid_label_dir = os.path.join(output_dir, output_dir+r'\valid\labels')
    test_image_dir  = os.path.join(output_dir, output_dir+r'\test\images')
    test_label_dir  = os.path.join(output_dir, output_dir+r'\test\labels')
    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    else:
        del_file(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)
    else:
        del_file(train_label_dir)

    if not os.path.exists(valid_image_dir):
        os.makedirs(valid_image_dir)
    else:
        del_file(valid_image_dir)
    if not os.path.exists(valid_label_dir):
        os.makedirs(valid_label_dir)
    else:
        del_file(valid_label_dir)

    if not os.path.exists(test_image_dir):
        os.makedirs(test_image_dir)
    else:
        del_file(test_image_dir)
    if not os.path.exists(test_label_dir):
        os.makedirs(test_label_dir)
    else:
        del_file(test_label_dir)

    # 复制图像和标签文件到对应的文件夹
    train_missing_files = copy_files(image_dir, train_image_dir, image_filenames[:train_count], '.jpg')
    train_missing_files += copy_files(label_dir, train_label_dir, image_filenames[:train_count], '.txt')

    valid_missing_files = copy_files(image_dir, valid_image_dir, image_filenames[train_count:train_count + valid_count],
                                     '.jpg')
    valid_missing_files += copy_files(label_dir, valid_label_dir,
                                      image_filenames[train_count:train_count + valid_count], '.txt')

    test_missing_files = copy_files(image_dir, test_image_dir, image_filenames[train_count + valid_count:], '.jpg')
    test_missing_files += copy_files(label_dir, test_label_dir, image_filenames[train_count + valid_count:], '.txt')

    # Print the count of each dataset
    print(f"Train dataset count: {train_count}, Missing files: {train_missing_files}")
    print(f"Validation dataset count: {valid_count}, Missing files: {valid_missing_files}")
    print(f"Test dataset count: {test_count}, Missing files: {test_missing_files}")

name=glob.glob(os.path.join(xml_file,"*.xml"))
#print(name)
for i in name :
    name_id=os.path.basename(i)[:-4]
    f(name_id,xml_file)


# 使用例子
image_dir = r"F:\yolov8\ultralytics-main\ultralytics-main\breathe_data\divide\总数据加第四次部分数据\images"
label_dir = r"F:\yolov8\ultralytics-main\ultralytics-main\breathe_data\divide\总数据加第四次部分数据\label"
output_dir = r"F:\yolov8\ultralytics-main\ultralytics-main\breathe_data\divide\总数据加第四次部分数据"



split_and_copy_dataset(image_dir, label_dir, output_dir)

