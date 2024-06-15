import paddlehub as hub
import time
import cv2

def word_detect(img):
    ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)# mkldnn加速仅在CPU下有效
    start_time = time.time()
    result = ocr.recognize_text(images=[cv2.imread(img)],output_dir='ocr_result', visualization=True)
    end_time = time.time()
    spend_time = end_time-start_time
    print(spend_time)

    return result

r = word_detect(r"C:\Users\78771\Desktop\20240321152325.png")


data = r[0]['data']
print(data)
text_data=[]
first_chinese_index = -1
for item in data:
        if '失败' in item['text']:
            text_data.append(item['text'])

for i, char in enumerate(text_data[0]):
    if '\u4e00' <= char <= '\u9fff':
        first_chinese_index = i
        break

print(text_data)
text = ['(1）电池使用中']
first_chinese_index = -1

# 找到第一个汉字的索引
for i, char in enumerate(text[0]):
    if '\u4e00' <= char <= '\u9fff':
        first_chinese_index = i
        break
# 提取第一个汉字及其后面的所有信息
if first_chinese_index != -1:
    chinese_and_after = text[0][first_chinese_index:]
    text_data.append(chinese_and_after)
else:
    print("未找到汉字")
"""fail_texts = []
for i, entry in enumerate(sorted_data):
    if entry['text'] == '失败' and i + 1 < len(sorted_data):
        if i==0:
            fail_texts.append('涡轮自检失败')
        elif i==2:
            fail_texts.append('02流量传感器自检失败')
        elif i==4:
            fail_texts.append('吸气流量传感器自检失败')
        elif i==6:
            fail_texts.append('呼气流量传感器自检失败')
        elif i==8:
            fail_texts.append('压力传感器自检失败')
        elif i==10:
            fail_texts.append('呼气阀测试自检失败')
        elif i==12:
            fail_texts.append('安全阀测试自检失败')
        elif i==14:
            fail_texts.append('泄漏量自检失败')
        elif i==16:
            fail_texts.append('顺应性自检失败')
        elif i==18:
            fail_texts.append('管路阻力自检失败')
        elif i==20:
            fail_texts.append('02传感器自检失败')"""

#print(fail_texts)
# 打印排序后的结果

# or 传递文件地址调用
# result = ocr.recognize_text(paths=[img_path])
