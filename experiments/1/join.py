from PIL import Image
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
# 打开三张图片
image1 = Image.open(curr_path+'/Figure_1.png')
image2 = Image.open(curr_path+'/Figure_2.png')
image3 = Image.open(curr_path+'/Figure_3.png')

# 确保所有图片都是相同大小的正方形
width, height = image1.size
size = (width, height)

# 创建一个新的大图
new_image = Image.new('RGB', (3 * width-60, height))

# 将三张图片拼接到一起
new_image.paste(image1, (0, 0))
new_image.paste(image2, (width-20, 0))
new_image.paste(image3, (2 * (width-20), 0))

# 保存或显示新的大图
new_image.show()
new_image.save('combined_image.png')