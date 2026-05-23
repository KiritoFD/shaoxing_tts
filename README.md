# 绍兴方言PDF拼音提取工具

## 项目概述

本项目用于从绍兴方言PDF（shaoxing.pdf）中提取词汇，并通过吴语学堂网站查询每个汉字的拼音，输出到Excel文件。

## 文件说明

### 核心文件（根目录）

| 文件 | 说明 |
|------|------|
| `process_pdf_ocr.py` | **核心脚本** - PDF文本提取与拼音查询主程序 |
| `merge_results.py` | 合并多个页面结果到 `result_all.xlsx` |
| `transform_nums_final.py` | 拼音数字转换脚本（将数字替换为指定值） |
| `pinyin_cache.json` | 拼音查询缓存文件 |
| `result_all.xlsx` | 合并后的原始数据（13079行） |
| `result_all_converted.xlsx` | 数字转换后的数据 |
| `300.xlsx` | 附加数据文件 |
| `shaoxing.pdf` | 绍兴方言PDF源文件 |
| `shaoxing_123-351.pdf` | 绍兴方言PDF（第123-351页） |
| `README.md` | 项目说明文档 |

### `test_results/` - 处理结果

| 文件 | 说明 |
|------|------|
| `result_all_page123.xlsx` | 第123页处理结果 |
| `result_all_page124.xlsx` | 第124页处理结果 |
| `result_all_page125.xlsx` | 第125页处理结果 |
| `result_all_page126.xlsx` | 第126页处理结果 |
| `result_page123.xlsx` | 第123页原始结果 |
| `test_page123.xlsx` | 测试结果 |
| `test_page123_paddle.xlsx` | Paddle测试结果 |

### `debug_images/` - 调试文件

| 文件 | 说明 |
|------|------|
| `debug_paddle.py` | PaddleOCR 调试脚本 |
| `debug_slow.py` | 性能诊断脚本 |
| `debug_*.png` | OCR识别调试截图 |

### `old_scripts/` - 历史脚本（已废弃）

| 文件 | 说明 |
|------|------|
| `screenshot_script.py` | 早期截图脚本（已被process_pdf_ocr.py整合） |
| `screenshot_*.py` | 各种截图方式测试脚本 |
| `run_test.py` | 辅助测试脚本 |

### Conda环境

| 环境名 | 说明 |
|------|------|
| `paddle_gpu2` | PaddleOCR GPU版本环境（paddlepaddle-gpu 3.0.0 + CUDA 12.6 + paddleocr 2.7.3） |
| `calamari_tf2` | Calamari OCR环境（TensorFlow 2.15.0，用于IPA识别） |

#### 创建paddle_gpu2环境（CUDA 12.6）

```bash
conda create -n paddle_gpu2 python=3.10 -y
conda activate paddle_gpu2
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install paddleocr==2.7.3
pip install "numpy<2.0"
pip install "opencv-python-headless<4.10"
```

#### 使用GPU模式运行PaddleOCR

```python
import paddle
paddle.device.set_device('gpu:0')
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='ch')
```

#### 性能对比

| 模式 | 检测时间 | 识别时间 | 总时间 |
|------|---------|---------|--------|
| GPU (RTX 4060) | 0.30秒 | 0.27秒 | 0.57秒 |
| CPU | 0.36秒 | 1.14秒 | 1.50秒 |

#### PaddleOCR API格式

**请求示例：**
```python
import paddle
paddle.device.set_device('gpu:0')
from paddleocr import PaddleOCR
import numpy as np

# 初始化
ocr = PaddleOCR(lang='ch')

# 识别图像（numpy数组或图片路径）
result = ocr.ocr(img)  # img可以是numpy数组或图片路径
```

**返回格式（2.7.3版本）：**
```python
# result是一个列表，每个元素是一页的结果
result = [
    [  # 第一页
        [  # 第一个文本框
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # 四个角点坐标（2x分辨率）
            ('识别文本', 置信度)  # 识别结果和置信度
        ],
        [  # 第二个文本框
            [[339, 226], [1014, 244], [1012, 319], [337, 301]],
            ('月亮io？lia，nio？lian', 0.77)
        ],
        ...
    ]
]

# 访问数据
for line in result:
    for word in line:
        box = word[0]      # 坐标
        text = word[1][0]  # 文本
        score = word[1][1] # 置信度
```

**注意事项：**
- 坐标是2x分辨率（PDF渲染时使用了2x缩放）
- 需要处理numpy版本兼容问题（降级到1.26.4）
- GPU模式需要CUDA 12.6驱动支持

---

## IPA OCR识别

IPA OCR识别工作已移至子目录，详见 [ipa_ocr_work/README.md](ipa_ocr_work/README.md)

### 快速开始

```bash
# 1. 激活环境
conda activate paddle_gpu2  # PaddleOCR
conda activate calamari_tf2 # Calamari OCR

# 2. 提取文本框
python e:\my_pro\ipa_ocr_work\scripts\extract_all_ipa_gpu2.py

# 3. 识别IPA
python e:\my_pro\ipa_ocr_work\scripts\predict_ipa_batch.py

# 4. 合并到Excel
python e:\my_pro\ipa_ocr_work\scripts\merge_ipa_to_excel_gpu.py
```

### 进度记录

| 页码 | 状态 | IPA识别数 |
|------|------|----------|
| 123 | ✅已完成 | 24 |
| 124 | ✅已完成 | 22 |
| 125 | ✅已完成 | 25 |
| 126 | ✅已完成 | 17 |
| 127 | ✅已完成 | 23 |
| 128-351 | 🔄待处理 | - |

**自动化处理**: 使用 `python e:\my_pro\ipa_ocr_work\scripts\process_all_pages.py 128 351` 可自动处理后续页面

---

### 1. process_pdf_ocr.py（核心脚本）

**功能：** PDF文本提取与拼音查询的主程序

**主要流程：**
1. 使用 fitz (PyMuPDF) 将PDF页面转换为图片
2. 使用 PaddleOCR 识别图片中的文字
3. 解析文本，提取每行开头的汉字
4. 通过 Selenium 访问吴语学堂网站（https://www.wugniu.com/search）查询拼音
5. 截图并使用 OCR 识别拼音
6. 结果保存到 Excel 文件

**关键函数：**
- `extract_pdf_text()`: 使用PaddleOCR从PDF页面提取文字
- `parse_chinese_before_letters()`: 解析汉字开头的条目
- `get_pinyin_screenshot()`: 查询单个汉字的拼音
- `fix_pinyin_char()`: 修正OCR识别错误的拼音（l→1, s→5, b→6）
- `process_page()`: 处理单个页面
- `process_all_pages()`: 批量处理页面
- `save_to_xlsx()`: 保存结果到Excel

**输出格式（Excel列）：**
- 汉字：提取的汉字词组
- 拼音：主要发音（数字结尾）
- 其他读法：多音字的其它发音
- 标记：错误标记（&表示查询失败）
- 后续汉字：行尾的汉字
