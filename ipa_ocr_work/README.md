# IPA OCR识别工作目录

## 任务状态

**状态**: 已完成第123页，识别结果已合并到Excel

## 工作流程

### 整体流程

```
PDF页面 → PaddleOCR定位文本框 → 裁剪文本框图像 → Calamari OCR识别IPA → 合并到Excel
```

### 详细步骤

#### 1. 使用PaddleOCR定位文本框

```bash
conda activate paddle_gpu2
python e:\my_pro\ipa_ocr_work\scripts\extract_all_ipa_gpu2.py
```

- 输入：`e:\my_pro\ipa_ocr_work\data\shaoxing_123-351.pdf`
- 输出：`e:\my_pro\ipa_ocr_work\images\ipa_candidates_gpu2\ocr_results.txt`
- 输出图像：`e:\my_pro\ipa_ocr_work\images\ipa_candidates_gpu2\ipa_candidate_*.png`

#### 2. 使用Calamari OCR识别IPA

```bash
conda activate calamari_tf2
python e:\my_pro\ipa_ocr_work\scripts\predict_ipa_batch.py
```

- 输入：`e:\my_pro\ipa_ocr_work\images\ipa_candidates_gpu2\ipa_candidate_*.png`
- 输出：`e:\my_pro\ipa_ocr_work\results\batch_recognition_results.txt`

#### 3. 合并到Excel

```bash
python e:\my_pro\ipa_ocr_work\scripts\merge_ipa_to_excel_gpu.py
```

- 输入1：`e:\my_pro\ipa_ocr_work\images\ipa_candidates_gpu2\ocr_results.txt`（PaddleOCR结果）
- 输入2：`e:\my_pro\ipa_ocr_work\results\batch_recognition_results.txt`（Calamari IPA结果）
- 输出：`e:\my_pro\result_all_converted.xlsx`（IPA识别列）

## Conda环境

| 环境名 | 说明 |
|------|------|
| `paddle_gpu2` | PaddleOCR GPU版本环境（paddlepaddle-gpu 3.0.0 + CUDA 12.6 + paddleocr 2.7.3） |
| `calamari_tf2` | Calamari OCR环境（TensorFlow 2.15.0，用于IPA识别） |

### 创建paddle_gpu2环境

```bash
conda create -n paddle_gpu2 python=3.10 -y
conda activate paddle_gpu2
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install paddleocr==2.7.3
pip install "numpy<2.0"
pip install "opencv-python-headless<4.10"
```

### 创建calamari_tf2环境

```bash
conda create -n calamari_tf2 python=3.10 -y
conda activate calamari_tf2
pip install tensorflow==2.15.0
pip install protobuf==3.20.3
```

## 目录结构

```
ipa_ocr_work/
├── data/                    # 源数据文件
│   └── shaoxing_123-351.pdf # 绍兴方言PDF
├── images/                  # 裁剪的图像文件
│   └── ipa_candidates_gpu2/ # GPU模式裁剪的文本框
│       ├── ocr_results.txt  # PaddleOCR识别结果
│       └── ipa_candidate_*.png # 裁剪的文本框图像
├── scripts/                 # Python脚本
│   ├── extract_all_ipa_gpu2.py    # PaddleOCR批量提取
│   ├── predict_ipa_batch.py      # Calamari批量识别
│   ├── merge_ipa_to_excel_gpu.py  # 合并到Excel
│   └── predict_ipa.py             # 单个IPA识别
└── results/                 # 输出结果
    └── batch_recognition_results.txt # Calamari识别结果
```

## 核心脚本说明

### extract_all_ipa_gpu2.py

使用GPU模式的PaddleOCR从PDF中提取所有文本框。

```python
from paddleocr import PaddleOCR
import fitz

ocr = PaddleOCR(lang='ch')
doc = fitz.open('data/shaoxing_123-351.pdf')
page = doc[page_num]  # 页码从0开始
img = page.get_pixmap(matrix=fitz.Matrix(2, 2))
result = ocr.ocr(np.array(img))
```

### predict_ipa_batch.py

使用Calamari OCR批量识别IPA字符。

```python
import tensorflow as tf

model = tf.saved_model.load('model/calamari/fold_0.ckpt')
chars = open('results/charset_list.txt').read().split()

logits = model(images)['root_3']
# CTC解码...
```

### merge_ipa_to_excel_gpu.py

将IPA识别结果合并到Excel。

```python
# 1. 读取PaddleOCR结果获取汉字词条
# 2. 读取Calamari OCR结果获取IPA
# 3. 根据索引匹配汉字和IPA
# 4. 写入Excel的IPA识别列
```

## IPA识别结果示例（第123页）

| 汉字 | IPA识别 |
|------|---------|
| 月亮 | lɦiøʔliaŋkȵiøʔliaŋ |
| 星 | ɕiŋ |
| 风 | lfoŋ |
| 七簇星 | tɕʰiʔtsʰoʔɕiŋ |

## 相关资源

- **OCR-IPA项目**: https://github.com/TUM-NLP/ocr-ipa
- **字符集**: `results/charset_list.txt`（388个字符，包括91个IPA字母）
- **网络代理**: 7897端口

## 已知问题与解决

### 1. Calamari模型输出头

**问题**: 识别结果中大量出现私有字符𝼆（U+1DF06）

**解决**: 使用 `root_3` 输出而非 `root`

### 2. TensorFlow版本

**问题**: SavedModel saved prior to TF 2.5 detected

**解决**: 使用TF 2.15.0 + protobuf 3.20.3

### 3. PaddleOCR GPU模式CUDA版本

**问题**: cudnn64_8.dll配置错误

**解决**: 使用paddlepaddle-gpu==3.0.0 + CUDA 12.6

## 进度记录

| 页码 | 状态 | IPA识别数 |
|------|------|----------|
| 123 | ✅已完成 | 24 |
| 124 | ✅已完成 | 22 |
| 125 | ✅已完成 | 25 |
| 126 | ✅已完成 | 17 |
| 127 | ✅已完成 | 23 |
| 128-130 | ✅已完成 | 23/25/26 |
| 131-351 | 🔄待处理 | - |

## 自动化脚本

使用 `process_all_pages.py` 可以自动处理所有页面：

```bash
conda activate paddle_gpu2
python e:\my_pro\ipa_ocr_work\scripts\process_all_pages.py 128 351
```

参数说明：
- 第一个参数：起始页码（默认123）
- 第二个参数：结束页码（默认351）

注意：需要确保 `paddle_gpu2` 环境中已安装 tensorflow（用于Calamari OCR）
